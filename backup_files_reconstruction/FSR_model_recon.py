from collections import defaultdict
from random import Random
import os
import json
import csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import torch
import torch.nn.functional as F
from torch.utils import data
import copy

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from chemprop.utils import get_metric_func
from chemprop.data.scaler import StandardScaler
from chemprop.features import load_features

from get_data import get_task_names
from nn_config_recon import NN_config
from nn_model import NN_Large_Predictor
from nn_data_fg import supData

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def scaffold_split(data, split_ratio, seed):
    # Scaffold Splitting techniques
    train_size, val_size, test_size = split_ratio[0] * len(data), split_ratio[1] * len(data), split_ratio[2] * len(data)
    train_indices, val_indices, test_indices = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
    smiles_ls = data['SMILES'].tolist()
    
    scaffold_to_indices = defaultdict(set)
    for i, smile in enumerate(smiles_ls):
        mol = Chem.MolFromSmiles(smile)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = mol, includeChirality = False)
        scaffold_to_indices[scaffold].add(i)

    random = Random(seed)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    for index_set in index_sets:
        if len(train_indices) + len(index_set) <= train_size:
            train_indices += index_set
            train_scaffold_count += 1
        elif len(val_indices) + len(index_set) <= val_size:
            val_indices += index_set
            val_scaffold_count += 1
        else:
            test_indices += index_set
            test_scaffold_count += 1

    return train_indices, val_indices, test_indices

# predictions
def evaluate_predictions(y_label, y_preds, num_tasks, metrics, dataset_type):
    
    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(y_preds)):
            if (not np.isnan(y_label[j][i])):   # Skip those without targets
                valid_preds[i].append(y_preds[j][i])
                valid_targets[i].append(y_label[j][i])

    results = defaultdict(list)
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                print('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                print('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                for metric in metrics:
                    results[metric].append(float('nan'))
                continue

            if len(valid_targets[i]) == 0:
                continue
        
        if len(valid_targets[i]) == 0:
            continue

        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

    results = dict(results)

    return results


def test_model_nn(data_generator, model_nn, num_tasks, metrics_ls, dataset_type, scaler, extra_features=False):
    y_pred = []
    y_label = []

    model_nn.eval()
    for batch in data_generator:

        if extra_features:
            v_D, label, features_batch = batch
            features_batch = features_batch.float().to(device)
        else:
            v_D, label = batch
            features_batch=None
        v_D = v_D.float().to(device)
        
        _, score, _ = model_nn(v_D, features_batch) 
        
        label_ids = label.to('cpu').numpy()
        if dataset_type == 'classification':
            m = torch.nn.Sigmoid()
            logits = m(score).detach().cpu().numpy()
        else:
            logits = score.detach().cpu().numpy()
            logits = scaler.inverse_transform(logits)
            label_ids = scaler.inverse_transform(label_ids)
         
        y_label = y_label + label_ids.tolist()
        y_pred = y_pred + logits.tolist()

    results = evaluate_predictions(y_label, y_pred, num_tasks, metrics_ls, dataset_type) 
        
    return results

def load_checkpoint(path, config):
    """
    Loads a model checkpoint.
    
    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = state['state_dict']

    # Build model
    model_nn = NN_Large_Predictor(**config)
    model_state_dict = model_nn.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Load pretrained parameter, skipping unmatched parameters
        if loaded_param_name not in model_state_dict:
            print(f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.')
        elif model_state_dict[loaded_param_name].shape != loaded_state_dict[loaded_param_name].shape:
            print(f'Warning: Pretrained parameter "{loaded_param_name}" '
                 f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[loaded_param_name].shape}.')
        else:
            print(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[loaded_param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model_nn.load_state_dict(model_state_dict)

    if use_cuda:
        print('Moving model to cuda')
    model_nn = model_nn.to(device)

    return model_nn


def FSR_CV(config):
    
    smiles_col = config['smiles_col']
    target_col = config['target_col']
    split_type = config['split_type']
    dataset_type = config['dataset_type']

    init_seed = 0
    
    BATCH_SIZE = config['batch_size']
    sizes = config['split_sizes']


    metrics_ls = config['metrics']
    deciding_metric = config['metric']

    print('\n--- Data Loading & Preparation ---')

    main_dir = r'/scratch/scratch6/joebobby'
    main_dir = '.\\'
    data_dir = os.path.join(main_dir, 'Data')
    data_path =  os.path.join(data_dir, config['file_name'])

    save_dir = os.path.join(main_dir, 'FSR_Results', config['savedir_name'])
    os.makedirs(save_dir, exist_ok = True)

    if target_col is None:
        target_col = get_task_names(path=data_path, smiles_columns=smiles_col)
        config['predict_out_dim'] = len(target_col)
    
    num_tasks  = config['predict_out_dim']

    if config['extra_features']:
        features_data = []
        features_path = os.path.join(data_dir, config['features_filename'])
        features_data.append(load_features(features_path))
        features_data = np.concatenate(features_data, axis=1)
        if not config['feature_categorical_columns'] == None:
            categorical_idx = config['feature_categorical_columns']
            numeric = features_data[:,[not (i in categorical_idx) for i in range(features_data.shape[1])]]
            categorical = features_data[:,[(i in categorical_idx) for i in range(features_data.shape[1])]]
            categorical = OneHotEncoder().fit_transform(categorical)
            #features_data = np.concatenate([numeric, categorical.toarray()], axis = 1)
            features_data = categorical.toarray()
        __, config['features_size'] = features_data.shape
        config['input_dim'] += categorical.shape[1]
        config['decode_fc2_dim'] = config['input_dim']

    else:
        features_data = None

    data_df = pd.read_csv(data_path, usecols=smiles_col+target_col)
    data_df.rename(columns = {smiles_col[0]: "SMILES"}, inplace = True)

    # Run training on different random seeds for each fold
    all_test_scores_dict = defaultdict(list)
    all_val_scores_dict = defaultdict(list)
    for fold_ind in range(config['num_folds']):

        print(f'\nFold {fold_ind}\n')

        fold_seed = init_seed + fold_ind
        save_dir_fold = os.path.join(save_dir, f'fold_{fold_ind}')
        os.makedirs(save_dir_fold, exist_ok = True)
        model_save_path = os.path.join(save_dir_fold, 'fsr_model_train_ckpt.pt')
        
        torch.manual_seed(0)
        
        # Split data
        print(f'Splitting data with seed {fold_seed}')
        if split_type == 'scaffold_balanced':
            print('Scaffold Splitting\n')
            train_indices, val_indices, test_indices = scaffold_split(data_df, sizes, fold_seed)
        else:
            print('Random Splitting\n')
            random = Random(fold_seed)
            indices = list(range(len(data_df)))
            random.shuffle(indices)

            train_size = int(sizes[0] * len(data_df))
            train_val_size = int((sizes[0] + sizes[1]) * len(data_df))

            train_indices = np.array(indices[:train_size])
            val_indices = np.array(indices[train_size:train_val_size])
            test_indices = np.array(indices[train_val_size:])


        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if dataset_type == 'regression':
            print('Fitting scaler for targets')
            train_df = data_df.iloc[train_indices, :][target_col]
            train_targets = train_df[target_col].values.tolist()
            scaler = StandardScaler().fit(train_targets)
            
        else:
            scaler = None

        if config['feature_scaling']:
            train_features = features_data[train_indices]
            features_scaler = MinMaxScaler()
            features_scaler.fit(train_features)
        else:
            features_scaler =None

        config['train_data_size'] = len(train_indices)

        print(f'Total size = {len(data_df):,} | '
          f'train size = {len(train_indices):,} | val size = {len(val_indices):,} | test size = {len(test_indices):,}')
        
        # create dataloaders
        partition_sup = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        targets_fin = data_df[target_col].values
        params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0}

        training_set = supData(partition_sup['train'], targets_fin, data_df, features_data, scaler, features_scaler)
        training_generator_sup = data.DataLoader(training_set, **params)

        validation_set = supData(partition_sup['val'], targets_fin, data_df, features_data, scaler, features_scaler)
        validation_generator_sup = data.DataLoader(validation_set, **params)

        testing_set = supData(partition_sup['test'], targets_fin, data_df, features_data, scaler, features_scaler)
        test_generator_sup = data.DataLoader(testing_set, **params)
        
        # Training the model

        loss_r_history = []
        loss_history = []
        
        # load model
        print(f'\nBuilding model')
        model_nn = NN_Large_Predictor(**config)
        print(model_nn)
        '''tofreeze_keys = ['predictor.1.weight', 'predictor.1.bias', 'predictor.4.weight', 'predictor.4.bias', 'predictor.7.weight', 'predictor.7.bias', 'predictor.10.weight', 'predictor.10.bias']
        for state_name in tofreeze_keys:
            for param in model_nn[state_name].parameters():
                param.requires_grad = False'''
        print(f"No. of trainable parameters:{sum(param.numel() for param in model_nn.parameters() if param.requires_grad)}\n")

        if use_cuda:
            print('Moving model to cuda')
        model_nn = model_nn.to(device)

        # optimizer
        opt = torch.optim.Adam(model_nn.parameters(), lr = config['LR'])

        best_score = float('inf') if config['minimize_score'] else -float('inf')
        model_best = copy.deepcopy(model_nn)
        best_epoch = 0
    
        print('--- Go for Training ---\n')
        
        for tr_epo in range(config['epochs']):
            for i, batch in enumerate(training_generator_sup):
                
                if config['extra_features']:
                    v_D, label, features_batch = batch
                    features_batch = features_batch.float().to(device)
                else:
                    v_D, label = batch
                    features_batch=None
                
                v_D = v_D.float().to(device)
                mask = torch.Tensor([[(not np.isnan(x)) for x in tb] for tb in label])
                targets = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in label])
                mask = mask.to(device)
                targets = targets.to(device)

                recon, score, __ = model_nn(v_D, features_batch)

                loss_r = config['reconstruction_coefficient'] * F.binary_cross_entropy(recon, v_D.float())
    
                total_loss = loss_r
                loss_r_history.append(loss_r.cpu().detach().numpy())
                loss_history.append(total_loss.cpu().detach().numpy())

                opt.zero_grad()
                total_loss.backward()
                opt.step()

                if(i % 20 == 0):
                    print('Training at Epoch ' + str(tr_epo) + ' iteration ' + str(i) + ', total loss is ' + '%.3f' % (total_loss.cpu().detach().numpy()) + ', recon loss is ' + '%.8f' %(loss_r.cpu().detach().numpy()) + ', avg: %.8f' %(np.mean(loss_r_history[-20:])))

            '''with torch.set_grad_enabled(False):
                val_results = test_model_nn(validation_generator_sup, model_nn, num_tasks, metrics_ls, dataset_type, scaler, config['extra_features'])
                for metric, scores in val_results.items():
                # Average validation score
                    avg_val_score = np.nanmean(scores)
                    print(f'Validation {metric} = {avg_val_score:.6f}')

                avg_val_score = np.nanmean(val_results[deciding_metric])
                if config['minimize_score'] and avg_val_score < best_score or \
                     not config['minimize_score'] and avg_val_score > best_score:
                    model_best = copy.deepcopy(model_nn)
                    best_score , best_epoch = avg_val_score, tr_epo
                    state = {
                                'state_dict': model_best.state_dict(),
                                'data_scaler': {
                                    'means': scaler.means,
                                    'stds': scaler.stds
                                } if scaler is not None else None,
                            }
                                'features_scaler': {
                                    'means': features_scaler.means,
                                    'stds': features_scaler.stds
                                } if features_scaler is not None else None,
                            }
                    '''
            model_best = copy.deepcopy(model_nn)
            state = {
                        'state_dict': model_best.state_dict(),
                        'data_scaler': {
                            'means': scaler.means,
                            'stds': scaler.stds
                        } if scaler is not None else None,
                    }
            torch.save(state, model_save_path)    
                
        print(f"\nBest Validation Score: {deciding_metric}  - {best_score} on epoch {best_epoch}\n")
        '''with torch.set_grad_enabled(False):
            
            model = load_checkpoint(model_save_path, config)
            test_results = test_model_nn(test_generator_sup, model, num_tasks, metrics_ls, dataset_type, scaler, config['extra_features']) 
            # for hyperparameter opitimization purpose
            validation_results = test_model_nn(validation_generator_sup, model, num_tasks, metrics_ls, dataset_type, scaler, config['extra_features'])

            for metric, scores in test_results.items():
                avg_test_score = np.nanmean(scores)
                print(f'Test {metric} = {avg_test_score:.6f}')
                all_test_scores_dict[metric].append(scores)
            
            for metric, scores in validation_results.items():
                all_val_scores_dict[metric].append(scores)
            
            # Save scores
            with open(os.path.join(save_dir_fold, 'test_scores.json'), 'w') as f:
                json.dump(test_results, f, indent=4, sort_keys=True)

    all_test_scores_dict = dict(all_test_scores_dict)
    all_val_scores_dict = dict(all_val_scores_dict)

    # Convert scores to numpy arrays
    for metric, scores in all_test_scores_dict.items():
        all_test_scores_dict[metric] = np.array(scores)
    
    for metric, scores in all_val_scores_dict.items():
        all_val_scores_dict[metric] = np.array(scores)

    # Report results
    print(f'{config["num_folds"]}-fold cross validation')

    # Report scores for each fold
    for fold_num in range(config["num_folds"]):
        for metric, scores in all_test_scores_dict.items():
            print(f'\tSeed {init_seed + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')

    # Report scores across folds
    for metric, scores in all_test_scores_dict.items():
        # average score for each model across tasks
        avg_scores = np.nanmean(scores, axis=1)
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        print(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

    # Save scores
    with open(os.path.join(save_dir, 'test_scores.csv'), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in metrics_ls:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(config['num_folds'])]
        writer.writerow(header)

        for task_num, task_name in enumerate(target_col):
            row = [task_name]
            for metric, scores in all_test_scores_dict.items():
                task_scores = scores[:, task_num]
                mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)
    
    # Determine mean and std score of deciding metric
    avg_val_scores = np.nanmean(all_val_scores_dict[deciding_metric], axis=1)
    mean_val_score, std_val_score = np.nanmean(avg_val_scores), np.nanstd(avg_val_scores)

    # Determine mean and std score of deciding metric
    avg_test_scores = np.nanmean(all_test_scores_dict[deciding_metric], axis=1)
    mean_test_score, std_test_score = np.nanmean(avg_test_scores), np.nanstd(avg_test_scores)'''

    return 0, 0, 0, 0#mean_test_score, std_test_score, mean_val_score, std_val_score


if __name__ == '__main__':
    # parameters configuration
    config = NN_config()
    __, __, __, __ = FSR_CV(config) 
    pass
