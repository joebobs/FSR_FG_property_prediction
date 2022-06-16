
# Required libraries
import numpy as np 
import json
import csv
import torch 
from torch.optim import Adam

import os
from tqdm import trange

from collections import defaultdict


from chemprop.data import split_data
from chemprop.data.data import MoleculeDataLoader
from chemprop.train.evaluate import evaluate, evaluate_predictions
from chemprop.train.predict import predict
from chemprop.constants import MODEL_FILE_NAME
from chemprop.nn_utils import param_count
from chemprop.nn_utils import NoamLR
from chemprop.utils import makedirs
from chemprop.data.utils import validate_dataset_type

from train import train
from molecule_model import MoleculeModel
from get_data import get_data, get_task_names


main_dir = r'/scratch/scratch6/joebobby'

savedir_name = 'dmpnn_delaney'
save_dir = os.path.join(main_dir, 'DMPNN_Results', savedir_name)
os.makedirs(save_dir, exist_ok = True)

# Arguments
data_path = os.path.join(main_dir, 'Data', 'delaney.csv')
features_path = None #os.path.join(main_data_path, 'amu_sars_cov_2_in_vitro.csv')
features_generator = None
if features_path is None and features_generator is None:
    use_input_features = False
    features_scaling = False
else:
    use_input_features = True
    features_scaling = True

features_only = False

smiles_columns = ['smiles']
target_columns = None
if target_columns is None:
    task_names = get_task_names(path=data_path, smiles_columns=smiles_columns) 
    target_columns = task_names
else:
    task_names = target_columns

num_tasks = len(task_names)

dataset_type = 'regression'

if dataset_type =='classification':
    minimize_score = False 
    metrics = ["auc", "prc-auc"]
    mainmetric = "auc"
else:
    minimize_score = True 
    metrics = ["rmse", "mae"]
    mainmetric = "rmse"

init_seed = 0
pytorch_seed = 0
split_type = 'random'
split_sizes = (0.8, 0.1, 0.1)
batch_size = 75
num_workers = 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 2300
dropout = 0.2
depth = 3
ffn_hidden_size = 2300
ffn_num_layers = 3
init_lr = 0.0001
warmup_epochs = 2.0
max_lr = 0.001
final_lr = 0.0001


epochs = 20
ensemble_size = 1
num_folds = 10

data = get_data(
                path=data_path, 
                smiles_columns=smiles_columns, 
                features_path=features_path,
                features_generator=features_generator,
                target_columns=target_columns, 
                skip_none_targets=True
               )

validate_dataset_type(data, dataset_type=dataset_type)
features_size = data.features_size()


test_scores_dict = defaultdict(list)
for fold_num in range(num_folds):
    
    print(f'\nFold {fold_num}')
    seed = init_seed + fold_num
    
    save_dir_fold = os.path.join(save_dir, f'fold_{fold_num}')
    makedirs(save_dir_fold)
    
    # Training a model on the set seed
    torch.manual_seed(pytorch_seed)
    data.reset_features_and_targets()
    
    # Splitting the dataset
    print(f"Splitting the dataset based on {split_type} Method....")
    train_data, val_data, test_data = split_data(
                                                data=data, 
                                                split_type=split_type, 
                                                sizes=split_sizes, 
                                                seed=seed,  
                                                )
    train_data_size = len(train_data)
    features_size = train_data.features_size()
    
    if not fold_num:
        print(f'Total size = {len(data):,} | train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')
    
    # Feature Scaling
    if features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if dataset_type == 'regression':
        print('\nFitting scaler')
        scaler = train_data.normalize_targets()
    else:
        scaler = None

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), num_tasks))

    # Creating Data Loaders
    train_data_loader = MoleculeDataLoader(
                                            dataset=train_data,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            class_balance=False,
                                            shuffle=True,
                                            seed=seed
                                        )
    val_data_loader = MoleculeDataLoader(
                                            dataset=val_data,
                                            batch_size=batch_size,
                                            num_workers=num_workers
                                        )
    test_data_loader = MoleculeDataLoader(
                                            dataset=test_data,
                                            batch_size=batch_size,
                                            num_workers=num_workers
                                        )
    
    # Training
    for model_idx in range(ensemble_size):
        
        if f'model_{model_idx}' in os.listdir(save_dir_fold):
            print(f"Model_{model_idx} Found")
            save_dir_temp = os.path.join(save_dir_fold, f'model_{model_idx}')
            path  = os.path.join(save_dir_temp, MODEL_FILE_NAME)
        else:
            save_dir_temp = os.path.join(save_dir_fold, f'model_{model_idx}')
            makedirs(save_dir_temp)
            path  = os.path.join(save_dir_temp, MODEL_FILE_NAME)
            
            print(f'\nBuilding model {model_idx}')
            model = MoleculeModel(device=device,
                                  hidden_size=hidden_size, 
                                  depth=depth,
                                  dropout=dropout, 
                                  ffn_hidden_size=ffn_hidden_size, 
                                  ffn_num_layers=ffn_num_layers,
                                  features_only=features_only,
                                  use_input_features=use_input_features,
                                  features_size=features_size,
                                  num_tasks=num_tasks,
                                  dataset_type=dataset_type)

            print(model)
            print(f'Number of parameters = {param_count(model):,}')
            model = model.to(device)

            # optimizer
            params = [{'params': model.parameters(), 'lr': init_lr, 'weight_decay': 0}]
            optimizer = Adam(params=params)

            # scheduler
            scheduler = NoamLR(optimizer=optimizer,
                                warmup_epochs=[warmup_epochs],
                                total_epochs=[epochs],
                                steps_per_epoch=train_data_size//batch_size,
                                init_lr=[init_lr],
                                max_lr=[max_lr],
                                final_lr=[final_lr])
            
            if dataset_type == 'classification':
                loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
            if dataset_type == 'regression':
                loss_func = torch.nn.MSELoss(reduction='none')

            best_score = float('inf') if minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0
            for epoch in trange(epochs):
                print(f'\nEpoch {epoch}')

                n_iter = train(
                                model=model,
                                data_loader=train_data_loader,
                                loss_func=loss_func,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                n_iter=n_iter,
                                dataset_type=dataset_type,
                                batch_size=batch_size
                            )

                val_scores = evaluate(
                            model=model,
                            data_loader=val_data_loader,
                            num_tasks=num_tasks,
                            metrics=metrics,
                            scaler=scaler,
                            dataset_type=dataset_type
                        )

                for metric, scores in val_scores.items():
                    # Average validation score
                    avg_val_score = np.nanmean(scores)
                    print(f'Validation {metric} = {avg_val_score:.6f}')

                avg_val_score = np.nanmean(val_scores[mainmetric])

                if minimize_score and avg_val_score < best_score or not minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    state = {
                            "state_dict": model.state_dict(),
                             "score": best_score,
                             "epoch": best_epoch,
                             "features_scaler": {
                                                    'means': features_scaler.means,
                                                    'stds': features_scaler.stds
                                                } if features_scaler is not None else None,
                            'data_scaler': {
                                                'means': scaler.means,
                                                'stds': scaler.stds
                                            } if scaler is not None else None
                            }
                    torch.save(state, path)

            print(f'Model {model_idx} best validation {mainmetric} = {best_score:.6f} on epoch {best_epoch}\n')
        
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        loaded_state_dict = loaded_state['state_dict']

        loaded_model = MoleculeModel(device=device,
                              hidden_size=hidden_size, 
                              depth=depth,
                              dropout=dropout, 
                              ffn_hidden_size=ffn_hidden_size, 
                              ffn_num_layers=ffn_num_layers,
                              features_only=features_only,
                              use_input_features=use_input_features,
                              features_size=features_size,
                              num_tasks=num_tasks,
                              dataset_type=dataset_type)

        model_state_dict = loaded_model.state_dict()

        pretrained_state_dict = {}
        for param_name in loaded_state_dict.keys():

            if param_name not in model_state_dict:
                print(f'Warning: Pretrained parameter "{param_name}" cannot be found in model parameters.')
            elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
                print(f'Warning: Pretrained parameter "{param_name}" '
                        f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                        f'model parameter of shape {model_state_dict[param_name].shape}.')
            else:
                print(f'Loading pretrained parameter "{param_name}".')
                pretrained_state_dict[param_name] = loaded_state_dict[param_name]

        # Load pretrained weights
        model_state_dict.update(pretrained_state_dict)
        loaded_model.load_state_dict(model_state_dict)

        loaded_model = loaded_model.to(device)

        # test predictions
        test_preds = predict(
                model=loaded_model,
                data_loader=test_data_loader,
                scaler=scaler
            )
        test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=num_tasks,
                    metrics=metrics,
                    dataset_type=dataset_type,
                )
        # Average test score
        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            print(f'Model_{model_idx} - test {metric} = {avg_test_score:.6f}')

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds/ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type
    )

    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(scores)
        print(f'fold_{fold_num} - Ensemble test {metric} = {avg_ensemble_test_score:.6f}')
        test_scores_dict[metric].append(scores)
    
    # Save scores
    with open(os.path.join(save_dir_fold, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

test_scores_dict = dict(test_scores_dict)

# Convert scores to numpy arrays
for metric, scores in test_scores_dict.items():
    test_scores_dict[metric] = np.array(scores)

# Report results
print(f'\n{num_folds}-fold cross validation')

# Report scores for each fold
for fold_num in range(num_folds):
    for metric, scores in test_scores_dict.items():
        print(f'\tSeed {init_seed + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')

# Report scores across folds
for metric, scores in test_scores_dict.items():
    avg_scores = np.nanmean(scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    print(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')  

# Save scores
with open(os.path.join(save_dir, 'test_scores.csv'), 'w') as f:
    writer = csv.writer(f)

    header = ['Task']
    for metric in metrics:
        header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                    [f'Fold {i} {metric}' for i in range(num_folds)]
    writer.writerow(header)

    for task_num, task_name in enumerate(task_names):
        row = [task_name]
        for metric, scores in test_scores_dict.items():
            task_scores = scores[:, task_num]
            mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
            row += [mean, std] + task_scores.tolist()
        writer.writerow(row)
