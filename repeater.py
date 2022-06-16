from FSR_model import FSR_CV
from nn_config import NN_config
import time
import sys
from contextlib import contextmanager
import sys, os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

config_org = NN_config()

config_mod_list = {}
'''
config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'freesolv.csv'
config['dataset_type'] = 'regression'
config['metrics']      = ['rmse', 'mae']
config['metric']       = 'rmse'
config['minimize_score'] = True
config['features_filename'] = 'features/freesolv_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'delaney.csv'
config['dataset_type'] = 'regression'
config['metrics']      = ['rmse', 'mae']
config['metric']       = 'rmse'
config['minimize_score'] = True
config['features_filename'] = 'features/delaney_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = ['lipo'] #list
config['file_name'] = 'lipo.csv'    
config['dataset_type'] = 'regression'
config['metrics']      = ['rmse', 'mae']
config['metric']       = 'rmse'
config['minimize_score'] = True
config['features_filename'] = 'features/lipo_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['mol']  #list
config['target_col'] = ['Class']    #list
config['file_name'] = 'bace.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/bace_rdkit.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = ["p_np"] #list
config['file_name'] = 'bbbp_ch.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/bbbp_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'ecoli_inhibition.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/ecoli_inhibition_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['Peptide_smiles']  #list
config['target_col'] = ['cleavage'] #list
config['file_name'] = 'impensData_with_smiles.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/impensData_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['Peptide_smiles']  #list
config['target_col'] = ['cleavage'] #list
config['file_name'] = '746Data_with_smiles.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/746Data_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['Peptide_smiles']  #list
config['target_col'] = ['cleavage'] #list
config['file_name'] = '1625Data_with_smiles.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/1625Data_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['Peptide_smiles']  #list
config['target_col'] = ['cleavage'] #list
config['file_name'] = 'schillingData_with_smiles.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/schillingData_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config
'''
config = {}
config['smiles_col'] = ['SMILES']  #list
config['target_col'] = None #list
config['file_name'] = 'mproxchem_ch.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/mproxchem_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'clintox.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/clintox_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'sider.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/sider_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'tox21_ch.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = False
config['features_filename'] = 'features/tox21_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'qm8.csv'
config['dataset_type'] = 'regression'
config['metrics']      = ['rmse', 'mae']
config['metric']       = 'rmse'
config['minimize_score'] = False
config['features_filename'] = 'features/qm8_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config
'''
config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'qm9.csv'
config['dataset_type'] = 'regression'
config['metrics']      = ['rmse', 'mae']
config['metric']       = 'rmse'
config['minimize_score'] = True
config['features_filename'] = 'features/qm9_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config

config = {}
config['smiles_col'] = ['smiles']  #list
config['target_col'] = None #list
config['file_name'] = 'toxcast_ch.csv'
config['dataset_type'] = 'classification'
config['metrics']      = ['auc', 'prc-auc']
config['metric']       = 'auc'
config['minimize_score'] = True
config['features_filename'] = 'features/toxcast_features.npy'
config['savedir_name']  = config['file_name'].split('.')[0]
config_mod_list[config['file_name'].split('.')[0]] = config
#'''
for i in config_mod_list.keys():
    start_time = time.time()
    config_changes = config_mod_list[i]
    for j in config_changes.keys():
        config_org[j] = config_changes[j]
    with suppress_stdout():

        mean_test_score, std_test_score, mean_val_score, std_val_score, all_test_scores_dict = FSR_CV(config_org)
        for metric, scores in all_test_scores_dict.items():
            # average score for each model across tasks
            avg_scores = np.nanmean(scores, axis=1)
            mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
            print(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')
    time_diff = (time.time() - start_time)/60
    print(f"--- {i} finished in {time_diff} minutes ---")