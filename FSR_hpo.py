
import json
import os

import numpy as np

from hyperopt import fmin, hp, tpe

from nn_config import NN_config
from nn_model import NN_Large_Predictor

from FSR_model import FSR_CV

main_dir = r'/scratch/scratch6/joebobby'
#main_dir = './'

# parameters configuration
config = NN_config()

SPACE = {
    'predict_dim' : hp.qloguniform('predict_dim', low=np.log(200), high=np.log(2000), q=1),
    'ffn_num_layers' : hp.quniform('ffn_num_layers', low=2, high=6, q=1),
    'dropout': hp.quniform('dropout', low=0.1, high=0.5, q=0.1),
    'LR': hp.qloguniform('LR', low = np.log(1e-6), high = np.log(1e-4), q = 1e-7)
}
INT_KEYS = ['predict_dim', 'ffn_num_layers']
#INT_KEYS = ['predict_dim', 'ffn_num_layers']

# Run grid search
results = []

# Define hyperparameter optimization
def objective(hyperparams):
    # Convert hyperparams from float to int when necessary
    for key in INT_KEYS:
        print(key)
        hyperparams[key] = int(hyperparams[key])

    config['predict_dim'] = hyperparams['predict_dim']
    config['ffn_num_layers'] = hyperparams['ffn_num_layers']
    config['dropout']  = hyperparams['dropout']
    config['LR'] = hyperparams['LR']

    print(config)
    __, __, mean_score, std_score = FSR_CV(config)

    # Record results
    temp_model = NN_Large_Predictor(**config)
    num_params = sum(param.numel() for param in temp_model.parameters() if param.requires_grad)
    print(f'num params: {num_params:,}')
    print(f'{mean_score} +/- {std_score} {config["metric"]}')

    results.append({
        'mean_score': mean_score,
        'std_score': std_score,
        'hyperparams': hyperparams,
        'num_params': num_params
    })

    # Deal with nan
    if np.isnan(mean_score):
        if config['dataset_type'] == 'classification':
            mean_score = 0
        else:
            raise ValueError('Can\'t handle nan score for non-classification dataset.')

    return (1 if config['minimize_score'] else -1) * mean_score

num_iters = config['num_iters']
seed = config['seed']
fmin(objective, SPACE, algo=tpe.suggest, max_evals=num_iters, rstate=np.random.RandomState(seed))

# Report best result
results = [result for result in results if not np.isnan(result['mean_score'])]
best_result = min(results, key=lambda result: (1 if config["minimize_score"] else -1) * result['mean_score'])

print('\nbest')
print(best_result['hyperparams'])
print(f'\nnum params: {best_result["num_params"]:,}')
print(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {config["metric"]}\n')

# Training the FSR Model with best hyperparams

best_hyperparams = best_result['hyperparams']
config['predict_dim'] = best_hyperparams['predict_dim']
config['ffn_num_layers'] = best_hyperparams['ffn_num_layers']
config['dropout']  = best_hyperparams['dropout']
config['LR'] = best_hyperparams['LR']

__, __, __, __ = FSR_CV(config)


# Save best hyperparameter settings ans the config settings as JSON files
save_dir = os.path.join(main_dir, 'FSR_Results', config['savedir_name'])
hyperparams_save_path = os.path.join(save_dir, 'hyperparams.json')

config_save_path = os.path.join(save_dir, 'params.json')

with open(hyperparams_save_path, 'w') as f:
    json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)

with open(config_save_path, 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)
