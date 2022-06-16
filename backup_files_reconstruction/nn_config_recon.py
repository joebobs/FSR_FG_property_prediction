def NN_config():
	
    config = {}
    config['seed'] = 0
    config['batch_size'] = 75
    config['num_folds']  = 1
    config['epochs']     = 1
    config['input_dim'] = 2586#2786
    config['LR'] = 1e-3
    config['split_sizes'] = (0.8, 0.1, 0.1)
    config['split_type'] = 'random'

    config['encode_fc1_dim'] = 1000 # encoder fc1
    config['encode_fc2_dim'] = 400  # encoder fc2
    config['decode_fc1_dim'] = 1000  # decoder fc1
    config['decode_fc2_dim'] = config['input_dim']  # decoder reconstruction
    config['predict_dim'] = 512 # for every layer
    config['reconstruction_coefficient'] = 0.01  # 1e-2
    config['predict_out_dim'] = 1 # predictor out
    config['dropout'] = 0.1
    config['ffn_num_layers'] = 4
 
    config['smiles_col'] = ['canonical_smiles']  #list
    config['target_col'] = ['target'] #list
    config['file_name'] = 'chembl.csv'

    config['dataset_type'] = 'classification'
    config['metrics']      = ['auc', 'prc-auc']
    config['metric']       = 'auc'
    config['minimize_score'] = False


    config['extra_features'] = False
    config['features_filename'] = 'features/mproxchem_features.npy' 
    #config['feature_categorical_columns'] = [16, 49, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 75, 80, 83, 89, 93, 94, 100, 102, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]
    config['feature_categorical_columns'] = [49, 51, 52, 53, 54, 55, 56, 57, 58, 62, 64, 75, 80, 93, 94, 119, 124, 125, 129, 130, 131, 133, 139, 141, 143, 144, 148, 150, 155, 158, 187]
    config['feature_scaling'] = False
    config['features_size'] = 0
    config['feature_ffn_size'] = 0

    config['savedir_name']  = 'chembl_recon_rdkit_fsr'
    config['num_iters'] = 50 # no of hyperparameter sets for BHP

    return config
