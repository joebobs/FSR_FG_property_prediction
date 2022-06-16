def NN_config():
	
    config = {}
    config['seed'] = 0
    config['batch_size'] = 75
    config['num_folds']  = 5
    config['epochs']     = 50
    config['input_dim'] = 5372#2586#252 + 233 #402 #153 + 233
    config['LR'] = 8e-5
    config['split_sizes'] = (0.8, 0.1, 0.1)
    config['split_type'] = 'random'
    config['pretrain'] = None#'D:\\Joe\\Acads\\Sem8\\DDP\\Code files\\FSR\\Data\\Pretrained_files\\fsr_encoding.pt'
    #config['pretrain'] = None#r'/scratch/scratch6/joebobby/FSR_Results/chembl_recon_rdkit_fsr/fold_0/fsr_model_train_ckpt.pt'
    config['main_dir'] = r'/scratch/scratch6/joebobby'
    config['main_dir'] = r'.\\'

    config['encode_fc1_dim'] = 1000 # encoder fc1
    config['encode_fc2_dim'] = 200  # encoder fc2
    config['decode_fc1_dim'] = 1000  # decoder fc1
    config['decode_fc2_dim'] = config['input_dim']  # decoder reconstruction
    config['predict_dim'] = 1500 # for every layer
    config['reconstruction_coefficient'] = 1  # 1e-2
    config['predict_out_dim'] = 1 # predictor out
    config['dropout'] = 0.5
    config['ffn_num_layers'] = 4
 
    config['smiles_col'] = ['smiles']  #list
    config['target_col'] = None #list
    config['file_name'] = 'ecoli_inhibition.csv'

    config['dataset_type'] = 'classification'
    config['metrics']      = ['auc', 'prc-auc']
    config['metric']       = 'auc'
    config['minimize_score'] = False


    config['extra_features'] = True
    config['num_encoding_pretrain'] = None#'D:\\Joe\\Acads\\Sem8\\DDP\\Code files\\FSR\\Data\\Pretrained_files\\numerical_encoding.pt'
    config['cat_encoding_pretrain'] = 'D:\\Joe\\Acads\\Sem8\\DDP\\Code files\\FSR\\Data\\Pretrained_files\\categorical_encoding.pt'
    config['features_filename'] = 'features/ecoli_inhibition_features.npy' 
    #config['feature_categorical_columns'] = [16, 49, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 75, 80, 83, 89, 93, 94, 100, 102, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]
    #config['feature_categorical_columns'] = [49, 51, 52, 53, 54, 55, 56, 57, 58, 62, 64, 75, 80, 93, 94, 119, 124, 125, 129, 130, 131, 133, 139, 141, 143, 144, 148, 150, 155, 158, 187]
    config['feature_categorical_columns'] = [163, 102, 186, 151,  89, 145, 146, 164,  60, 147, 149, 195, 137,
       168, 170, 160, 175, 152, 156, 126, 167, 191, 193, 118, 142, 162,
       138, 188, 159, 182, 171, 181, 174, 192, 176, 177, 184, 153, 183,
       194, 190, 196, 173, 172, 127, 132, 165, 136, 166, 187, 198, 128,
       178, 117, 161, 189, 135, 185, 133, 120,  62, 134,  51, 144, 157,
       154,  54, 143, 148, 123, 122, 114, 169, 121, 179, 180,  55, 139,
        63, 119, 158, 140,  83,  52,  64, 131,  56,  53, 115, 116, 150,
       129, 155,  80,  16, 197, 125, 141, 100, 130, 124,  93,  58,  57,
        49,  50,  59,  61]
    config['feature_scaling'] = True
    config['cat_features_size'] = 0
    config['cat_encode_fc1_dim'] = 700 # encoder fc1
    config['cat_encode_fc2_dim'] = 100  # encoder fc2
    config['cat_decode_fc1_dim'] = 700  # decoder fc1
    config['cat_decode_fc2_dim'] = 0  # decoder reconstruction (currently 0)

    config['num_features_size'] = 200 - len(config['feature_categorical_columns']) ### 200 is the rdkit feature size in this version
    config['num_encode_fc1_dim'] = 70 # encoder fc1
    config['num_encode_fc2_dim'] = 30  # encoder fc2
    config['num_decode_fc1_dim'] = 70  # decoder fc1
    config['num_decode_fc2_dim'] = config['num_features_size']  # decoder reconstruction

    config['savedir_name']  = 'ecoli_inhibition'
    config['num_iters'] = 50 # no of hyperparameter sets for BHP

    return config   