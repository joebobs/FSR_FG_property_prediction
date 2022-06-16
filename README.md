# FSR_FG_property_prediction
created by Joe Bobby for study on molecular property prediction for dual degree project

### Running model
The FSR_model.py file can run the model using the parameters set on the nn_config.py file
The repeater.py file is capable of running the model on multiple datasets at once

### For pretraining:
The FSR_model_recon used the nn_config_recon file to produce partially pretrained models

### For feature generation:
Use the file feature_generator.ipynb, a jupyter notebook to produce features for datasets
The file also has the code to create the one hot encoder and the min-max_scaler used in another part of the code

A requirements.txt file and the complete data required will be uploaded shortly
The file paths are mostly absolute, so change that to your local system path before proceeding.
