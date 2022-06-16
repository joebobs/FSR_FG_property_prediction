import numpy as np
import pandas as pd
import torch
from torch.utils import data

from subword_nmt.apply_bpe import BPE
import codecs


dataFolder = r'/scratch/scratch6/joebobby/Data'
#dataFolder = r'./Data'

vocab_path = dataFolder + '/codes_drug_chembl_1500.txt'
bpe_codes_fin = codecs.open(vocab_path)
bpe = BPE(bpe_codes_fin, merges=-1, separator='')

vocab_map = pd.read_csv(dataFolder + '/subword_units_map_drug_chembl_1500.csv')
idx2word = vocab_map['index'].values
words2idx = dict(zip(idx2word, range(0, len(idx2word))))
max_set = 30

def smiles2index(s1):
    # print(s1)
    t1 = bpe.process_line(s1).split() #split
    i1 = [words2idx[i] for i in t1] #index
    return i1 

def index2multi_hot(i1):
    v1 = np.zeros(len(idx2word),)
    v1[i1] = 1
    return v1

def smiles2vector(s1): 
    i1 = smiles2index(s1)
    v_d = index2multi_hot(i1)
    return v_d


class supData(data.Dataset):

    def __init__(self, list_IDs, labels, df_ddi,  features_data, scaler, features_scaler):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_ddi
        self.features_data = features_data
        self.scaler = scaler
        self.features_scaler = features_scaler
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].SMILES
        v_d = smiles2vector(s1)
        v_d = np.concatenate([v_d, self.features_data[index]])
        y = self.labels[index]

        if self.scaler is not None:
            y = self.scaler.transform(y)
            y = y.astype('float64')
            y = torch.from_numpy(y)

        #if self.features_scaler is not None:
        if True:
            feat = self.features_data[index]
            feat = torch.from_numpy(feat)
            return v_d, y, feat
        else:
            return v_d, y
    
