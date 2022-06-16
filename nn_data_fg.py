import numpy as np
import pandas as pd
import torch
from torch.utils import data

from subword_nmt.apply_bpe import BPE
import codecs
import pickle
from rdkit import Chem

dataFolder = r'/scratch/scratch6/joebobby/Data'
dataFolder = r'./Data'

overflow_limit = 1e10

fgroups = pd.read_csv(dataFolder + '/Functional_groups_filtered.csv')
fgroups_list = list(map(lambda x: Chem.MolFromSmarts(x), fgroups['SMARTS'].tolist()))

#fgroups = pd.read_csv(dataFolder + '/Functional_groups_filtered_daylight.csv')
#fgroups_list = list(map(lambda x: Chem.MolFromSmarts(x), fgroups['daylight.smarts'].tolist()))
memory = {}

def index2multi_hot_fg(molecule):
    v1 = np.zeros(len(fgroups_list),)
    for idx in range(len(fgroups_list)):
        if molecule.HasSubstructMatch(fgroups_list[idx]):
            v1[idx] = 1
    return v1

def smiles2vector_fg(s1, len_accept = True):
    if s1 not in memory:
        molecule = Chem.MolFromSmiles(s1)
        v_d = index2multi_hot_fg(molecule)
        if len_accept:
            memory[s1] = v_d
    else:
        v_d = memory[s1]
    return v_d

vocab_path = dataFolder + '/codes_drug_chembl_1500.txt'
bpe_codes_fin = codecs.open(vocab_path)
bpe = BPE(bpe_codes_fin, merges=-1, separator='')

vocab_map = pd.read_csv(dataFolder + '/subword_units_map_drug_chembl_1500.csv')
idx2word = vocab_map['index'].values
words2idx = dict(zip(idx2word, range(0, len(idx2word))))
max_set = 30
memory = {}

def smiles2index(s1):
    # print(s1)
    t1 = bpe.process_line(s1).split() #split
    i1 = [words2idx[i] for i in t1] #index
    return i1 

def index2multi_hot(i1):
    v1 = np.zeros(len(idx2word),)
    v1[i1] = 1
    return v1

def smiles2vector_fsr_fg(s1, len_accept = True):
    if s1 not in memory:
        i1 = smiles2index(s1)
        v_d = index2multi_hot(i1)
        molecule = Chem.MolFromSmiles(s1)
        v_d2 = index2multi_hot_fg(molecule)
        v_d = np.concatenate([v_d,v_d2])
        if len_accept:
            memory[s1] = v_d
    else:
        v_d = memory[s1]
    return v_d

def smiles2vector_fsr(s1, len_accept = True):
    if s1 not in memory:
        i1 = smiles2index(s1)
        v_d = index2multi_hot(i1)
        if len_accept:
            memory[s1] = v_d
    else:
        v_d = memory[s1]
    return v_d

class supData(data.Dataset):

    def __init__(self, list_IDs, labels, df_ddi, features_data, scaler, features_scaler, methods):
        'Initialization'
        global memory
        memory = {}
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_ddi
        self.cat_features_data = features_data[0]
        self.num_features_data = features_data[1]
        self.scaler = scaler
        self.features_scaler = features_scaler
        self.counter = 0
        self.length = len(self.list_IDs) # For memoization
        self.methods = methods
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].SMILES
        if 'FSR' in self.methods and 'FG' in self.methods:
            v_d = smiles2vector_fsr_fg(s1, self.length < overflow_limit)
        elif 'FG' in self.methods:
            v_d = smiles2vector_fg(s1, self.length < overflow_limit)
        elif 'FSR' in self.methods:
            v_d = smiles2vector_fsr(s1, self.length < overflow_limit)
        else:
            raise ValueError
        #v_d = np.concatenate([v_d, self.features_data[index]])
        y = self.labels[index]

        if self.scaler is not None:
            y = self.scaler.transform(y)
            y = y.astype('float64')
            y = torch.from_numpy(y)

        if self.features_scaler is not None and self.num_features_data is not None:
            try:
                # Scaler for numerical data type
                feat = self.features_scaler.transform(self.num_features_data[index].reshape(1,-1))
            except:
                print(index)
                print(max(self.num_features_data[index]))
                print(sum([i==i for i in self.num_features_data[index]])) # should be 92 features that have no nan values
            feat[feat>1] = 1
            feat[feat<0] = 0
            num_feat = torch.from_numpy(feat.reshape(-1))
            num_feat[num_feat!=num_feat] = 0
            cat_feat = torch.from_numpy(self.cat_features_data[index])
            return v_d, y, cat_feat, num_feat
        elif self.cat_features_data is not None:
            feat = self.num_features_data[index]
            num_feat = torch.from_numpy(feat)
            num_feat[num_feat!=num_feat] = 0
            cat_feat = torch.from_numpy(self.cat_features_data[index])
            return v_d, y, cat_feat, num_feat
        else:
            return v_d, y
#'''
