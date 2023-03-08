import os
import csv
import pandas as pd
import glob
import numpy as np
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence,pack_sequence


def load_folder(directory,label_lists):
    glued_data =  []
    labels = []
    print(directory)
    for file_name in glob.glob(directory+'*.csv'):
        print("Enter:",file_name)
        patient = file_name.split("\\")[-1].split(".")[0]
        if patient in label_lists["positive"]:
            labels.append(float(1))
        else:
            labels.append(float(0))
        #print(labels)
        x = pd.read_csv(file_name, index_col =0,low_memory=False)
        #redorder Features
        x = x[['3930','3940','4035']]
        #impute Nan with mean
        x['3930'] = fill_nan(x,'3930')
        x['3940'] = fill_nan(x,'3940')
        x['4035'] = fill_nan(x,'4035')
        #x = np.expand_dims(x.to_numpy().astype(np.float64),0)
        x = x.to_numpy().astype(np.float64)
        print(x.shape,type(glued_data))
        glued_data.append(x)
    return glued_data,labels
def fill_nan(data,col):
    mean_value=data[col].mean()
    # Replace NaNs in column col with the
    # mean of values in the same column
    data[col].fillna(value=mean_value, inplace=True)
    return data[col]
def get_max_len(data):
    lengths = [ t.shape[1] for t in data ]
    max_len = max(lengths)
    return max_len

class myDataset(Dataset):
    def __init__(self, data,label) -> None:
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (self.data[index],self.label[index])

    def __len__(self):
        return len(self.data)
    
def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item[1] for item in batch]
    return (data,targets)
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t[0].shape[0] for t in batch ])
    ## padd
    data = [ torch.Tensor(t[0]) for t in batch ]
    targets = torch.FloatTensor([item[1] for item in batch])
    #print(targets)
    data = torch.nn.utils.rnn.pad_sequence(data,batch_first = True)
    ## compute mask
    mask = (batch != 0)
    return (data,targets)#, lengths, mask
    