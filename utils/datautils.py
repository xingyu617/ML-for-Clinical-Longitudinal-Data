import os
import csv
import pandas as pd
import glob
import numpy as np
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence,pack_sequence

def my_train_test_valid_split(X,y,imbalanced = True):
    if imbalanced == True: # split based on the class proportions
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, stratify=y_train,test_size=0.25, random_state=1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train,test_size=0.25, random_state=1)
    
    print("train test splict:", len(X_train), len(X_val), len(X_test))
    return X_train, y_train, X_test,y_test, X_val, y_val
def load_folder(data_dir,label_dir):
    glued_data =  []
    labels = []
    
    with open(label_dir) as json_file:
        label_lists = json.load(json_file)
    #print(label_lists.keys())
    for file_name in glob.glob(data_dir+'*.csv'):
        
        patient = file_name.split("/")[-1].split(".")[0]
        #print(patient)
        if patient in label_lists["positive"]:
            
            labels.append(float(1))
        else:
            labels.append(float(0))
        #print(labels)
        x = pd.read_csv(file_name, index_col =0,low_memory=False)
        #redorder Features
        #print(x)
        x = x[['3930','3940','4035']]
        #impute Nan with mean
        x['3930'] = fill_nan(x,'3930')
        x['3940'] = fill_nan(x,'3940')
        x['4035'] = fill_nan(x,'4035')
        #x = np.expand_dims(x.to_numpy().astype(np.float64),0)
        x = x.to_numpy().astype(np.float64)
        #print(x.shape,type(glued_data))
        glued_data.append(x)
    #pad all data to max length
    max_len = get_max_len(glued_data)
    pad_data =[]
    for x in glued_data:
        x = np.pad(x, ((0,max_len-x.shape[0]),(0,0)), 'constant')
        pad_data.append(x)
    return pad_data,labels
def fill_nan(data,col):
    mean_value=data[col].mean()
    # Replace NaNs in column col with the
    # mean of values in the same column
    data[col].fillna(value=mean_value, inplace=True)
    return data[col]
def get_max_len(data):
    lengths = [ t.shape[0] for t in data ]
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
    targets = torch.LongTensor([item[1] for item in batch])
    #print(targets)
    data = torch.nn.utils.rnn.pad_sequence(data,batch_first = True)
    ## compute mask
    mask = (batch != 0)
    return (data,targets)#, lengths, mask
    