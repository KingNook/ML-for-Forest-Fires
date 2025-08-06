import torch
import numpy as np
import pandas as pd
import os
import json

from torch import nn
from torch.utils.data import DataLoader, Dataset

from compute_wind_speed import append_wind_speed
from open_data import open_data_dir
from dask_addons import FlattenedDaskDataset
from open_fire_data import FlattenedTruthTable

class DNN(nn.Module):
    '''
    predictor for 1 variable
    '''

    def __init__(self, input_size = 13, output_size = 1, hidden_sizes = [10, 10, 10]):
        super(DNN, self).__init__()
        
        assert type(hidden_sizes) == list or type(hidden_sizes) == tuple
        assert len(hidden_sizes) >= 1
        
        self.layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]

        for i in range(1, len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*self.layers)
        

    def forward(self, x):

        return self.network(x)
    

class geoDataset(Dataset):

    def __init__(
            self, 
            input_data: FlattenedDaskDataset, 
            fire_data: FlattenedTruthTable,
            feature_num: int = -1):
        '''
        input data is climate variables \\
        training data is fire data
        '''

        self.input_data = input_data
        self.fire_data = fire_data

        self.feature_num = feature_num if feature_num > 0 else input_data.total_features

        ## validation eg have the same number of datapoints


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        '''
        should return (features, label), where
        - features are relevent input datapoints
        - label is the relevant fire data (either 1 or 0)
        '''

        features = [self.input_data[index, i] for i in range(self.feature_num)]
        feature_tensor = torch.tensor(features)
        label = self.fire_data[index]

        if label == 1:
            print(f'{feature_tensor = }, {label = }')

        if self.batch_prep:
            return features, label
        
        return feature_tensor, label

class BatchDataLoader:
    
    def __init__(self, data_dir_path):

        self.path = data_dir_path

    def load_batches(self):
        '''
        returns iterator which gives one batch each time
        '''

        data_files = os.listdir(self.path)

        ## check only files of correct form
        if 'settings.json' in data_files:
            ## read settings
            with open('settings.json', 'r') as settings_file:
                settings = json.load(settings_file)

        else:
            ## default settings
            settings = {
                'batch_size': 64,
                'batch_num': 1,
                'num_files': 1,
                'dataset_name': None
            }

        for file_num in range(settings['num_files']):
            file_name = f'{settings['dataset_name']}-batch_{file_num}.json' if settings['dataset_name'] else f'batch_{file_num}.json'
            file_path = os.path.join(self.path, file_name)
            with open(file_path, 'r') as batch_file:
                for line in batch_file:
                    ## should read line by line
                    batch = json.loads(line) # loads as list of (feature, label) pairs
                    yield batch

    def __iter__(self):
        return self.load_batches()
    

def train(model, dataloader, loss_fn, optimizer):

    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        prediction = model(X)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

