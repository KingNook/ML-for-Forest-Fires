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
            batch_prep: bool = False, 
            feature_num: int = -1):
        '''
        input data is climate variables \\
        training data is fire data
        '''

        self.input_data = input_data
        self.fire_data = fire_data

        self.batch_prep = batch_prep

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

def save_batches(
        data: geoDataset,
        save_path: str,
        batch_size: int = 64, 
        batch_num: int = 50,
        dataset_name: str = None,
        drop_last: bool = False):
    '''
    divides data into batches then saves batches to disk in separate files (?) so can be loaded individually
    each batch should be readable as a torch tensor (although it will likely be written to a json file)

    # args
    - **data**: *geoDataset* / similar \\
    dataset to be divvied up
    - **batch_size**: *int* \\
    size of each batch
    - **batch_num**: *int* \\
    number of batches per file
    - **save_path**: *path_like* \\
    path to directory to which batches will be saved
    '''

    dataset_name = dataset_name if dataset_name != None else ''
    file_length = batch_size * batch_num ## samples per file

    total_data_size = len(data)

    from math import ceil
    total_files = ceil(total_data_size / file_length)

    ## write to settings.json to make reading files easier
    with open(os.path.join(save_path, 'settings.json'), 'w') as settings_file:
        settings_file.write(json.dumps({
            'batch_size': batch_size,
            'batch_num': batch_num,
            'num_files': total_files,
            'batch_name': dataset_name if dataset_name else None
        }))

    print(f'[save_batches] {total_files} files')

    for i in range(total_files):
        print(f'[save_batches] batch {i} / {total_files}')
        file_name = f'{dataset_name}-batch_{i}.json' if dataset_name else f'batch_{i}.json'
        file_path = os.path.join(save_path, file_name)

        with open(file_path, 'w') as batch_file:
            # grab samples i * batch size to i * batch_size + 1
            start_point = i * file_length

            for j in range(batch_num):
                features = []
                labels = []
                batch_start = start_point + j * batch_size
                for k in range(batch_size):
                    sample_num = batch_start + k
                    if sample_num < total_data_size:
                        sample = data[sample_num]
                        features.append(sample[0])
                        labels.append(sample[1])
                    else:
                        break ## avoid pointless iteration

                ## each batch should be [(n features), (n samples)]

                batch_file.write(json.dumps((features, labels))) 
                batch_file.write('\n')

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

