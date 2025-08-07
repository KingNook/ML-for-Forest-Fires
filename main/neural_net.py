import torch
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

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
            feature_num: int = -1,
            return_batches: bool = True):
        '''
        input data is climate variables \\
        training data is fire data
        '''

        self.input_data = input_data
        self.fire_data = fire_data

        self.start_date = input_data.start_date
        assert pd.to_datetime(self.start_date) == pd.to_datetime(fire_data.start_date)

        self.return_batches = return_batches

        self.feature_num = feature_num if feature_num > 0 else input_data.total_features

        grid_shape = input_data.sizes['longitude']
        self.batches_per_row = grid_shape[0] // 64 if grid_shape[0] >= 64 else 1
        self.rows = input_data.sizes['latitude']

        ## validation eg have the same number of datapoints


    def __len__(self):
        return len(self.input_data)
    
    def da_from_df(self, df: pd.DataFrame):

        lat_vals = self.input_data.latitude
        long_vals = self.input_data.longitude

        df = df[['latitude', 'longitude']].drop_duplicates()
        df['presence'] = 1

        da = df.set_index(['latitude', 'longitude'])['presence'].to_xarray()

        da = da.reindex({
            'latitude': lat_vals,
            'longitude': long_vals
        }, fill_value=0).fillna(0)

        return da

    def __getitem__(self, index):
        '''
        should return (features, label), where
        - features are relevent input datapoints
        - label is the relevant fire data (either 1 or 0)
        '''
        
        if self.return_batches:

            days, hour = divmod(index, 24)
            labels = self.fire_data.get_hourly_data(days, hour)

            label_grid = self.da_from_df(labels)

            if hour == 0:
                days -= 1
                hour = 24

            data = self.input_data.full_hourly_data(
                time = self.start_date + timedelta(days=days),
                step = hour
            )

            data['fire'] = label_grid

            return data

        else:

            features = [self.input_data[index, i] for i in range(self.feature_num)]
            feature_tensor = torch.tensor(features)
            label = self.fire_data[index]
            
            return feature_tensor, label

class BatchDataLoader:
    
    def __init__(self, dataset: geoDataset):
        self.ds = dataset

    def __iter__(self):
        pass
    

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

