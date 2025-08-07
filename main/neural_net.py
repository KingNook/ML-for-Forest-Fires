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
            approx_batch_size: int = 64,
            return_batches: bool = True):
        '''
        input data is climate variables \\
        training data is fire data
        '''

        self.input_data = input_data
        self.fire_data = fire_data

        self.start_date = input_data.start_date
        self.end_date = input_data.end_date
        assert pd.to_datetime(self.start_date) == pd.to_datetime(fire_data.start_date)

        self.return_batches = return_batches

        self.feature_num = feature_num if feature_num > 0 else input_data.total_features

        self.cols = input_data.sizes['longitude']
        self.batches_per_row = round(self.cols[0] / approx_batch_size) if self.cols[0] > 64 else 1
        self.rows = input_data.sizes['latitude']

        self.lat_vals = self.input_data.latitude
        self.long_vals = self.input_data.longitude

        ## validation eg have the same number of datapoints

        ## define batches
        self.calc_batches()

    def calc_batches(self):
        '''
        ONE-OFF TO BE CALLED ON __INIT__ \\
        calculates the coords of each batch
        '''

        batch_size, _ = divmod(self.cols, self.batches_per_row)
        self.batch_long = [self.long_vals[i*batch_size:(i+1)*batch_size] for i in range(self.batches_per_row-1)]
        self.batch_long.append([self.long_vals[(self.batches_per_row-1)*batch_size:]])

    def __len__(self):
        return (self.end_date - self.start_date).days * 24 * self.batches_per_row * self.rows
    
    def da_from_df(self, df: pd.DataFrame):

        df = df[['latitude', 'longitude']].drop_duplicates()
        df['presence'] = 1

        da = df.set_index(['latitude', 'longitude'])['presence'].to_xarray()

        da = da.reindex({
            'latitude': self.lat_vals,
            'longitude': self.long_vals
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

        self.batch_longs = dataset.batch_long ## longitude values of each batch
        self.start_date = dataset.start_date

        self.lat_vals = dataset.lat_vals
        self.long_vals = dataset.batch_long

        self.max_lat_idx = self.ds.rows
        self.max_long_idx = self.ds.batches_per_row

        assert self.max_long_idx == len(self.long_vals)

        self.feature_vars = []
        self.label_vars = []

        self.reset_idx()
        

    def reset_idx(self):
        self.idx = 0

        self.lat_idx = 0
        self.long_idx = 0

        self.refresh_data = True


    def extract_data(self, data: xr.Dataset, lat_idx: int, long_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        grabs the batch according to given indices
        ## Parameters
        **data**: *xarray.Dataset* \\
        Data for the whole grid for a single time, step

        **lat_idx**: *int* \\
        Ranges from `0` to `self.max_lat_idx`; self-explanatory

        **long_idx**: *int* \\
        Specifies which set of longitude values should be used for the batch

        ## Returns
        **batch**: *tuple*, (X, Y) \\
        A tuple containing `X`, the input features, and `Y`, the observed data (ie 1 for fire, 0 for not)
        '''

        values = data.sel(
            latitude = self.lat_vals[lat_idx],
            longitude = self.long_vals[long_idx]
        )

        features = [values[var].values for var in self.feature_vars] ## rows are features, columns are samples
        features = np.ndarray(features).transpose() ## rows are samples, columns are features

        labels = np.ndarray([values[var].values for var in self.label_vars]).transpose()

        return torch.tensor(features), torch.tensor(labels)


    def __iter__(self):
        return self

    def __next__(self):
        if self.refresh_data:
            try:
                grid_data = self.ds[self.idx]
                self.refresh_data = False
            except IndexError:
                self.reset_idx()
                raise StopIteration
        
        batch = self.extract_data(grid_data, self.lat_idx, self.long_idx)

        self.long_idx += 1
        if self.long_idx == self.max_long_idx:
            self.long_idx = 0
            self.lat_idx += 1

            if self.lat_idx == self.max_lat_idx:
                self.lat_idx = 0
                self.idx += 1
                self.refresh_data = True

        yield batch




def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        prediction = model(X)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")