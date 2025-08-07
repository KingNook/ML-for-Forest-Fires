## imports
#  modules
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import torch
import os
import time

# custom code
from dask_addons import FlattenedDaskDataset
from open_data import open_data_dir, data_dir_to_zarr
from neural_net import DNN, geoDataset, BatchDataLoader
from open_fire_data import FlattenedTruthTable

warnings.filterwarnings('ignore')

setup_start_time = time.time()

DATA_DIR_PATH = './data/_ZARR_FILES'

data_path = os.path.join(DATA_DIR_PATH, 'alaska_full.zarr')
prior_data_path = os.path.join(DATA_DIR_PATH, 'alaska_prior.zarr')

data = xr.open_zarr(data_path, decode_timedelta=False)
prior_data = xr.open_zarr(prior_data_path, decode_timedelta=False)

input_ds = FlattenedDaskDataset(data, prior_data)
input_ds.setup()

fire_data = FlattenedTruthTable(
    pd.read_csv('./data/_FIRE/alaska_range_csv/data.csv'),
    lat_vals = input_ds.latitude,
    long_vals = input_ds.longitude
)

ds = geoDataset(input_ds, fire_data, feature_num = 13)

prog_start_time = time.time()
print(f'Setup done in {prog_start_time - setup_start_time:.02f}s')
## == DON'T TOUCH CODE ABOVE == ##

dataloader = BatchDataLoader(ds)

test = enumerate(dataloader)

for i in test:

    print(i)