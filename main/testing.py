from geodataclass import FlattenedData

from open_data import open_data_dir, data_dir_to_zarr

## imports
#  modules
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import torch
import os

# custom code
import dask_addons
from open_data import open_data_dir, data_dir_to_zarr
from neural_net import DNN, geoDataset
from open_fire_data import FlattenedTruthTable

warnings.filterwarnings('ignore')

def clean_nan(dds):
    for k, v in dds.items():
        dds[k] = v.fillna(0)

    return dds

DATA_DIR_PATH = './data/_ZARR_FILES'

data_path = os.path.join(DATA_DIR_PATH, 'alaska_full.zarr')
prior_data_path = os.path.join(DATA_DIR_PATH, 'alaska_prior.zarr')

data = xr.open_zarr(data_path, decode_timedelta=False)
prior_data = xr.open_zarr(prior_data_path, decode_timedelta=False)

ds = dask_addons.FlattenedDaskDataset(data, prior_data)
ds.setup()

print(ds[0, 8])