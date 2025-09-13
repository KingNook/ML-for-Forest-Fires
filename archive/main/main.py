print('prog start')
import pickle

import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from open_data import open_data_dir
from compute_wind_speed import append_wind_speed
# from open_fire_data import fire_ds, FlattenedTruthTable
print('imports done')

prior_data = open_data_dir('./data/alaska_prior')
data = open_data_dir('./data/alaska_main')

print('data open')

print('data prepared')

def merge_data(dds) -> xr.Dataset:
    '''
    take a dict of datasets and merge
    '''
    return xr.concat([ds for ds in dds.values()], dim='time')

merge_data(input_data.data).to_zarr('./data/_ZARR_READY/alaska_main')
merge_data(input_data.prior_data).to_zarr('./data/_ZARR_READY/alaska_prior')