import xarray as xr
from open_data import open_data_dir, data_dir_to_zarr
from compute_wind_speed import append_wind_speed
from dask_addons import FlattenedDaskDataset

import time

# path = r'C:\Users\nookh\Github\summer-25\data\alaska_main\2010-03_input_data\data.grib'
# path = r'C:\Users\nookh\Github\summer-25\data\alaska_prior\2009-09_proxy_data\data.grib'

# data_dir_to_zarr('./data/alaska_TEST_DATA', './data/_ZARR_FILES/TEST_MAIN.zarr')
# data_dir_to_zarr('./data/alaska_TEST_PRIOR', './data/_ZARR_FILES/TEST_PRIOR.zarr')

prior_data = xr.open_zarr('./data/_ZARR_FILES/TEST_PRIOR.zarr', decode_timedelta=False)
data = xr.open_zarr('./data/_ZARR_FILES/TEST_MAIN.zarr', decode_timedelta=False)

ds = FlattenedDaskDataset(data, prior_data)

prep_data, prep_prior = ds.setup()

prep_data.to_zarr('./data/_ZARR_FILES/test_main')
prep_prior.to_zarr('./data/_ZARR_FILES/test_prior')