'''
add types of vegetation
'''

import xarray as xr
import numpy as np

def clean_coords(ds):
        '''
        rounds latitude, longitude, step so as to avoid floating point precision issues
        '''

        float_dims = ['latitude', 'longitude']
        for dim in float_dims:
            ds[dim] = np.round(ds[dim].astype(float), decimals=1)

        return ds

def resave_data(name):
    data = xr.open_dataarray(f'./data/veg_type-world/{name}.grib')
    clean_data = clean_coords(data)
    clean_data.to_zarr(f'./data/_ZARR/{name}-world')