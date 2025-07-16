'''
functions for dealing with `.grib` files

aims:
- we want to be able to yoink *only* the values out of our file -- ie ignore all metadata; probably sensible to store this in an `xarray` Dataset? or possible DataArray?
- output these dataframes to a sensibly-named file under some directory under `./data/`
'''

import xarray as xr

with xr.open_dataset(filename_or_obj = './data/TEST_DATA/data.grib', engine = 'cfgrib') as ds: 

    print(type(ds['tp']), ds['tp'])
    