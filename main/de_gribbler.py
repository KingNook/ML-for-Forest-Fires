'''
functions for dealing with `.grib` files

aims:
- we want to be able to yoink *only* the values out of our file -- ie ignore all metadata; probably sensible to store this in an `xarray` Dataset? or possible DataArray?
- output these dataframes to a sensibly-named file under some directory under `./data/`
'''

import xarray as xr
import pandas as pd
import numpy as np

from IPython.display import display

DATA_PATH = './sample_month/cvh.grib' # './data/data.grib'

with xr.open_dataset(filename_or_obj = DATA_PATH, engine = 'cfgrib') as ds: 

    print(ds.cvh)