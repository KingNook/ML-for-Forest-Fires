from geodataclass import FlattenedData

from open_data import open_data_dir, data_dir_to_zarr

import warnings
import traceback
import sys

import numpy as np

def clean_nan(dds):
    for k, v in dds.items():
        dds[k] = v.fillna(0)

    return dds

def custom_warning(message, category, filename, lineno, file=None, line=None):
    '''log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))'''
    pass

warnings.showwarning = custom_warning
'''
data = clean_nan(open_data_dir('./data/alaska_TEST_DATA'))

for key in data.keys():
    data[key].to_zarr(f'./zarr_test/{key}_alaskaTEST')
'''

print(data_dir_to_zarr('./data/alaska_main', './data/_ZARR_FILES/alaska_full.zarr'))