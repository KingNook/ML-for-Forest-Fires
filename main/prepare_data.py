'''
various functions for preparing data before passing it into the model
'''

import xarray as xr
import numpy as np
import os

from dask_addons import FlattenedDaskDataset

config = {
    'tp':     {'method': 'minmax', 'log': True},
    't2m':    {'method': 'standard', 'log': False},
    'ws10':   {'method': 'minmax', 'log': True},
    'lai_hv': {'method': 'minmax', 'log': False},
    'lai_lv': {'method': 'minmax', 'log': False},
    'sp':     {'method': 'standard', 'log': False},
    'd2m':    {'method': 'standard', 'log': False},
}

def normalise_feature_global(xarr, method='standard', log=False):
    if log:
        xarr = xr.where(xarr > 0, np.log1p(xarr), 0)

    match method:
        case 'standard':
            mean = xarr.mean().item()
            std = xarr.std().item()
            return (xarr - mean) / std
        case 'minmax':
            min_val = xarr.min().item()
            max_val = xarr.max().item()
            return (xarr - min_val) / (max_val - min_val)
        case _:
            raise ValueError("Method must be 'standard' or 'minmax'")

def normalise_dataset_global(ds, config=config):
    '''
    normalises all data variables in config
    ## Parameters
    **ds**: *xarray.Dataset*
    Dataset to be normalise-ified

    **config**: *dict*
    Mapping variable names to dicts like:
    ```
    {
        'method': 'standard' or 'minmax', 
        'log': True/False
    }
    ```
    '''
    processed = {}
    for var, opts in config.items():
        da = ds[var]
        da = normalise_feature_global(da, method=opts.get('method', 'standard'), log=opts.get('log', False))
        processed[var] = da
    return xr.Dataset(processed)

def concat_gribs_from_subdirs(root_dir, output_name='combined.grib'):
    '''
    Combines all 'data.grib' files from subdirectories into one big ol' GRIB file.
    
    ## Parameters:
    **root_dir**: *str*
    Path to the folder containing subdirectories

    **output_path**: *str*
    Path for combined GRIB file
    '''
    output_path = os.path.join(root_dir, output_name)
    with open(output_path, 'wb') as outfile:
        for subdir in sorted(os.listdir(root_dir)):
            sub_path = os.path.join(root_dir, subdir)
            grib_path = os.path.join(sub_path, 'data.grib')
            if os.path.isdir(sub_path) and os.path.isfile(grib_path):
                print(f"Adding: {grib_path}")
                with open(grib_path, 'rb') as infile:
                    outfile.write(infile.read())
    print(f"\n✅ Combined GRIB saved to: {output_path}")

if __name__ == '__main__':

    import time
    start_time = time.time()

    def time_elapsed(end, start = start_time):
        return f'{end - start:.02f}s'

    prior_data = xr.open_dataset('./data/la_forest_prior/combined.grib', chunks='auto', decode_timedelta=False)
    data = xr.open_dataset('./data/la_forest_main/combined.grib', chunks={'time': 365 }, decode_timedelta=False)
    data_open_time = time.time()
    print(f'Data open in: {time_elapsed(data_open_time)}')
    ds = FlattenedDaskDataset(data, prior_data)
    ds.setup()
    setup_time = time.time()
    print(f'Setup done in: {time_elapsed(setup_time, data_open_time)} // {time_elapsed(setup_time)}')
    ds.data.to_zarr('./data/_ZARR_READY/la_main_data')
    ds.prior_data.to_zarr('./data/_ZARR_READY/la_prior_data')
    finish = time.time()
    print(f'Data writing done in: {time_elapsed(finish)}s // {time_elapsed(finish)}')