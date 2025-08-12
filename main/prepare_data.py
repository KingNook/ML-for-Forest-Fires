'''
various functions for preparing data before passing it into the model
'''

import xarray as xr
import numpy as np
import pandas as pd
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

def process_fire_data(df: pd.DataFrame):
    '''
    - round latitude, longitude, acq_time
    '''

    df = df.copy()

    df['latitude'] = df['latitude'].round(1)
    df['longitude'] = df['longitude'].round(1)

    df['acq_time'] = np.floor(df['acq_time']/100).astype(np.float64)
    
    mask = df['acq_time'] == 0

    df.loc[mask, 'acq_time'] = 24.0
    df.loc[mask, 'acq_date'] = df.loc[mask, 'acq_date'] - pd.Timedelta(days=1)

    df['acq_date'] = df['acq_date'].astype(int)

    return df

def ds_from_df(df: pd.DataFrame, lat: list, long: list, time: list, step: list) -> xr.DataArray:

    df_proc= process_fire_data(df)[['longitude', 'latitude', 'acq_date', 'acq_time']].drop_duplicates()

    df_renamed = df_proc.rename(columns = {
        'acq_date': 'time',
        'acq_time': 'step'
    })

    df_renamed['presence'] = 1

    da = df_renamed.set_index(['longitude', 'latitude', 'time', 'step'])

    idx = pd.MultiIndex.from_product([long, lat, time, step], names=['longitude', 'latitude', 'time', 'step'])
    da = da.reindex(idx)
    da = da['presence'].to_xarray()

    da = da.reindex({
        'latitude': lat,
        'longitude': long,
        'time': time,
        'step': step
    }).fillna(0)

    ds = xr.Dataset()
    ds['fire'] = da

    return ds.assign(time=lambda ds: ds.time.astype('datetime64[ns]'))


if __name__ == '__main__':

    import time
    start_time = time.time()

    def time_elapsed(end, start = start_time):
        return f'{end - start:.02f}s'

    prior_data = xr.open_dataset('./data/la_forest_prior/combined.grib', chunks='auto', decode_timedelta=False)
    data = xr.open_dataset('./data/la_forest_main/combined.grib', chunks={'time': 365 }, decode_timedelta=False)
    fire_data = pd.read_csv('./data/_FIRE/la_forest_csv/data.csv', parse_dates=['acq_date'])
    data_open_time = time.time()
    print(f'Data open in: {time_elapsed(data_open_time)}')
    ds = FlattenedDaskDataset(data, prior_data)
    
    fire_ds = ds_from_df(fire_data, data.latitude.values, data.longitude.values, data.time.values, data.step.values)
    ds.data = ds.data.merge(fire_ds)

    ds.setup()

    setup_time = time.time()
    print(f'Setup done in: {time_elapsed(setup_time, data_open_time)} // {time_elapsed(setup_time)}')
    ds.data.to_zarr('./data/_ZARR/canada')
    ds.prior_data.to_zarr('./data/_ZARR/canada_prior')
    finish = time.time()
    print(f'Data writing done in: {time_elapsed(finish)}s // {time_elapsed(finish)}')