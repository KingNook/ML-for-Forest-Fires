'''
various functions for preparing data before passing it into the model
'''

import xarray as xr
import numpy as np
import pandas as pd
import os

from dask_addons import FlattenedDaskDataset

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

    # df['acq_date'] = df['acq_date'].astype(int)

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

    ds = ds.assign(time=lambda ds: ds.time.astype('datetime64[ns]')).transpose('latitude', 'longitude', 'time', 'step')

    return ds

def clean_na(ds: xr.Dataset):

    ds_stacked = ds.stack(dt = ('time', 'step'))
    ds_clean = ds_stacked.dropna(dim = 'dt', how='all')
    ds_unstacked = ds_clean.unstack(dim = 'dt')
    
    return ds_unstacked.chunk('auto')

def raw_data_to_zarr(data_path: str, extent_name: str, main: bool = True, prior: bool = True):
    '''
    writes `<data_path>/<extent_name>_main/combined.grib` to a ZARR group at `<data_path>/_ZARR/<extent_name>_main`
    '''

    if main:
        main_data_path = os.path.join(data_path, f'{extent_name}_main', 'combined.grib')
        data = xr.open_dataset(main_data_path, decode_timedelta=False)
        clean_data = clean_na(data)
        main_zarr_path = os.path.join(data_path, '_ZARR', f'{extent_name}_main')
        clean_data.to_zarr(main_zarr_path, mode='w', align_chunks=True)

    if prior:
        prior_data_path = os.path.join(data_path, f'{extent_name}_prior', 'combined.grib')
        prior_data = xr.open_dataset(prior_data_path, decode_timedelta=False)
        clean_prior = clean_na(prior_data)
        prior_zarr_path = os.path.join(data_path, '_ZARR', f'{extent_name}_prior')
        clean_prior.to_zarr(prior_zarr_path, mode='w', align_chunks=True)

def setup_from_zarr(data_path, extent_name, fire_dir, main = '', prior = ''):

    main_name = f'{extent_name}_main' if main == '' else main
    prior_name = f'{extent_name}_prior'if prior == '' else prior

    main_data_path = os.path.join(data_path, '_ZARR', main_name)
    prior_data_path = os.path.join(data_path, '_ZARR', prior_name)

    prior_data = xr.open_zarr(prior_data_path, decode_timedelta=False)
    data = xr.open_zarr(main_data_path, decode_timedelta=False)

    ds = FlattenedDaskDataset(data, prior_data, clean_data=False)
    ds.setup()

    lat = data.latitude.values.round(decimals=1)
    long = data.longitude.values.round(decimals=1)
    time_vals = data.time.values.astype('datetime64[ns]')
    step = data.step.values.astype(float)

    fire_data = pd.read_csv(os.path.join(data_path, '_FIRE', fire_dir, 'data.csv'), parse_dates=['acq_date'])
    fd = fire_data.loc[(fire_data['type'] == 0) & (fire_data['confidence'] >= 70)]

    fire_ds = ds_from_df(fd, lat, long, time_vals, step)
    ds.data['fire'] = fire_ds['fire']

    ds.rechunk()

    ds_out = ds.data.reset_coords()

    encoding = {var: {} for var in ds_out.data_vars}

    ds_out.to_zarr(f'./data/_ZARR_READY/{extent_name}', mode='w', encoding=encoding)

if __name__ == '__main__':

    import time
    start_time = time.time()

    def time_elapsed(end, start = start_time):
        return f'{end - start:.02f}s'

    #raw_data_to_zarr('./data', 'canada-post', prior=False)

    data_open_time = time.time()
    print(f'Data written to ZARR in: {time_elapsed(data_open_time)}')

    setup_from_zarr('./data', 'canada-post', 'canada-post_csv', prior='canada_main')

    finish = time.time()
    print(f'Data setup done in: {time_elapsed(finish, data_open_time)} // {time_elapsed(finish)}')