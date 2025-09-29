'''
Contains code for managing data once it's been downloaded

Note that default settings will expect data to be stored under `./data/<extent>/main/combined.grib` and `./data/<extent>/prior/combined.grib`

Contains code to
- convert data from GRIB -> Zarr
- calculates wind speed
- calculate proxy variables + write to new Zarr
- convert a DataFrame to a Dataset
- add fire data to the dataset

Note that code here only calculates the following proxy variables (as of 28/08/25):
- 30/90/180 day rolling averages for Temperature
- 30/90/180 day rolling averages for Precipitation
'''
from constants import DEFAULT_FIRE_CONFIG

import xarray as xr
from xarray import Dataset, DataArray

import pandas as pd
from pandas import DataFrame

import numpy as np

import os

def grib_to_zarr(grib_path: str, zarr_path: str, name: str = 'combined_zarr'):
    '''
    Converts a `.grib` file to a Zarr group

    Parameters
    ----------
    grib_path: str or path-like
        Path to `.grib` file. Will likely be something like `./data/<extent>/main/combined.grib`

    zarr_path: str or path-like
        Path to directory that the Zarr group will be written to

    name: str
        Name of Zarr group. Default is `combined_zarr`
    '''

    ds = xr.open_dataset(grib_path, decode_timedelta=False)
    output_path = os.path.join(zarr_path, name)
    ds.to_zarr(output_path)

def compute_wind_speed(ds: Dataset) -> DataArray:
    '''
    Computes the resultant wind speed

    Note that this WILL THROW AN ERROR if it cannot find the `u10` and `v10` variables in the Dataset

    Parameters
    ----------
    ds: Dataset
        Dataset with wind variables `u10` and `v10`

    Returns
    -------
    ws10: DataArray
        DataArray of resultant wind speed
    '''

    square_speed = ds['u10']**2 + ds['v10']**2

    return square_speed**0.5

def compute_proxy(main_ds: Dataset, prior_ds: Dataset, proxy_var: str, timeframe: int) -> DataArray:
    '''
    Computes the rolling average of the proxy variable over the given timeframe

    Parameters
    ----------
    main_ds: Dataset
        Dataset covering time range of interest
    
    prior_ds: Dataset
        Dataset spanning at least `timerange` days before the start of `main_ds`

    proxy_var: str
        Variable of interest. Both `main_ds` and `prior_ds` must contain this as a data variable

    timeframe: int
        Number of days over which the rolling average is taken

    Returns
    -------
    proxy_var: DataArray
        Rolling average for time period covered by `main_ds`
    '''

    assert proxy_var in prior_ds.data_vars, f'{proxy_var} not found in prior dataset'
    assert proxy_var in main_ds.data_vars, f'{proxy_var} not found in main dataset'

    ds = xr.concat(prior_ds[proxy_var], main_ds[proxy_var])
    window = timeframe * 24

    stacked = ds.stack(dt = ['time', 'step']).rolling(window=window).mean()

    unstacked = stacked.unstack('dt')

    return unstacked


def process_fire_data(df: DataFrame):
    '''
    Processes fire data from MODIS to a useable format

    Parameters
    ----------
    df: DataFrame
        Dataframe with fire data read from the csv provided by MODIS

    Returns
    -------
    clean_df: DataFrame
        Dataframe containing fire data in a compatible form
    '''

    df = df.copy()

    df['latitude'] = df['latitude'].round(1)
    df['longitude'] = df['longitude'].round(1)

    df['acq_time'] = np.floor(df['acq_time']/100).astype(np.float64)
    
    mask = df['acq_time'] == 0

    df.loc[mask, 'acq_time'] = 24.0
    df.loc[mask, 'acq_date'] = df.loc[mask, 'acq_date'] - pd.Timedelta(days=1)

    return df

def da_from_df(df: DataFrame, ds: Dataset, dims: list = ['latitude', 'longitude', 'time', 'step']) -> DataArray:
    '''
    Turns the fire DataFrame into a DataArray, ready to add to the main Dataset

    Parameters
    ----------
    df: DataFrame
        DataFrame containing Fire data -- read directly from csv (will be processed)

    ds: Dataset
        Reference Dataset to get reference dimensions

    dims: list (optional)
        Choice of dimensions to put into the DataArray. By default, this will be 'latitude', 'longitude', 'time' and 'step'

    Returns
    -------
    fire_array: DataArray
        DataArray containing fire data, with dimensions matching the input data
    '''

    ## we get
    # dims = names
    # dim_vals = values
    # reference_dims = dict

    reference_dims = {
        ref_dim : ds[ref_dim].data for ref_dim in dims
    }

    dim_vals = list(reference_dims.values())


    df_proc= process_fire_data(df)[['longitude', 'latitude', 'acq_date', 'acq_time']].drop_duplicates()

    df_renamed = df_proc.rename(columns = {
        'acq_date': 'time',
        'acq_time': 'step'
    })

    df_renamed['presence'] = 1

    da = df_renamed.set_index(dims)

    idx = pd.MultiIndex.from_product(dim_vals, names = dims)
    da = da.reindex(idx)
    da = da['presence'].to_xarray()

    da = da.reindex(reference_dims).fillna(0)

    return da

    ds = xr.Dataset()
    ds['fire'] = da

    ds = ds.assign(time=lambda ds: ds.time.astype('datetime64[ns]')).transpose('latitude', 'longitude', 'time', 'step')

    return ds

def add_DAILY_fire_data(main_dataset: Dataset, fire_data: DataFrame, config: dict = DEFAULT_FIRE_CONFIG):
    '''
    NOTE: need to change helper functions (`process_fire_data` and `da_from_df`) to take configs before implementing this
    
    Adds daily fire data to a dataset; note that this considers fires from 1:00 to 1:00 as opposed to 0:00 to 24:00 as may be expected;
    this doesn't really have a huge effect since there are not a lot of fires that happen entirely between 0:00 and 1:00 anyways

    Parameters
    ----------
    main_dataset: Dataset
        Dataset of input data (would like to add fire data to this)

    fire_data: DataFrame
        DataFrame (as read from csv) of fire data -- should be sourced from MODIS (other formats not currenlty supported)

    config: dict (optional)
        Thresholds for what is considered a fire; by default this is
            - confidence > 70
            - type == 0 (ie vegetation fire)
    
    Returns
    -------
    new_dataset: Dataset
        Dataset with new 'fire' variable
    '''

    raise NotImplementedError

def add_HOURLY_fire_data(main_dataset: Dataset, fire_data: DataFrame, config: dict = DEFAULT_FIRE_CONFIG):
    '''
    NOTE: need to check overpass frequency etc to see if it's even worth doing hourly data
    
    Adds hourly fire data to a dataset -- note that fire data was downloaded from MODIS as a csv, so it is easiest to read as a DataFrame using pandas.read_csv
    
    It may be worth looking at the [MODIS user guide](https://modis-fire.umd.edu/files/MODIS_C6_C6.1_Fire_User_Guide_1.0.pdf) (see page 39/64 for information about column headers)
    
    Parameters
    ----------
    main_dataset: Dataset
        Dataset to which data will be added

    fire_data: DataFrame
        Dataframe from which fire data is read -- should have data of MODIS form

    config: dict (optional)
        Thresholds for what is considered a fire; by default this is
            - confidence > 70
            - type == 0 (ie vegetation fire)

    Returns
    -------
    new_dataset: Dataset
        Dataset with new 'fire' variable
    '''

    params = config.keys()

    fire_ds = fire_data

    if 'confidence' in params:
        fire_ds = fire_ds[fire_ds['confidence'] > config['confidence']]

    if 'type' in params:
        fire_ds = fire_ds[fire_ds['type'].isin(config['type'])]

    fda = da_from_df(fire_data)

    return main_dataset.assign(fire = fda)

def setup_dataset(main_path: str, prior_path: str, proxy_config: dict, drop_vars: bool = False) -> Dataset:
    '''
    This will:
    - calculate total wind speed
    - calculate proxy variables
        - 30/90/180 day temperature
        - 30/90/180 day precipitation

    Parameters
    ----------
    main_path: str or path-like
        Path to 'main' dataset (ie dataset that contains the time range of interest)

    prior_path: str or path-like
        Path to 'prior' dataset. This dataset must contain enough data to calculate the furthest proxy, eg if furthest proxy is 720 days (eg 360 stepped x2), then this would have to stretch back at least 720 days (ie about 2 years) before the start of the main data

    proxy_config: dict with (var, timeframe) pairs
        Keys are the proxy variables, values are tuples containing timeframes for which proxies are calculated.

        For single timeframes, put a comma to mark it as a tuple, eg `(1,)`, else Python will read `(1)` as an integer

    drop_vars: bool, optional
        Whether to drop the base variables used to calculate proxy variables after proxies have been calculated. Default is False

    Returns
    -------
    prepared_dataset: Dataset
        An xarray Dataset object that contains all relevant variables
    '''

    try:
        main_ds = xr.open_zarr(main_path)
    except:
        main_ds = xr.open_dataset(main_path)
    
    try:
        prior_ds = xr.open_zarr(prior_path)
    except:
        prior_ds = xr.open_dataset(prior_path)
    
    main_ds['ws10'] = compute_wind_speed(main_ds)
    
    # proxies
    for proxy_var, timeframes in proxy_config:
        for timeframe in timeframes:
            proxy_name = f'mu_{proxy_var}_{timeframe}'
            main_ds[proxy_name] = compute_proxy(main_ds, prior_ds, proxy_var, timeframe)

    return main_ds