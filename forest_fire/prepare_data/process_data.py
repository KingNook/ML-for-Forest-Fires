'''
Contains code for managing data once it's been downloaded
Note that default settings will expect data to be stored under './data/<extent>/main/combined.grib' and './data/<extent>/prior/combined.grib'

Contains code to
- convert data from GRIB -> Zarr
- calculates wind speed
- calculate proxy variables + write to new Zarr

Note that code here only calculates the following proxy variables (as of 28/08/25):
- 30/90/180 day rolling averages for Temperature
- 30/90/180 day rolling averages for Precipitation
'''

import xarray as xr
from xarray import Dataset, DataArray

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