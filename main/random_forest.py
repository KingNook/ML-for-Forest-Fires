import pandas as pd
import dask.dataframe as dd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ALL_FEATURES = ['mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_180', 'mu_tp_90', 'mu_tp_30', 'd2m', 'sp', 't2m', 'tp', 'ws10', 'lai_hv', 'lai_lv']
NON_VEG_FEATURES = ['d2m', 'mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_90', 'mu_tp_30', 'sp', 'mu_tp_180', 't2m', 'tp', 'ws10']
NON_PROXY_FEATURES = ['d2m', 'lai_hv', 'lai_lv', 'sp', 'tot_t2m', 't2m', 'tot_tp', 'tp', 'ws10']
PROXY_FEATURES = ['mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_90', 'mu_tp_30', 'mu_tp_180']
PRINT_FEATURES = ALL_FEATURES + ['tvh', 'tvl', 'fire']

def prep_samples(ds: xr.Dataset, features: list[str] = ALL_FEATURES, label_var: str = 'fire', drop_vars: list | None = None, pre_stacked: bool = False, include_tv: bool = False, compute: bool = False, clean_nan: bool = True):
    '''
    takes dataset w/ all required variables + 'fire'
    outputs X, y

    ds might be a dask dataframe; need to account for this too
    '''

    if pre_stacked or type(ds) == dd.DataFrame:
        stacked = ds
    else:
        stacked = ds.stack(sample = ('time', 'step', 'latitude', 'longitude'))

    if drop_vars != None:
        feat = features.copy()
        for var in drop_vars:
            try:
                feat.remove(var)
            except ValueError:
                print(f'Variable {var} not found in features')
        
        features = feat

    features_stacked = stacked[features]
    labels_stacked = stacked['fire']

    if type(ds) == xr.Dataset:       

        samples = len(stacked['sample'])
        X = features_stacked.to_array().transpose('sample', 'variable').chunk({'sample':samples // 10}).data
        y = labels_stacked.chunk({'sample':samples // 10}).data
    elif type(ds) == dd.DataFrame:
        X = ds[features].to_dask_array().compute()
        y = labels_stacked.to_dask_array().compute()

    if include_tv:

        try:
            time_steps = len(ds.time) * 24
        except TypeError:
            ## if only 1 timestep, ds.time will be unsized
            time_steps = 1
        
        long_vals = ds.longitude.values
        long_vals[long_vals < 0] = long_vals[long_vals < 0] + 360 ## tv has 0 to 360 while most extents have -180 to 180

        try:
            tv = xr.open_zarr('./data/_ZARR/veg_types')
        except Exception as e:
            # for if in jupyter notebooks probably
            print(f'Exception: {e} \nAttempting to open veg data from ../data/_ZARR/veg_types instead')
            tv = xr.open_zarr('../data/_ZARR/veg_types')

        tv_local = tv.sel(
            latitude = ds.latitude,
            longitude = long_vals
        ).stack(sample = ['latitude', 'longitude']).to_array().transpose('sample', 'variable')

        X = np.append(X, np.repeat(tv_local, time_steps, axis=0), axis=1)

    ## clean nans
    if clean_nan:
        mask = ~np.isnan(X).any(axis=1) | np.isnan(y)
        X = X[mask]
        y = y[mask]

    if compute:
        return (X.compute(), y.compute())

    else:
        return (X, y)

def plot_feature_distribution(original, new, features: list = ALL_FEATURES, cts: bool = True):

    for i, feature in enumerate(features):
        plt.figure(figsize=(10, 5))
        if cts:
            sns.histplot(original.transpose()[i], bins=30, stat='density', label='Original', shrink=0.5)
            sns.histplot(new.transpose()[i], bins=30, stat='density', label='Resampled', multiple='dodge')
        else:
            sns.barplot(original.transpose()[i], label='Original')
            sns.barplot(new.transpose()[i], label='Resampled')
        plt.title(feature)