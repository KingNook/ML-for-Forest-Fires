import dask.array as da
import pandas as pd
import dask.dataframe as dd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler

downsampler = RandomUnderSampler(sampling_strategy=0.1, random_state=10)

FINAL_TRY = ['d2m', 'lai_hv', 'lai_lv', 'mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_90', 'mu_tp_30', 'sp', 'mu_tp_180', 't2m', 'tp', 'ws10']
OLD_FEATURES = ['d2m', 'lai_hv', 'lai_lv', 'mu_tp_180', 'mu_t2m_30', 'mu_tp_90', 'mu_t2m_180', 'mu_t2m_90', 'sp', 'mu_tp_30', 't2m', 'ws10', 'tp']
REORDERED_OLD_FEATURES = ['lai_lv', 'mu_t2m_30', 'mu_t2m_90', 'd2m', 'mu_tp_180', 'mu_t2m_180', 'lai_hv', 'mu_tp_90', 't2m', 'sp', 'mu_tp_30', 'tp', 'ws10']
ROF_2 = ['d2m', 'lai_hv', 'fire', 'lai_lv', 'mu_tp_180', 'mu_t2m_30', 'mu_tp_90', 'mu_t2m_180', 'mu_t2m_90', 'sp', 'mu_tp_30', 't2m', 'ws10', 'tp']


ALL_FEATURES = ['mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_180', 'mu_tp_90', 'mu_tp_30', 'sp', 'd2m', 't2m', 'tp', 'mu_ws10_1', 'lai_hv', 'lai_lv']
NON_VEG_FEATURES = ['d2m', 'mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_90', 'mu_tp_30', 'sp', 'mu_tp_180', 't2m', 'tp', 'ws10']
NON_PROXY_FEATURES = ['d2m', 'lai_hv', 'lai_lv', 'sp', 'tot_t2m', 't2m', 'tot_tp', 'tp', 'ws10']
PROXY_FEATURES = ['mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_90', 'mu_tp_30', 'mu_tp_180']
PRINT_FEATURES = ALL_FEATURES + ['tvh', 'tvl', 'fire']

def prep_samples_downsampled(ds: xr.Dataset, features: list[str] = ALL_FEATURES, label_var: str = 'fire', start_date: str = '2010-01-01', drop_vars: list | None = None, pre_stacked: bool = False, include_tv: bool = False, compute: bool = False, clean_nan: bool = True, bonus_proxies: bool = False):
    '''
    takes dataset w/ all required variables + 'fire'
    outputs X, y

    ds might be a dask dataframe; need to account for this too
    '''

    stepped_var_names = ('ws10', 't2m', 'tp')

    if pre_stacked or type(ds) == dd.DataFrame:
        stacked = ds
    else:
        stacked = ds.isel(time = slice(7, None)).stack(sample = ('time', 'step', 'latitude', 'longitude'))

    labels_stacked = stacked[label_var]
    idxs = np.array(range(len(labels_stacked.sample))).reshape(-1, 1)

    X_idxs, y = downsampler.fit_resample(idxs, labels_stacked.data.compute())
    X_idxs = X_idxs.reshape(-1)
    stacked = stacked.isel(sample = X_idxs).persist()

    if drop_vars != None:
        feat = features.copy()
        for var in drop_vars:
            try:
                feat.remove(var)
            except ValueError:
                print(f'Variable {var} not found in features')
        
        features = feat

    stepped_vars = []
    for offset in range(7):
        selection = 7 - offset
        stacked_offset = ds.isel(time = slice(offset, -selection)).stack(sample = ('time', 'step', 'latitude', 'longitude'))
        stacked_offset = stacked_offset.isel(sample = X_idxs)
        for var in stepped_var_names:
            var_name = f'mu_{var}_1'

            da_var = stacked_offset[var_name]
            stepped_vars.append(da_var.data)
    
    stepped_vars = da.array(stepped_vars).transpose()
    features_stacked = stacked[features].persist()

    if type(ds) == xr.Dataset:       

        X = da.append(
            features_stacked.to_array().transpose('sample', 'variable').data,
            stepped_vars,
            axis = 1
        )
    elif type(ds) == dd.DataFrame:
        X = da.append(
            ds[features].to_dask_array().compute(),
            stepped_vars,
            axis = 1
        )

    if include_tv:
        try:
            time_steps = int(len(ds.time) * 24)
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

        X = da.append(X, da.repeat(tv_local, time_steps, axis=0), axis=1)

    ## clean nans
    if clean_nan:
        mask = ~da.isnan(X).any(axis=1) | da.isnan(y)
        X = X[mask]
        y = y[mask]

    if compute:
        return (X.compute(), y.compute())

    else:
        return (X, y)

def prep_samples(ds: xr.Dataset, features: list[str] = ALL_FEATURES, label_var: str = 'fire', start_date: str = '2010-01-01', drop_vars: list | None = None, pre_stacked: bool = False, include_tv: bool = False, compute: bool = False, clean_nan: bool = True):
    '''
    takes dataset w/ all required variables + 'fire'
    outputs X, y

    ds might be a dask dataframe; need to account for this too
    '''

    if pre_stacked or type(ds) == dd.DataFrame:
        stacked = ds.persist()
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

    features_stacked = stacked[features].persist()
    labels_stacked = stacked[label_var]

    if type(ds) == xr.Dataset:       

        samples = len(stacked['sample'])
        chunk_size = int(samples) // 10
        X = features_stacked.to_array().transpose('sample', 'variable').chunk({'sample': chunk_size}).data
        y = labels_stacked.chunk({'sample': chunk_size}).data
    elif type(ds) == dd.DataFrame:
        X = ds[features].to_dask_array().compute()
        y = labels_stacked.to_dask_array().compute()

    if include_tv:

        try:
            time_steps = int(len(ds.time) * 24)
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

        X = da.append(X, np.repeat(tv_local, int(time_steps), axis=0), axis=1)

    ## clean nans
    if clean_nan:
        mask = ~da.isnan(X).any(axis=1) | da.isnan(y)
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