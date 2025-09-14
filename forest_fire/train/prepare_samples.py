'''
Given a geospatial Xarray dataset, prepare samples that can be passed into an sklearn model, ie numpy array / numpy-backed dask array (w/ chunk sizes computed)

This covers code from
- random_forest.py
'''
import numpy as np
import pandas as pd
import xarray as xr

from xarray import Dataset

import dask
import dask.array as da
import dask.dataframe as dd

VEGETATION_TYPE_PATH = './data/_ZARR/veg_types'

ORIGINAL_FEATURES = ['d2m', 'lai_hv', 'lai_lv', 'mu_t2m_180', 'mu_t2m_30', 'mu_t2m_90', 'mu_tp_90', 'mu_tp_30', 'sp', 'mu_tp_180', 't2m', 'tp', 'ws10']

def prep_samples(
        ds: Dataset, 
        features: list[str] = ORIGINAL_FEATURES, 
        label_var: str = 'fire',
        pre_stacked: bool = False, 
        include_tv: bool = False, 
        compute: bool = False,
        compute_chunks: bool = False, 
        clean_nan: bool = True
    ) -> tuple:
    '''
    Takes a Dataset containing relevant features + a label variable, and returns an (X, y) pair that can be used for training / testing / whatever else

    IF THIS IS IS FAILING DUE TO MEMORY CONSTRAINTS, check whether distributed computing is being used -- for HPC this might be better but for a local machine, this greatly decreases the limit on how large an object can be (since if we have n workers, each worker only gets 1/n of the total available memory)

    Parameters
    ----------
    ds: Dataset or DataFrame
        Data structure containing feature and label data. Built for an Xarray Dataset so that will be more consistent, but pandas/dask DataFrames should also be fine

    features: list[str], optional
        List of features that will go into final samples. Note that the order of `features` will match the columns of the final array. By default, this is the list used for the final adaboost model 

    label_var: str, optional
        The variable in ds that corresponds to labels. By default, this is 'fire'

    pre_stacked: bool, optional
        Whether the Dataset is already stacked (there should be a MultiIndex, 'sample', if it is stacked). By default, this is False

    include_tv: bool, optional
        Whether to include type of vegetation -- temporally static dataset so it doesn't make sense to include it in large ds. By default, this is False

        NOTE IF THIS IS SET TO True, the VEGETATION_TYPE_PATH constant at the top of this file must be changed to the path to a Zarr group w/ tvh and tvl variables

    compute: bool, optional
        Whether to call `.compute()` on X and y before returning. By default, this is False.

        This takes precedence over `compute_chunks`

    compute_chunks: bool, optional
        Whether to call `.compute_chunk_sizes()` on X and y. By default, this is False

        Note if `compute=True`, then this is irrelevant.

    clean_nan: bool, optional
        Whether to remove rows where any feature contains NaN -- may be necessary depending on the model used. By default, this is True

    Returns
    -------
    training_set: (X, y)
        X will be an n-dimensional numpy array (or numpy-backed dask array) in the shape (n_samples, n_features)

        y will be a 1-dimensional numpy array in the shape (n_samples,)
    '''

    if pre_stacked or type(ds) == dd.DataFrame:
        stacked = ds.persist()
    else:
        stacked = ds.stack(sample = ('time', 'step', 'latitude', 'longitude'))

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
            tv = xr.open_zarr(VEGETATION_TYPE_PATH)
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
    elif compute_chunks:
        return (X.compute_chunk_sizes(), y.compute_chunk_sizes())

    else:
        return (X, y)