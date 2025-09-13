from sklearnex import patch_sklearn
patch_sklearn()

import os
os.environ["SCIPY_ARRAY_API"] = "1"

import xarray as xr
from open_data import open_data_dir, data_dir_to_zarr
from compute_wind_speed import append_wind_speed
from dask_addons import FlattenedDaskDataset

import time
import seaborn as sns

from random_forest import *
from imblearn.under_sampling import RandomUnderSampler

ds = xr.open_zarr('./data/_ZARR_READY/la_forest')
X, y = prep_samples(ds, include_tv=True, compute=True)
downsampler = RandomUnderSampler(sampling_strategy=1, replacement=False)
X_res, y_res = downsampler.fit_resample(X, y)
from random_forest import PRINT_FEATURES


feat_df = pd.DataFrame(
    np.append(X, y.reshape(-1, 1), axis=1),
    columns=PRINT_FEATURES
)
pred_df = pd.DataFrame(
    np.append(X_res, y_res.reshape(-1, 1), axis=1),
    columns=PRINT_FEATURES
)
feature = 'tp'
og_vals = pd.DataFrame(feat_df[feature])
pred_vals = pd.DataFrame(pred_df[feature])
og_vals['source'] = 'original'
pred_vals['source'] = 'resampled'
plot_df = pd.concat([og_vals, pred_vals]).reset_index(drop=True)
fig, axs = plt.subplots(8, 2, figsize=(20, 50))

for i, var in enumerate(PRINT_FEATURES):
    if i == 15:
        break
    x, y = divmod(i, 8)
    curr_df = feat_df[['fire', var]].dropna(axis=0, how='any')
    sns.histplot(curr_df, x=var, hue='fire', bins=20, ax=axs[y, x], stat='density')

plt.show()