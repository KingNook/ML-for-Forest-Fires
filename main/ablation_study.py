import os
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

import xarray as xr
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score

from joblib import dump, load

from random_forest import prep_samples, ALL_FEATURES

main_ds = xr.open_zarr('./data/_ZARR_READY/canada')
test_ds = xr.open_zarr('./data/_ZARR_READY/canada-post')
print('Data loaded')

train_downsampler = RandomUnderSampler(sampling_strategy=1, replacement=False)
test_downsampler = RandomUnderSampler(sampling_strategy=0.1, replacement=False)

X, y = prep_samples(main_ds, compute=True)
X, y = train_downsampler.fit_resample(X, y)
print('Train samples prepped')

X_test, y_test = prep_samples(test_ds, compute=True)
# X_test, y_test = test_downsampler.fit_resample(X_test, y_test)
print('Test samples prepped')

features = np.array(ALL_FEATURES)

with open('canada-post-full.aa_results.txt', 'w') as outfile:

    for i in range(1, 2**8):
        # mu_t2m (x3), mu_tp (x3), d2m, 'sp', 't2m', 'tp', 'ws10', lai (x2), tv (x2)
        choice = f'{i:08b}'

        mask = np.array([bool(int(choice[0]))]*3 + [bool(int(choice[1]))]*3 + [bool(int(choice[i])) for i in range(2, 7)] + [bool(int(choice[7]))]*2)
        
        feature_selection = features[mask]  
        dropped = features[~mask]
        outfile.write(f'Training features: {", ".join(feature_selection)}\n')
        outfile.write(f'Dropped: {", ".join(dropped)}\n')

        train_sel = X[:,mask]
        test_sel = X_test[:,mask]

        clf = AdaBoostClassifier()
        clf.fit(train_sel, y)

        y_pred = clf.predict(test_sel)
        outfile.write(classification_report(y_test, y_pred) + '\n')
        
        delta = 0.58 - f1_score(y_test, y_pred)
        outfile.write(f'Drop in f1 score: {delta}\n')
        print(f'Dropped: {", ".join(dropped)}, delta = {delta:.02f}')