from sklearnex import patch_sklearn
patch_sklearn()

import os
os.environ["SCIPY_ARRAY_API"] = "1"
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import load, dump
import dask

from random_forest import prep_samples, plot_feature_distribution
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline
from importlib import reload

from IPython.display import HTML

ds = xr.open_zarr('./data/_ZARR_READY/canada')
X, y = prep_samples(ds, include_tv=True, compute=True)

downsampler = RandomUnderSampler(sampling_strategy=0.1)

X_resampled, y_resampled = downsampler.fit_resample(X, y)

clf = AdaBoostClassifier()
clf.fit(X_resampled, y_resampled)

dump(clf, './main/models/can.adaboost.new_order.joblib')