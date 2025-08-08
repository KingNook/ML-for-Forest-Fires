import xarray as xr
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

ds = xr.open_zarr('./data/_ZARR_READY/la_main_data', decode_timedelta=False)
# prior_data = xr.open_zarr('./data/_ZARR_READY/la_prior_data')

features = ['d2m', 'lai_hv', 'lai_lv', 'sp', 't2m', 'tot_t2m', 'tot_tp', 'tp', 'ws10']
stacked = ds.stack(sample = ('time', 'step', 'latitude', 'longitude')).dropna(dim = 'sample')

X = np.stack([
    stacked[var].values for var in features
], axis = 1)
y = stacked['fire'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

dump(clf, 'test_model.joblib')