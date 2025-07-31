print('prog start')
import pickle
from sklearn.ensemble import RandomForestClassifier
from geodataclass import FlattenedData
from open_data import open_data_dir
from compute_wind_speed import append_wind_speed
from open_fire_data import fire_ds, FlattenedTruthTable
print('imports done')

prior_data = open_data_dir('./data/alaska_prior')
data = open_data_dir('./data/alaska_main')

print('data open')

input_data = FlattenedData(append_wind_speed(data), prior_data)
target_data = FlattenedTruthTable(fire_ds, grid_shape = input_data.grid_shape)

print('data flat')

## train model

print('model trained')

with ('model.pickle', 'wb') as model_file:

    pickle.dump(rf, model_file)