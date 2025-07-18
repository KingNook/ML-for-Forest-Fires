from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

x_data = np.array([1, 2, 4]).reshape(-1, 1)
y_data = np.array([1, 4, 16])

rf = RandomForestClassifier()

rf.fit(x_data, y_data)

with open('model_file.txt', 'wb') as model_file:

    pickle.dump(rf, file = model_file)