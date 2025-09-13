'''
take data and calculate total windspeed
then add to grib file and write to disk
'''


'''
take in dict of grib files as before
for each, calculate total wind speed and concatenate onto datset? 
also create function that writes ^ dataset collection structure to a series of grib files
'''

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt

from open_data import open_data_dir

def append_wind_speed(data: dict[str, xr.Dataset]):
    '''
    given a dict of datasets (as returned by open_data_dir), adds new variable wind speed ('ws10')
    '''

    new_data = dict()

    for month, month_data in data.items():

        u = month_data['u10']
        v = month_data['v10']

        daily_speeds = []
        for day_full in month_data.time.to_numpy():
            day = pd.Timestamp(day_full).strftime(r'%Y-%m-%d')

            speed = (u.sel(time=day)**2 + v.sel(time=day)**2)**0.5
            daily_speeds.append(speed)

        monthly_speed = xr.concat(
            objs = daily_speeds,
            dim = 'time',
            data_vars = 'all'
        )

        month_data['ws10'] = monthly_speed

        new_data[month] = month_data

    return new_data

if __name__ == '__main__':
    new_data = append_wind_speed(open_data_dir('./data/alaska_TEST_DATA'))

    for key, val in new_data.items():
        print(f'{val['ws10'] = }')
        data = val.isel(time=slice(0, 10, 1), step=0)['ws10']
        print(data.coords)
        graph = data.plot(x='longitude', y='latitude', col='time', col_wrap = 5)

    plt.show()