import xarray as xr
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

if __name__ == '__main__':
    fire_data = pd.read_csv(
        './data/FIRE/alaska_range_csv/data.csv',
        parse_dates = ['acq_date']
    )

    veg_files = fire_data[
        (fire_data['type'] == 0) & (fire_data['confidence'] >= 60)
    ]

    important_vars = [
        'latitude',
        'longitude',
        'acq_date',
        'acq_time'
    ]

    var_series = [
        veg_files[var] for var in important_vars
    ]

    var_series[0] = var_series[0].round(1)
    var_series[1] = var_series[1].round(1)
    var_series[2] = var_series[2]
    var_series[3] = (var_series[3]/100).apply(np.ceil).astype(np.float64)

    fire_ds = pd.concat(
        objs = var_series,
        axis = 1,
        names = important_vars
    )

    print(fire_ds)

class FlattenedTruthTable:
    '''
    holds a dataframe and a coord range -- translates index into long/lat/time coords, then checks against table to see if there is an entry
    '''

    def __init__(self, data, start_date = '2010-01-01'):
        self.data = data
        self.grid_shape = data.shape

        self.data['latitude'] = data['latitude'].round(1)
        self.data['longitude'] = data['longitude'].round(1)
        self.data['acq_time'] = np.ceil(data['acq_time']/100).astype(np.float64)

        if type(start_date) == str:
            self.start_date = datetime.strptime(start_date, r'%Y-%m-%d')
        elif type(start_date) == datetime:
            self.start_date = start_date


    def check_match(self, long, lat, time, date):
        fire_series = self.data[
            (self.data['longitude']==long) & (self.data['latitude']==lat) & (self.data['acq_time']==time) & (self.data['acq_date']==date)
        ]

        if fire_series.empty:
            return 0
        else:
            return 1

    def __getitem__(self, key):

        grid_cells = int(np.prod(self.grid_shape))
        
        ## time/space split -- quotient is hour index, remainder is corr cell
        ts_split = divmod(key, grid_cells) 
        long, lat = divmod(ts_split[1], self.grid_shape[1])
        days, hour = divmod(ts_split[0], 24)

        date = self.start_date + timedelta(days)

        ## return 1 if yes fire and 0 if not
        return self.check_match(long, lat, hour, date)