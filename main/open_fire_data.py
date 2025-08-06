import xarray as xr
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class FlattenedTruthTable:
    '''
    holds a dataframe and a coord range -- translates index into long/lat/time coords, then checks against table to see if there is an entry
    '''

    def __init__(self, data: pd.DataFrame, lat_vals: list[np.float64], long_vals: list[np.float64], start_date: str | datetime = '2010-01-01'):
        '''
        provides reformatting and easy indexing to pandas dataframe
        ## Parameters
        **data**: *pandas.DataFrame* \\
        from `pd.read_csv()`, contains data for all detected fires; expected to have columns for:
        - latitude
        - longitude
        - acq_date
        - acq_time
        - confidence
        - type

        **long_vals**, **lat_vals**: *array-like* \\
        longitude and latitude values -- should be the same as those for the input dataset; can rip them
        using `ds.latitude/longitude.values` where `ds` is a `FlattenedDaskDataset` instance

        **start_date**: *str* or *datetime* \\
        the 0th item will be at 0:00 on `start_date`; can rip this using `ds.start_date`
        '''
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