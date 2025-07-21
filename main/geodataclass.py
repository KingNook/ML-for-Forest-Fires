'''
defines a new array-like data type

main advantage is this should allow us to treat a lazily loaded set of xr.datasets as array-like objects, so we can pass it directly into our ml model
'''

import xarray as xr
import numpy as np
import datetime

class MonthlyData:
    '''
    currently returns the whole grid for all variables for given day
    potentially useful for unet / cnn so will leave like this
    
    need different class for inidividual datapoints for rf / dnn though
    '''

    def __init__(self, data):
        '''
        data should be a dict with:
        - key -- of the form `yyyy-mm`
        - value -- an `xarray.dataset` representing the data for that month
            - should have datapoints for each day in the month under the header `['time']`

        also should define the start date as the first day of the first month for which we have data
        '''

        self.key_format = '%Y-%m'

        ## NEED DATA CHECKS
        months = data.keys()

        ## check keys are in yyyy-mm format
        months = [datetime.datetime.strptime(month, self.key_format) for month in months]

        self.start_date = min(months)

        self.data = data

    def __getitem__(self, index):
        '''
        should return the item from start_date + index days
        '''

        ## check if we have enough days (ie check for index error here)
        pass

        reference_date = self.start_date + datetime.timedelta(days = int(index))
        ref_key = reference_date.strftime(self.key_format)
        full_key = reference_date.strftime(r'%Y-%m-%d')

        return self.data[ref_key].sel(time=full_key)

    def __iter__(self):
        return CustomIterator(self)

class CustomIterator:
    '''
    just in case if array_like is called as an iterable
    '''

    def __init__(self, data):
        self.idx = 0
        self.data = data

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1

        try:
            return self.data[self.idx - 1]
        
        except IndexError:
            self.idx = 0
            raise StopIteration
        
class Flattened_MonthlyData:
    '''
    ith index will take only one datapoint (one lat/long coord from 1 hour from 1 day)
    '''

    def __init__(self, data):
        '''
        data should be `xr.DataSet` with fields:
        - time (datetime)
        - step (float64, ranges from 1.0 to 24.0 -- represents which hour of the day)
        - latitude
        - longitude

        idea is to flatten wrt the lat/long/time axes
        '''

        self.data = MonthlyData(data)

        self.dims = self.data[0].sizes ## SHOULD CHECK THAT THIS IS CONSISTENT ACROSS ALL DATAPTS

        self.internal_size = np.prod(list(self.dims.values()))

    def __getitem__(self, index):
        '''
        goes down by:
        - year
        - month
        - day
        - hour
        - lat
        - long
        '''

        indices = divmod(index, self.internal_size)

        daily_data = self.data[indices[0]]

        hour_sel = divmod(indices[1], self.dims['step']) ## should be 24
        lat_sel = divmod(hour_sel[1], self.dims['latitude'])
        long_sel = lat_sel[1]

        hours = daily_data.step
        lat = daily_data.latitude
        long = daily_data.longitude

        return daily_data.sel(
            step=hours[hour_sel[0]],
            latitude=lat[lat_sel[0]],
            longitude=long[long_sel]
        )

    def __iter__(self):
        return CustomIterator(self)