from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import xarray as xr

## DATA VALIDATION FUNCTIONS ##
def validate_month_format(data, key_format):
    '''
    check:
    - keys are correct format
    - no missing months
    '''

    pass

## CUSTOM ITERATOR CLASS ##
class CustomIterator:

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

## CUSTOM CLASSES ##
class MonthlyData:
    '''
    Class to allow array-like access (indexed monthly) to a collection of DataSets
    '''

    def __init__(
            self,
            data: dict[str, xr.Dataset],
            region: str = ''
    ):
        '''
        input:
        - `data` -- a dictionary with:
            - keys (`str`) of the form `yyyy-mm` representing the month of the data
            - values (`xr.DataSet`) data from the corresponding month
        - `region` -- geographical region which the data describes
        '''
        
        ## should validate data:
        # no missing datapoints
        # all the same dimensions (lat/long/time)

        self.data = data

        self.key_format = f'%Y-%m'
        
        months = tuple(data.keys())

        self.sizes = data[months[0]].sizes

        self.grid_shape = (self.sizes['longitude'], self.sizes['latitude'])

        validate_month_format(months, self.key_format)

        self.months = [
            datetime.strptime(month, self.key_format) for month in months
        ]

        self.start_date = min(self.months)
        self.end_date = max(self.months) + relativedelta(day=31)

        self.period = (self.end_date - self.start_date).days

    ## DATA VALIDATION ##
    def _validate_index(self, i: int) -> int:
        '''
        validates <i> as an index, and returns i as an int type
        (since it may potentially come in as a weird int-like datatype eg np.int64)
        '''

        # check i int-like
        if int(i) != i:
            raise TypeError(f'Index {i} insufficiently int-like')

        # adjust for negatives
        if i < 0:
            index = int(i + len(self) + 1)
        else:
            index = int(i)

        # index should initially lie within [-length, +length)
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} out of range for {__name__} of length {len(self)}')
        
        return index

    ## HIDDEN METHODS ##
    def __len__(self):
        '''
        number of MONTHS in dataset
        '''

        return len(self.months)

    def __getitem__(self, index):
        if isinstance(index, slice):
            
            ## handle empty values eg [:-1]
            start = index.start if index.start else 0
            step = index.step if index.step else 1

            if index.stop:
                if index.stop > 0:
                    stop = index.stop
                else:
                    stop = len(self) + index.stop
            else:
                stop = len(self)
            
            return [self._get_single_item(i) for i in range(start, stop, step)]

        else:
            i = self._validate_index(index)
            return self._get_single_item(i)
        
    def _get_single_item(self, i: int):
        '''
        return Dataset from <i>th month
        '''

        ## THE REST OF THE FUNCTION ##
        reference_date = self.start_date + relativedelta(months = i)
        key = reference_date.strftime(self.key_format)

        return self.data[key]
    
    def __iter__(self):
        return CustomIterator(self)


class DailyData(MonthlyData):
    '''
    Class to allow array-like access (indexed daily) to a collection of DataSets
    '''

    def __init__(self, data):
        super().__init__(data)

    def __len__(self):
        return self.period

    def _get_single_item(self, i: int):
        '''
        get Dataset from <i>th day
        '''

        ref_date = self.start_date + timedelta(days = i)
        month_key = ref_date.strftime(self.key_format) # key for self.data
        time_key = ref_date.strftime(r'%Y-%m-%d') # matches format of 'time' column in dataset

        return self.data[month_key].sel(time=time_key)


class HourlyData(DailyData):
    '''
    Class to allow array-like access (indexed hourly) to a collection of DataSets
    '''

    def __init__(self, data):
        super().__init__(data)

    def __len__(self):
        return super().__len__() * 24
    
    def _get_single_item(self, i: int):

        ref_date = self.start_date + timedelta(hours=i)
        month_key = ref_date.strftime(self.key_format)
        time_key = ref_date.strftime(r'%Y-%m-%d')
        step_key = ref_date.hour + 1.0 # want this to be a float, and ranges from 1.0 to 24.0

        return self.data[month_key].sel(
            time=time_key,
            step=step_key
        )


class FlattenedData(HourlyData):
    '''
    Class to allow array-like access (indexed point-wise wrt lat/long) to a collection of DataSets
    '''

    def __init__(self, data):
        super().__init__(data)
    
    def _get_single_item(self, i):

        # np.prod returns a np.int64 object which causes issues for some methods
        # but can be converted into a python int object w/out losing any information
        grid_cells = int(np.prod(self.grid_shape))
        
        ## time/space split -- quotient is hour index, remainder is corr cell
        ts_split = divmod(i, grid_cells) 

        hour_data = super()._get_single_item(ts_split[0])

        long, lat = divmod(ts_split[1], self.grid_shape[1])

        return hour_data.sel(
            longitude = long,
            latitude = lat
        )