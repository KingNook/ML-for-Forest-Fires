from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import xarray as xr
import pandas as pd

from typing import Literal

from tools import track_runtime

def hourlydaterange(start_date, end_date):
    hours = (end_date - start_date).days * 24
    for n in range(0, hours):
        n += 1
        yield start_date + timedelta(hours = n)

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

    key_format = f'%Y-%m'

    def __init__(
            self,
            data: dict[str, xr.Dataset],
            prior_data: dict[str, xr.Dataset],
            region: str = ''
    ):
        '''
        input:
        - `data` -- **required** a dictionary with:
            - keys (`str`) of the form `yyyy-mm` representing the month of the data
            - values (`xr.DataSet`) data from the corresponding month
        - `prior_data` -- **required** data from 180 days (at least) before the start date of `data`; necessary if doing 30/90/180 day averages
        - `region` -- *optional* geographical region which the data describes
        '''
        
        ## should validate data:
        # no missing datapoints
        # all the same dimensions (lat/long/time)

        self.data = data
        self.prior_data = prior_data
        
        months = tuple(data.keys())

        self.variables = list(data[months[0]].keys())

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    proxy_vars = ('tp', 't2m')

    def __init__(self, *args, **kwargs):
        print(f'[{self.__class__}] begin init')
        super().__init__(*args, **kwargs)

        self._compute_proxies(self.proxy_vars)

        print(f'[{self.__class__}] finish init')

    def __len__(self):
        return super().__len__() * 24
    
    def _get_hourly_data_by_date(self, ref_date: datetime):
        month_key = ref_date.strftime(self.key_format)
        time_key = ref_date.strftime(r'%Y-%m-%d')
        step_key = ref_date.hour + 1.0 # want this to be a float, and ranges from 1.0 to 24.0

        if ref_date >= self.start_date:
            return self.data[month_key].sel(
                time=time_key,
                step=step_key
            )
        else:
            return self.prior_data[month_key].sel(
                time=time_key,
                step=step_key
            )
    
    def _get_single_item(self, i: int):

        ref_date = self.start_date + timedelta(hours=i)

        return self._get_hourly_data_by_date(ref_date)
    
    ## init stuff ##
    @track_runtime
    def _compute_proxies(self, var_names: list[str]):
        '''
        compute proxy vars and add them to dataset
        '''
        print(f'computing proxes: {var_names = }')
        date_form = r'%Y-%m-%d'

        final_names = [f'tot_{var_name}' for var_name in var_names]

        warmup_start_date = self.start_date - timedelta(days = 180)
        prior_start_date = warmup_start_date + timedelta(days = 30)
        end_date = self.end_date
        prior_period = 24*150 # ie all days up to self.start_date // start date of actual data
        
        pt = []
        t = []

        for i in range(len(var_names)):
            var_name = var_names[i]

            rolling_total = [self._get_hourly_data_by_date(warmup_start_date)[var_name]]

            for date in hourlydaterange(warmup_start_date, prior_start_date):
                new_data = self._get_hourly_data_by_date(date)[var_name].fillna(0) ## add something to point out potentially problematic datapoints
                rolling_total[0] = rolling_total[0] + new_data

            x_coords = rolling_total[0].longitude.values
            y_coords = rolling_total[0].latitude.values

            rt_np = rolling_total[0].to_numpy().reshape((1, 1, self.grid_shape[1], self.grid_shape[0]))

            rolling_total[0] = xr.DataArray(
                    data = rt_np,
                    dims = ['time', 'step', 'latitude', 'longitude'],
                    coords = {
                        'latitude': y_coords,
                        'longitude': x_coords,
                        'time': [np.datetime64(prior_start_date.strftime(date_form))],
                        'step': [np.float64(prior_start_date.hour+1)]
                    },
                    name = final_names[i]
                )
            
            for date in hourlydaterange(prior_start_date, end_date):

                old_total = rolling_total[-1].to_numpy()

                old_data = self._get_hourly_data_by_date(date - timedelta(days=30))[var_name].fillna(0).to_numpy().reshape(old_total.shape)
                new_data = self._get_hourly_data_by_date(date)[var_name].fillna(0).to_numpy().reshape(old_total.shape)

                new_total_np = old_total - old_data + new_data

                new_total = xr.DataArray(
                    data = new_total_np,
                    dims = ['time', 'step', 'latitude', 'longitude'],
                    coords = {
                        'latitude': y_coords,
                        'longitude': x_coords,
                        'time': [np.datetime64(date.strftime(date_form))],
                        'step': [np.float64(date.hour+1)]
                    },
                    name = final_names[i]
                )

                rolling_total.append(new_total)

            for i in range(len(rolling_total)):

                rolling_total[i] = rolling_total[i].stack(datetime = ['time', 'step'])

            prior_totals = xr.concat(
                objs = rolling_total[:prior_period],
                dim = 'datetime',
                data_vars = 'all'
            )
            # Stack 'time' and 'step' into a single MultiIndex dimension
            prior_totals = prior_totals.unstack('datetime')
            pt.append(prior_totals)

            totals = xr.concat(
                objs = rolling_total[prior_period:],
                dim = 'datetime',
                data_vars = 'all'
            )
            totals = totals.unstack('datetime')
            t.append(totals)

        for i in range(len(final_names)):
            final_name = final_names[i]

            self._add_variable_prior_data(final_name, pt[i])
            self._add_variable_data(final_name, t[i])

            print(f'added proxy var {final_name = }')


    def _add_variable_data(self, var_name, data):
        '''
        given a dataset, add relevant month to each entry in dataset
        '''
        
        for month in self.data.keys():
            start_date = datetime.strptime(month, self.key_format)
            end_date = start_date + relativedelta(day = 31)

            monthly_data = data.sel(
                time = slice(start_date, end_date)
            )

            self.data[month][var_name] = monthly_data
            print(f'updated data for {month = }')

    def _add_variable_prior_data(self, var_name, data):
        '''
        given a dataset, add relevant month to each entry in dataset
        '''

        
        for month in self.prior_data.keys():

            try:
                monthly_data = data.sel(
                    time = month
                )

                self.prior_data[month][var_name] = monthly_data
                print(f'updated data for {month = }')
            except Exception as e:
                print(f'No data for {month = } // {e = }')

            

class FlattenedData(HourlyData):
    '''
    Class to allow array-like access (indexed point-wise wrt lat/long) to a collection of DataSets

    __init__ takes the following inputs:
    - `data` -- **required** a dictionary with:
        - keys (`str`) of the form `yyyy-mm` representing the month of the data
        - values (`xr.DataSet`) data from the corresponding month
    - `prior_data` -- **required** data from 180 days (at least) before the start date of `data`; necessary if doing 30/90/180 day averages
    - `region` -- *optional* geographical region which the data describes
        
    '''

    @track_runtime
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

        self.feature_num = 13

        # self.load()

    def load(self):
        for ds in self.prior_data.values():
            ds.load()
        
        for ds in self.data.values():
            ds.load()

    ## HIDDEN METHODS ##
    def __len__(self):

        hours = super().__len__()
        grid_size = np.prod(self.grid_shape)

        return int(hours * grid_size)

    def _sum_backwards(
            self,
            ref_var: str,
            period: int,
            current_date: datetime,
            coords: tuple[np.float64, np.float64]
        ):
        '''
        ## input
        - `ref_var` -- either `tot_tp` or `tot_t2m`
        - `period` -- must be either `90` or `180`
        - `current_date` -- start point
        - `coords` -- `(longitude, latitude)` pair of coordinates
        '''

        data_points = [
            self._get_local_data(
                data = self._get_hourly_data_by_date(current_date - timedelta(days=i)),
                coords = coords
            )[ref_var] for i in range(0, period, 30)
        ]

        return sum(data_points)

    def _get_feature(self, sample, var_index):
        '''
        ## inputs
        - `sample` -- DataSet with single time, step, latitude, longitude coordinate;
            - must contain variables listed in info section below

        ## info
        custom mapping between feature index and feature (short) name
        necessary because i have been a nicompoop and i need some way to access post-calculation data by index (ie wind speed)
        0. 'tp' -- total precipitation
        1. 'd2m' -- 2m dewpoint
        2. 't2m' -- 2m temperature
        3. 'lai_hv' -- lai for high vegetation
        4. 'lai_lv' -- lai for low vegetation
        5. 'sp' -- surface pressure
        
        6. 'ws10' -- total windspeed (will be a combination of u10 and v10) // should be added in advance

        - 7,8,9 -- 'mu_p_30/90/180' -- precipitation proxy vars
        - 10,11,12 -- 'mu_t_30/90/180' -- temperature proxy vars
        
        will only store the 30 day totals, then can calculate the 30, 90, 180 day averages from those

        13. 'skt' -- skin temperature -- having issues importing this so might ignore for now
        '''
        ## VERIFICATION -- check there is only 1 time and step value


        fixed_vars = ['tp', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'sp', 'ws10']

        n = len(fixed_vars) # future proofing in case if i do end up adding in skin temperature
        if var_index < n:
            var_name =  fixed_vars[var_index]
            return sample[var_name]

        ## proxy vars (further calculation needed)
        elif n <= var_index < n + 3:
            ## precipitation
            m = var_index - n
            var_name = 'tp'

        elif n + 3 <= var_index < n + 6:
            ## temperature
            m = var_index - n - 3
            var_name = 't2m'
        else:
            ## nb skin temp will raise an error for now
            raise IndexError(f'Unknown feature number, {var_index = }')
        
        ref_var = f'tot_{var_name}' # 30 day total
        current_date = sample.time.values ## should give np.datetime64 since only 1 value
        coords = (np.float64(sample.latitude.values), np.float64(sample.longitude.values))
        match m:
            case 0:
                # 30 day average
                return sample[ref_var] / 30
            case 1:
                # 90 day average
                total = self._sum_backwards(ref_var, 90, current_date, coords)
                return total / 90
            
            case 2:
                # 180 day average
                total = self._sum_backwards(ref_var, 180, current_date, coords)
                return total / 180

            case _:
                ## tbh there is something seriously wrong if it gets to here
                raise IndexError(f'Bad code -- end of MonthlyData._get_feature()') 
    
    def __getitem__(self, key) -> xr.DataArray | xr.Dataset | list[xr.Dataset]:
        '''
        add support for tuple-type indexing
        '''

        if type(key) == tuple: # datarray branch
            '''
            key will be (sample_num, feature_num)
            '''
            if len(key) != 2:
                raise KeyError(f'Invalid key: {key} -- should be of the form (sample, feature)')

            sample_num, feature_num = key

            sample = super().__getitem__(sample_num)

            return self._get_feature(sample, feature_num)


        else: # dataset / list of datasets branch
            return super().__getitem__(key)
        
    def _get_local_data(self, data, coords):
        '''
        coords are (long, lat)
        '''

        long_vals = data.longitude
        lat_vals = data.latitude

        return data.sel(
            longitude = long_vals[coords[0]],
            latitude = lat_vals[coords[1]]
        )


    def _get_single_item(self, i):

        # np.prod returns a np.int64 object which causes issues for some methods
        # but can be converted into a python int object w/out losing any information
        grid_cells = int(np.prod(self.grid_shape))
        
        ## time/space split -- quotient is hour index, remainder is corr cell
        ts_split = divmod(i, grid_cells) 
        long, lat = divmod(ts_split[1], self.grid_shape[1])

        hour_data = super()._get_single_item(ts_split[0])

        return hour_data.isel(
            longitude = long,
            latitude = lat
        )