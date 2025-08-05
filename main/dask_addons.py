import time
import xarray as xr
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tools import hourlydaterange, track_runtime

def is_final_day(date: datetime):
    '''
    checks if `date` is the final day of the month
    '''

    return date == date + relativedelta(day = 31)



class FlattenedDaskDataset:
    '''
    takes in dataset, indexing treats this as a flattened dataset; cannot convert directly to numpy for memory reasons 
    '''

    proxy_vars = ('tp', 't2m')
    proxy_timeframes = (30, 90, 180)

    def __init__(self, data: xr.Dataset, prior_data: xr.Dataset):
        '''
        Use `xr.open_zarr()` to get Datasets from the zarr groups;
        try to use `decode_timedelta=False` so that `step` is given as `np.float64`

        # Parameters
        **data**: *xarray.Dataset* \\
        data for the main time-range of interest (should be the same time range for which fire data has been downloaded)

        **prior_data**: *xarray.Dataset* \\
        data for at least 180 days before the start of the period of study
        '''

        self.data = data
        self.prior_data = prior_data

        self.start_date = pd.to_datetime(min(data.time.values)).to_pydatetime() + relativedelta(hour=23)
        self.end_date = pd.to_datetime(max(data.time.values)).to_pydatetime()

        ## grid values
        self.sizes = self.data.sizes
        self.grid_size = self.sizes['latitude'] * self.sizes['longitude']

        example_pt = self._get_hourly_data(0)
        self.longitude = example_pt.longitude.values
        self.latitude = example_pt.latitude.values

        ## features 
        self.data_vars = list(self.data.variables)
        self.input_feature_num = len(self.data_vars)
        self.proxy_num = len(self.proxy_vars) * len(self.proxy_timeframes)
        self.total_features = self.input_feature_num + self.proxy_num

    def setup(self):
        '''
        compute proxies and wind speed, then drop u10 and v10, then RETURNS the final dataset
        this is the dataset that should be written as a zarr group

        # Returns
        `(data, prior_data)`
        '''

        self.compute_running_totals()
        self.compute_resultant_speed()

        return (self.data.drop_vars(('u10', 'v10')), self.prior_data.drop_vars(self.proxy_vars))

    @track_runtime
    def compute_running_totals(self):
        date_format = r'%Y-%m-%d'

        warmup_start_date = self.start_date - timedelta(days = 180)
        prior_start_date = warmup_start_date + timedelta(days = 30)
        end_date = self.end_date
        prior_period = 24*150

        main_ds = []
        prior_ds = []

        for var in self.proxy_vars:
            print(list(self.data.variables))
            start_time = time.time()
            final_name = f'tot_{var}'

            print(f'[HourlyData] computing proxy: {var = }')

            rolling_total = []

            initial_da = self._get_hourly_data(warmup_start_date)[var]

            for date in hourlydaterange(warmup_start_date, prior_start_date):                
                new_data = self._get_hourly_data(date)[var] ## add something to point out potentially problematic datapoints (ie if the whole thing is NaN)
                initial_da = initial_da + new_data.reset_coords('time', drop=True) 

            rt_np = initial_da.to_numpy().reshape((1, 1, self.sizes['latitude'], self.sizes['longitude']))

            da = xr.DataArray(
                    data = rt_np,
                    dims = ['time', 'step', 'latitude', 'longitude'],
                    coords = {
                        'latitude': self.latitude,
                        'longitude': self.longitude,
                        'time': [np.datetime64(prior_start_date.strftime(date_format))],
                        'step': [np.float64(prior_start_date.hour+1)]
                    },
                    name = final_name
                )
            
            rolling_total.append(da.stack(dt = ['time', 'step']))
            
            old_total = rt_np

            n = 0
            for date in hourlydaterange(prior_start_date, end_date):
                if n % 720 == 0:
                    print(f'[compute rt | {time.time() - start_time:.02f}s elapsed] currently on {date.strftime(r'%Y-%m-%d %H:%M:%S')}')
                n += 1

                old_data = self._get_hourly_data(date - timedelta(days=30))[var].to_numpy().reshape(old_total.shape)
                new_data = self._get_hourly_data(date)[var].to_numpy().reshape(old_total.shape)

                new_total_np = old_total - old_data + new_data
                old_total = new_total_np

                new_total = xr.DataArray(
                    data = new_total_np,
                    dims = ['time', 'step', 'latitude', 'longitude'],
                    coords = {
                        'latitude': self.latitude,
                        'longitude': self.longitude,
                        'time': [np.datetime64(date.strftime(date_format))],
                        'step': [np.float64(date.hour+1)]
                    },
                    name = final_name
                )

                rolling_total.append(new_total.stack(dt = ['time', 'step']))

            print('totals calculated')

            prior_totals = xr.concat(
                objs = rolling_total[:prior_period],
                dim = 'dt'
            )
            prior_totals = prior_totals.unstack('dt')
    
            prior_ds.append(prior_totals)

            totals = xr.concat(
                objs = rolling_total[prior_period:],
                dim = 'dt'
            )
            totals = totals.unstack('dt')
            main_ds.append(totals)

        # use this if above fails
        for ds in prior_ds:
            self.prior_data.merge(ds)

        for ds in main_ds:
            self.data.merge(ds)

    def compute_resultant_speed(self):
        '''
        combine u10 and v10 into a singular resultant wind speed variable, ws10 (wind speed @ 10m)
        '''

        wind_speed = (self.data['u10']**2 + self.data['v10']**2)**0.5

        self.data['ws10'] = wind_speed



    ## HIDDEN METHODS ##

    def __getitem__(self, key):
        '''
        acceptable types of indexing:
        - `int` **i**: returns ith item, looking W->E by longitude then N->S by latitude *to be confirmed*
        - `tuple` **i, j**: returns jth feature of ith item
        - `slice` **slice(start, stop, step)**: behaviour as expected
        '''

        if type(key) == int:
            return self._get_single_item(key)

        elif type(key) == tuple:
            point_data = self._get_single_item(key[0])
            raise NotImplementedError

        elif type(key) == slice:
            raise NotImplementedError

        else:
            raise TypeError(f'Invalid index, {key}, with type {type(key) = }')
        
    def _get_single_item(self, index):
        
        time_space_split = divmod(index, self.grid_size)

        hourly_data = self._get_hourly_data(time_space_split[0])

        long_lat = divmod(time_space_split[0], self.sizes['latitude']) ## should be (longitude, latitude)

        return hourly_data.isel(
            longitude = long_lat[0],
            latitude = long_lat[1]
        )

    def _get_hourly_data(self, index: int | datetime) -> xr.Dataset:
        
        if type(index) == int:
            date = self.start_date + timedelta(hours = index)
        elif type(index) == datetime:
            date = index
        elif type(index) == pd.Timestamp:
            date = index.to_pydatetime()
        else:
            raise TypeError(f'Unsupported type, {type(index) = }')

        time = np.datetime64(date.strftime(r'%Y-%m-%d'))

        if date <= self.start_date:
            data =  self.prior_data.sel(
                time = time,
                step = np.float64(date.hour + 1)
            )

        elif date < self.end_date:
            data = self.data.sel(
                time = time,
                step = np.float64(date.hour + 1)
            )
        else:
            raise KeyError(f'Cannot get data for {date = } (after {self.end_date = })')
        
        try:
            if len(data.coords['time']) == 2:
                return data.isel(time=1)
            else:
                raise ValueError(f'wacky time values: {data.coords['time']}')
            
        except:
            ## only 1 time value
            return data
            