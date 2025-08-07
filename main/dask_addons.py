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

        self.data = self.clean_na(data)
        self.prior_data = self.clean_na(prior_data)

        ## check we have no repeated days
        assert pd.infer_freq(self.data.time) != None
        assert pd.infer_freq(self.prior_data.time) != None

        self.start_date = pd.to_datetime(min(data.time.values)).to_pydatetime() + timedelta(days = 1)
        self.end_date = pd.to_datetime(max(data.time.values)).to_pydatetime()

        ## grid values
        self.sizes = self.data.sizes
        self.grid_size = self.sizes['latitude'] * self.sizes['longitude']

        self.longitude = self.data.copy().longitude.values
        self.latitude = self.data.copy().latitude.values

        ## features 
        self.data_vars = list(self.data.variables)
        self.input_feature_num = len(self.data_vars)
        self.proxy_num = len(self.proxy_vars) * len(self.proxy_timeframes)
        self.total_features = self.input_feature_num + self.proxy_num

        ## setup
        # self.setup()

    def setup(self):
        '''
        compute proxies and wind speed, then drop u10 and v10, then RETURNS the final dataset
        this is the dataset that should be written as a zarr group

        # Returns
        `(data, prior_data)`
        '''

        self.compute_running_totals()
        self.compute_resultant_speed()
        
        self.data = self.data.drop_vars(('u10', 'v10'))
        self.prior_data = self.prior_data.drop_vars(self.proxy_vars)

    def clean_na(self, ds: xr.Dataset):
        
        ds_stacked = ds.stack(dt = ('time', 'step'))
        ds_clean = ds_stacked.dropna(dim = 'dt', how='all')
        return ds_clean.unstack(dim = 'dt')

    @track_runtime
    def compute_running_totals(self):
        
        for var in self.proxy_vars:
            final_name = f'tot_{var}'

            da_main = self.data[var]
            da_prior = self.prior_data[var]

            da = xr.concat((da_prior, da_main), dim = 'time')

            times = da.time.values.astype("datetime64[h]")
            steps = da.step.values.astype("timedelta64[h]")
            valid_time = times[:, None] + steps[None, :]
            da2 = da.assign_coords(valid_time=(("time", "step"), valid_time)).chunk({"time": 100, "step": -1})
            da_stack = da2.stack(vt=("time", "step"))
            da_roll = da_stack.rolling(vt=720, min_periods=1, center=False).mean().drop_duplicates(dim='vt')
            final_da = da_roll.drop_vars("valid_time").unstack("vt")

            self.data[final_name] = final_da.sel(time = slice(self.start_date + timedelta(days=-1), None))
            self.prior_data[final_name] = final_da.sel(time = slice(None, self.start_date))



    def compute_resultant_speed(self):
        '''
        combine u10 and v10 into a singular resultant wind speed variable, ws10 (wind speed @ 10m)
        '''

        wind_speed = (self.data['u10']**2 + self.data['v10']**2)**0.5

        self.data['ws10'] = wind_speed



    ## HIDDEN METHODS ##

    ### len ###
    def __len__(self):
        hours = (self.end_date - self.start_date).days * 24
        return hours * self.grid_size

    ### full hourly data
    def full_hourly_data(self, time, step):

        grid_data = self.data.sel(time=time, step=step)

        for var in self.proxy_vars:
            tot_name = f'tot_{var}'

            current_total = grid_data[tot_name]

            prev_totals = [self._get_hourly_data(time-timedelta(days=i))[tot_name] for i in range(30, 180, 30)]

            grid_data[f'mu_{var}_30'] = current_total / 30
            grid_data[f'mu_{var}_90'] = (current_total + sum(prev_totals[:2])) / 90
            grid_data[f'mu_{var}_180'] = (current_total + sum(prev_totals)) / 180

        return grid_data


    ### get item / indexing ###
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
            assert len(key) == 2
            index, feature_num = key
            point_data = self._get_single_item(index)
            return self._get_feature(point_data, feature_num)

        elif type(key) == slice:
            raise NotImplementedError

        else:
            raise TypeError(f'Invalid index, {key}, with type {type(key) = }')
        
    def _get_feature(self, sample, var_index):
        '''
        ***copied from `geodataclass.py`***
        ## Parameters
        **sample**: *xarray.Dataset* \\
        should contain a single point and contain **exactly** the variables listed below

        **var_index**: *int* \\
        corresponds to the feature being requested -- consult info subsection for more

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

        - 7,8,9 -- 'mu_tp_30/90/180' -- precipitation proxy vars
        - 10,11,12 -- 'mu_t2m_30/90/180' -- temperature proxy vars
        
        will only store the 30 day totals, then can calculate the 30, 90, 180 day averages from those

        13. 'skt' -- skin temperature -- having issues importing this so might ignore for now
        '''

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
        match m:
            case 0:
                # 30 day average
                return sample[ref_var] / 30
            case 1:
                # 90 day average
                total = sample[ref_var] + sum([self._get_past_data(sample, i)[ref_var] for i in range(30, 90, 30)])
                return total / 90
            
            case 2:
                # 180 day average
                total = sample[ref_var] + sum([self._get_past_data(sample, i)[ref_var] for i in range(30, 180, 30)])
                return total / 180

            case _:
                ## tbh there is something seriously wrong if it gets to here
                raise IndexError(f'Bad code -- end of MonthlyData._get_feature()') 
            
    def _get_past_data(self, sample, days):
        '''
        sample should have single data point (1 lat, long, time, step value)
        '''

        ## check correct form
        dims = sample.sizes
        ## want dims.values() to all be 1

        lat = np.float64(sample['latitude'])
        long = np.float64(sample['longitude'])
        time = pd.to_datetime(sample['time'].values).to_pydatetime()
        step = np.float64(sample['step'].values)

        date = time - timedelta(days = days)

        data = self._get_hourly_data(date)

        return data.sel(
            latitude = lat,
            longitude = long
        )
        
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

        # handle step=24.0 (ie 0:00)
        h = date.hour
        if h == 0:
            step = np.float64(24)
            time = (date + timedelta(days=-1)).strftime(r'%Y-%m-%d')
        else:
            step = np.float64(h + 1)
            time = date.strftime(r'%Y-%m-%d')

        ## choose dataset
        if date < self.start_date:
            source = self.prior_data
        elif date < self.end_date:
            source = self.data
        else:
            raise KeyError(f'Cannot get data for {date = } (after {self.end_date = })')
        
        data = source.sel(time=time, step=step)
        return data
            