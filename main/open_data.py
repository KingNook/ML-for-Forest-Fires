'''
given a directory, reads the data into some sort of datastructure

directory should look like: data/<dir_name>/<yyyy>-<mm>_<data_name>
'''

import cfgrib

import os
import xarray as xr

from geodataclass import *
from request_climate_data import track_runtime
from unzippify import temp_cwd

@track_runtime
def read_grib_file(data_path):
    '''
    sub_function -- reads data_path/data.grib, returns grib file (opened)
    '''

    try:
        if os.path.isdir(data_path):
            new_data = xr.open_dataset(
                os.path.join(data_path, 'data.grib'), 
                engine='cfgrib', 
                decode_timedelta=False,
                backend_kwargs={
                    'indexpath':''
                }
                )
    except cfgrib.dataset.DatasetBuildError:
        print('oops')

    return new_data
    
def generate_kv_pair(data_dir, data_point):

    key = data_point[:7]

    data_path = os.path.join(data_dir, data_point)

    print(f'reading {data_path}')

    new_data = read_grib_file(data_path)

    return {key: new_data}

@track_runtime
def open_data_dir(
        data_dir: str
    ) -> dict[str, xr.Dataset]:
    '''
    hopefully this should lazy load a whole directory of datasets -- 

    ADD LARGE FILE HANDLING -- check size of directory first and if it's smalller than a certain size,
    use xr.load_datset() for speed reasons // if it's large, use xr.open_dataset() instead
    '''

    files = os.listdir(data_dir)

    data_dict = dict()
    for data_point in files:
        
        kv = generate_kv_pair(data_dir, data_point)

        data_dict.update(kv)

    print('done!')

    return data_dict

if __name__ == '__main__':

    # testing gdc behaviour
    data = DailyData(generate_kv_pair('./data/alaska_main/', '2014-11_input_data'))

    print(data[0]['u10']**1)
    print(data[0]['u10']**2)
    print((data[0]['u10']**2)**0.5)

def close_data_dir(data: dict[str, xr.Dataset]):
    '''
    closes all datasets to free up resources
    '''

    for ds in data.values():
        ds.close()