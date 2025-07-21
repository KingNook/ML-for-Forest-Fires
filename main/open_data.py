'''
given a directory, reads the data into some sort of datastructure

directory should look like: data/<dir_name>/<yyyy>-<mm>_<data_name>
'''

import os
import xarray as xr
import cfgrib

from unzippify import temp_cwd

def open_data_dir(data_dir):
    '''
    hopefully this should lazy load a whole directory of datasets -- 

    ADD LARGE FILE HANDLING -- check size of directory first and if it's smalller than a certain size,
    use xr.load_datset() for speed reasons // if it's large, use xr.open_dataset() instead
    '''

    files = os.listdir(data_dir)

    data_dict = dict()
    for data_point in files:
        
        key = data_point[:7]

        data_path = os.path.join(data_dir, data_point)

        print(f'reading {data_path}')

        if os.path.isdir(data_path):

            # nb idr if this needs to be a file object or if file path is fine
            new_data = xr.open_dataset(os.path.join(data_path, 'data.grib'), engine='cfgrib', decode_timedelta=False) 

            data_dict[key] = new_data

    print('done!')

    return data_dict

if __name__ == '__main__':

    test = open_data_dir('./data/alaska_prior')

    import geodataclass

    test2 = geodataclass.MonthlyData(test)

    for i in test2:
        print(i)