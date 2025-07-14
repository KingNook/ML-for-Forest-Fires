'''
for reference, the FIRMS API docs are actually decent, unlike some other data stores (cough cough CDS) -- https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html
'''

import requests
import dotenv
import os
import pandas as pd

import typing

dotenv.load_dotenv(dotenv.find_dotenv())
MAP_KEY = os.environ['MAP_KEY']

# INFO -- remaining transactions, transactions used, transaction reset time
INFO_URL = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY

# DATA AVAILABILITY -- csv listing available datasets and dates from which these are available -- note can replace <all> with any individual dataset identifier
DA_URL = 'https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/' + MAP_KEY

# AREA DATA -- get data for ceratin area
AREA_BASE_URL = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY

# COUNTRIES -- list of countries
COUNTRIES_URL = 'https://firms.modaps.eosdis.nasa.gov/api/countries'
COUNTRY_BASE_URL = 'https://firms.modaps.eosdis.nasa.gov/api/country/csv/' + MAP_KEY


def get_data_from_url(url: str, data_format: typing.Literal['json', 'csv'] = 'json'):
    # probably a bad function since it 'does multiple things' and as a result, has no well-defined / consistent output type

    # technically should be some sort of check, eg using python `validators` module, but i'm the only person using this and the url is hard-coded so eh
    
    match data_format:
        case 'json':
            response = requests.get(url=url)
            data = response.json()
        
        case 'csv':
            data = pd.read_csv(url)

        case _:
            data = None

            raise ValueError(f'IDK what to do with data type: {data_format}')

    return data


def get_current_transactions(info_url: str = INFO_URL) -> int:

    data = get_data_from_url(info_url, data_format='json')

    return data['current_transactions']

def track_transactions_spent(func):
    '''wrapper to track how many transactions a function / an action spends'''

    def wrapped_fn(*args, **kwargs):
        t_start = get_current_transactions()
        output = func(*args, **kwargs)
        t_delta = get_current_transactions() - t_start

        print(f'[{func.__name__}] ({t_start + t_delta}) Transactions spent: {t_delta}')
        
        return output

    return wrapped_fn

@track_transactions_spent
def get_available_datasets(dataset: str = 'all', base_dataset_url: str = DA_URL) -> pd.DataFrame:
    '''
    grabs list of available datasets as well as the range of dates for which data is available

    parameters:
    - dataset: (default: 'all') replace with dataset id (from 'data_id' column) if only interested in available dates for single dataset
    - base_dataset_url: don't touch unless something major changes; tbh should probably hard-code this into constants at top
    '''

    '''see below comment for output from 14/07/25'''

    '''
                data_id    min_date    max_date
    0          MODIS_NRT  2025-04-01  2025-07-14
    1           MODIS_SP  2000-11-01  2025-03-31
    2   VIIRS_NOAA20_NRT  2025-03-01  2025-07-14
    3    VIIRS_NOAA20_SP  2018-04-01  2025-02-28
    4   VIIRS_NOAA21_NRT  2024-01-17  2025-07-14
    5     VIIRS_SNPP_NRT  2025-03-01  2025-07-14
    6      VIIRS_SNPP_SP  2012-01-20  2025-02-28
    7        LANDSAT_NRT  2022-06-20  2025-07-13
    8           GOES_NRT  2022-08-09  2025-07-14
    9           BA_MODIS  2000-11-01  2025-04-01
    10          BA_VIIRS  2012-03-01  2025-04-01
    '''

    ds_url = base_dataset_url + '/' + dataset

    return get_data_from_url(ds_url, data_format='csv')

@track_transactions_spent
def get_area_data(source: str, area_coords: str, day_range: int, date: str = '', area_base_url: str = AREA_BASE_URL):
    '''
    get fire data from a specified area
    
    parameters:
    - `source` -- should be `Literal[<all available datasets>]`: which dataset to get data from (eg 'MODIS_SP')
    - `day_range` -- `int`: number of days we want data for -- will count backwards from current date if <date> is not specified
    - `area_coords` -- `str`: (default: 'world') lat/long max/min determining extent of data gathered (or 'world' if we want all data)
    - `date` -- `str`: (optional) -- date of data (for historical data) // NEED TO CHECK IF THIS IS START OR END DATE OF DATA RANGE
    '''

    area_data_url = area_base_url + '/' + source + '/' + area_coords + '/' + str(day_range) + '/' + date

    data = get_data_from_url(area_data_url, data_format='csv') 

    return data


def get_countries():
    '''
    fixed url -- gets list of countries + country codes + lat/long max/min extents of each

    nb this does not use any transactions
    '''

    data = pd.read_csv(COUNTRIES_URL, sep=';')

    return data

@track_transactions_spent
def get_country_data(source, country, day_range, date=''):
    '''
    NB NEED TO DO TYPE HINTS + DOCS FOR THIS
    '''

    country_data_url = COUNTRY_BASE_URL + '/' + source + '/' + country + '/' + day_range + '/' + date

    data = get_data_from_url(country_data_url, data_format = 'csv')

    return data

print(get_country_data('MODIS_NRT', 'PER', '5'))