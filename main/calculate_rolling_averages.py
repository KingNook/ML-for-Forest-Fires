'''
we need to calculate 30, 90, 180 day rolling averages of total_precipitation and 2m_temperature

the plan:
- calculate 30 day totals at each point we need
- calculate 90 day total at <date> by sum of 30 day totals at <date>, <date-30>, <date-60>
- calculate 180 day total at <date> by sum of 90 day totals at <date> and <date-90>

for this, we will need the data 180 days backwards from the start -- use python `datetime` to find this and to get the days needed between the points
OR we could just count back 6 months -- have put in a sample request to check how large this dataset would be -- may be the correct way to go about this (hold 6 months of data prior to the month we are looking at)
'''

import CDS_api_requests

import cdsapi

DEFAULT_YEARS = [
    "2010"
]

ALL_MONTHS = [
    "01", "02", "03",
    "04", "05", "06"
]

ALL_DAYS = [
    "01", "02", "03",
    "04", "05", "06",
    "07", "08", "09",
    "10", "11", "12",
    "13", "14", "15",
    "16", "17", "18",
    "19", "20", "21",
    "22", "23", "24",
    "25", "26", "27",
    "28", "29", "30",
    "31"
]

ALL_TIMES = [
    "00:00", "01:00", "02:00",
    "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00",
    "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00",
    "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00",
    "21:00", "22:00", "23:00"
]

def test_era5_request(variables: list[str], years: list[str] = DEFAULT_YEARS, split: bool = True):
    '''
    puts together a packet with default parameters -- nb the era5 datasets are:
    - reanalysis-era5-land
    - reanalysis-era5-single-levels
    '''

    request = {
        "variable": variables,
        "year": years,
        "month": ALL_MONTHS,
        "day": ALL_DAYS,
        "time": ALL_TIMES,
        "data_format": "grib",
        "download_format": "zip"
    }

    if split:
        return CDS_api_requests.split_large_request(request)
    
    else:
        return request

dataset = 'reanalysis-era5-land'

client = cdsapi.Client()
client.retrieve(dataset, test_era5_request(['total_precipitation', '2m_temperature'], split = False)).download()