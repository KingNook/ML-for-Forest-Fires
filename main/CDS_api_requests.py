'''
templates for sending requests to CDS (mostly for ERA5 data)
'''

import typing
import copy
import numpy as np

from math import ceil

from extents import FIRMS_to_CDS, validate_extent

type Request = dict[str, list|str]

## == PROCESSING REQUESTS == ##

def num_items(data):
    '''
    returns the number of items in `data`

    if `data` is an string, returns 1, otherwise returns `len(data)`

    NOTE this is not perfect, as it doesn't consider any other data types -- fine for this project though
    '''

    if type(data) == str:
        return 1

    else:
        try:
            return len(data)
        
        except AttributeError:
            
            # when the data does not have a __len__() method defined
            return 0

## i don't think this one is used anymore
def summarise_request_size(request: Request) -> np.ndarray[int]:
    '''
    note this returns a numpy array -- use `<array>.prod()` to get the estimate of request size

    estimates the request size -- it's the right order of magnitude, but overestimates slightly since it doesn't account for the number of days in each month; complexity added by counting those as well is really not worth it -- this is just so that i can break down large requests into smaller, higher-priority requests (so that they can process quicker; this process of sending multiple smaller requests was recommended by CDS docs at https://confluence.ecmwf.int/display/CKB/Climate+Data+Store+%28CDS%29+documentation)
    '''

    num_variables = num_items(request['variable'])
    num_years = num_items(request['year'])
    num_months = num_items(request['month'])
    num_days = num_items(request['day'])
    num_times = num_items(request['time'])

    return np.array(
        [num_variables, num_years, num_months, num_days, num_times]
    )

def split_request_by_feature(request: Request, feature: str) -> np.ndarray[Request]:
    '''
    given a Request (dict) and a feature, creates a list of Requests where num_<feature> = 1 in each
    '''

    num_feature = num_items(request[feature])

    if num_feature <= 1:
        # probably typo in feature if num_feature == 0 -- should really raise an AttributeError here instead
        return [request]
    
    split_request = []
    features = request[feature] # should be a list -- potensh a type check here would be good

    for i in range(num_feature):
        new_request = copy.deepcopy(request)
        new_request[feature] = [features[i]]

        split_request.append(new_request)

    return np.array(split_request, dtype=dict)

def split_list_by_feature(request_list: list[Request], feature: str) -> np.ndarray[Request]:
    '''
    applies `split_request_by_feature()` to each element of list and spits out 1D array
    '''

    split_request = np.array([
        split_request_by_feature(request, feature) for request in request_list
    ], dtype=dict)

    return split_request.flatten()

def split_request_by_feature_list(request: Request, feature_list: str) -> np.ndarray[Request]:
    '''
    splits up each entry in feature_list from request
    '''

    req = [request]

    for feature in feature_list:
        req = split_list_by_feature(req, feature)

    return req

def split_large_request(request: Request, max_size: int= -1) -> tuple[Request]:
    '''
    CDS limits all requests (API or webform) to 121000 credits (nb i assume API has the same limit as webform) -- if i want to request a larger amount of data, it is recommended to split this up into smaller requests -- my guess would be to limit to circa 30k credits max? // by default will break down into <1 variable, 1 month> per request

    recommendation is to split requests into: **all variables, monthly requests**

    inputs:
    - request: formatted as required for the CDS API
    - max_size: maximum size per final request sent to CDS API -- i would guess set this to something <= 30k
    '''
    
    # split request -- see 2025-07-16 daily note
    sr = split_request_by_feature_list(request, ['year', 'month'])

    return sr

## == CREATING REQUESTS == ##
# since most will be boilerplate, may as well create methods to make life easier

# years used for training in McNorton et al (2024)
DEFAULT_YEARS = [
    "2010", "2011", "2012",
    "2013", "2014"
]

ALL_MONTHS = [
    "01", "02", "03",
    "04", "05", "06",
    "07", "08", "09",
    "10", "11", "12"
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

def era5_land_request(
        variables: list[str], 
        extent: typing.Literal['world'] | list[str] = 'world', 
        years: list[str] = DEFAULT_YEARS, 
        months: list[str] = ALL_MONTHS,
        days: list[str] = ALL_DAYS,
        ) -> list[Request]:
    '''
    extent should be the CDS version
    '''

    if extent == 'world':

        request = {
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": ALL_TIMES,
            "data_format": "grib",
            "download_format": "zip"
        }
        
    else:

        validate_extent(extent, extent_format='CDS')

        request = {
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": ALL_TIMES,
            "data_format": "grib",
            "download_format": "zip",
            "area": extent
        }


    return split_large_request(request)