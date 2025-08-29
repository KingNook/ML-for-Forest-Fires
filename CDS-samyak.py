from typing import Literal
from extents import Extent
from constants import *
from copy import deepcopy
import threading
from math import ceil

import numpy as np
import cdsapi

default_client = cdsapi.Client()

type Request = dict[str, list|str]

def count_items(data):
    '''
    Returns the number of items in data if data is a list/tuple/array etc, or 1 otherwise

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

def split_request_by_feature(request: Request, feature: str) -> np.ndarray[Request]:
    '''
    Given a 'Request' and a feature, returns a list of 'Request's in which each element only corresponds to one 'feature'
    '''

    num_feature = count_items(request[feature])

    if num_feature <= 0:
        print(f'Feature: {feature} not found')
        return [request]

    elif num_feature == 1:
        return [request]
    
    split_request = []
    features = request[feature] # should be a list -- potensh a type check here would be good

    for i in range(num_feature):
        new_request = deepcopy(request)
        new_request[feature] = [features[i]]

        split_request.append(new_request)

    return np.array(split_request, dtype=dict)

def split_list_by_feature(request_list: list[Request], feature: str) -> np.ndarray[Request]:
    '''
    Applies `split_request_by_feature(request, feature)` to each `request` in a list
    '''

    split_request = np.array([
        split_request_by_feature(request, feature) for request in request_list
    ], dtype=dict)

    return split_request.flatten()

def split_large_request(request: Request) -> tuple[Request]:
    '''
    Splits a 'Request' into a list of 'Requests', each of which corresponds to exactly one month of data

    Parameters
    ----------
    request: Request
        Request to be split

    Returns
    -------
    monthly_requests: list
        List of one-month requests
    '''

    req = [request]

    for feature in ['year', 'month']:
        req = split_list_by_feature(req, feature)

    return tuple(req)

def format_requests(
        variables: list[str], 
        extent: Extent | Literal['world'], 
        years: list[str] = TRAIN_YEARS, 
        months: list[str] = ALL_MONTHS,
        days: list[str] = ALL_DAYS,
        split: bool = True
        ) -> list[Request]:
    '''
    Returns a list of 'request' dictonaries to be processed by the CDS API. If unsure of any of these, go to the webform for the relevant dataset and copy from the example API request at the bottom

    Parameters
    ----------
    variables: list
        List of variable names

    extent: Extent
        Extent for which data is wanted -- see `extents.py` for examples \\
        For a new extent, give coordinates in 'FIRMS' format

    years: list
        List of years for which data is wanted. By default, this is from 2010-2014 inclusive

    months: list
        List of months for which data is wanted. By default this is Jan-Dec (ie all months) \\
        If not all months have data (eg some are in the future), will only retrieve data for months for which data is available

    days: list
        List of days for which data is wanted. By default this is 1-31 (ie all days) \\
        If not all days have data (eg month with <31 days), will retrieve data for all days

    hours: list
        List of hours for which data is wanted. By default, this is 0:00-23:00 (ie all hours) \\
        Note that data for 0:00 may be returned as 24:00 on the previous day

    splt: bool, optional
        Whether to split into monthly requests, or one large request. Default is True (Recommended by CDS documentation)
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
        request = {
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": ALL_TIMES,
            "data_format": "grib",
            "download_format": "zip",
            "area": extent.CDS
        }


    if split:
        requests = split_large_request(request)
    else:
        requests = [request]
    
    return requests

def send_request(
    request: Request, 
    dataset: str ='reanalysis-era5-land', 
    target: str = '',
    client: cdsapi.Client = default_client
    ):
    '''
    Sends a download request to the CDS store

    Parameters
    ----------
    request: Request (dict)
        Parameters for the request. Recommended to either use template copied from bottom of webform, or output from `format_requests()`

    dataset: str, optional
        Dataset which is being requested. By default thsi is ERA5-Land

    target: str or path-like, optional
        Path to downloaded file. If the path contains non-existent directories, the download will fail

    client: cdsapi.Client, optional
        Client object from `cdsapi`. If you have custom settings (eg downloading via a mirror), pass your own, otherwise this one will work fine.
    '''

    if target == '':
        client.retrieve(dataset, request).download()
    else:
        client.retrieve(dataset, request, target)