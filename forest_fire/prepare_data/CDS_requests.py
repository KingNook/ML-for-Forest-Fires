'''
This contains all of the code that relates to requests to the CDS (Climate Data Store) managed by Copernicus/ECMWF 

RECOMMENDED: to use `request_total_data()` unless specifics need to be changed (in which case use `format_request()` with `multi_download()`)

Note that the CDS API personal access token must be set up in advance: https://cds.climate.copernicus.eu/how-to-api

From the old files, this covers functionality from
- CDS_api_requests.py (done)
- download_climate_data.py (done)
- request_climate_data.py (done)
- concat_gribs_from_subdirs -- prepare_data.py (WIP)
'''
from ..tools import track_runtime, unpack_folder

from typing import Literal
from extents import Extent
from constants import *
from copy import deepcopy
import threading
from math import ceil
import os

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

@track_runtime(name='Data Request')
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

def download_data(
        requests: list[Request],
        data_dir: str
        ):
    '''
    Downloads data to `data_dir`

    Parameters
    ----------
    requests: list
        List of requests

    data_dir: str or path-like
        Path to directory for download
    '''
    
    for request in requests: 

        year = request['year'][0]
        month = int(request['month'][0])
        download_path = f'{data_dir}/{year}-{month:02d}.zip'

        send_request(request, target = download_path)

def multi_download(
        requests: list[Request],
        data_dir: str,
        max_threads: int = 2
    ):    
    '''
    Downloads all requests in a list of requests - this uses multi-threading to queue 2 requests at a time (CDS will only process one at a time, but if multiple are queued then the waiting time between requests is reduced; I don't think there is any benefit in having more than 2 threads)

    Data will be downloaded as `.zip` files in the directory `data_dir`, named by their relevant month 

    Parameters
    ----------
    requests: list
        List of 'Request's to be downloaded. I would recommend generating this using `format_requests()`

    data_dir: str or path-like
        Path to directory that data will be downloaded to

    max_threads: int, optional
        Maximum number of threads
    '''

    if max_threads == -1:
        max_threads = len(requests)

    jobs = []
    job_size = ceil(len(requests) / max_threads)
    
    for i in range(max_threads-1):
        job = requests[i * job_size : (i+1) * job_size]
        jobs.append(job)

    jobs.append(requests[(max_threads-1)*job_size:])

    threads = []

    for job in jobs:
        thread = threading.Thread(target=download_data, args=(job, data_dir))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def concat_gribs_from_subdirs(root_dir, output_name='combined.grib'):
    '''
    Combines all 'data.grib' files from subdirectories into one big ol' GRIB file. The combined grib file will be written to the main directory
    
    Parameters
    ----------
    root_dir: str
        Path to the folder containing subdirectories

    output_name: str, optional
        Path to combined GRIB file. Default is 'combined.grib'
    '''
    output_path = os.path.join(root_dir, output_name)
    with open(output_path, 'wb') as outfile:
        for subdir in sorted(os.listdir(root_dir)):
            sub_path = os.path.join(root_dir, subdir)
            grib_path = os.path.join(sub_path, 'data.grib')
            if os.path.isdir(sub_path) and os.path.isfile(grib_path):
                print(f"Adding: {grib_path}")
                with open(grib_path, 'rb') as infile:
                    outfile.write(infile.read())
    print(f"\n✅ Combined GRIB saved to: {output_path}")

def request_total_data(
        extent: Extent,
        data_path: str = './data',
        prior: bool = True,
        main: bool = True,
        test: bool = False
    ):

    name = extent.name

    input_variables = MAIN_VARS
    prior_variables = PRIOR_VARS

    if prior:
        proxy_requests = format_requests(
            variables = prior_variables,
            extent = extent.CDS,
            years = ['2009'],
            months = ['07', '08', '09', '10', '11', '12']
        )

        prior_dir_name = os.path.join(data_path, f'{name}/prior')

        multi_download(proxy_requests, prior_dir_name, max_threads=2)
        unpack_folder(prior_dir_name, remove=True)
        concat_gribs_from_subdirs(prior_dir_name)

    if main:
        input_requests = format_requests(
            variables = input_variables,
            extent = extent.CDS, 
            years = TRAIN_YEARS
        )
        
        dir_name = os.path.join(data_path, f'{name}/main')

        multi_download(input_requests, dir_name, max_threads=-1)
        unpack_folder(dir_name, remove=True)
        concat_gribs_from_subdirs(dir_name)

    if test:
        input_requests = format_requests(
            variables = input_variables,
            extent = extent.CDS, 
            years = TEST_YEARS
        )
        
        dir_name = os.path.join(data_path, f'{name}/test')

        multi_download(input_requests, dir_name, max_threads=-1)
        unpack_folder(dir_name, remove=True)
        concat_gribs_from_subdirs(dir_name)