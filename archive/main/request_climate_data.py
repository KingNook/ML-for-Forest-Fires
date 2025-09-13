import numpy as np
import cdsapi
import datetime

import time

import threading

from math import ceil

from tools import track_runtime

import CDS_api_requests, unzippify
from extents import ALASKA_RANGE_EXTENT

# need whole dataset at once for calculations -- will be calculating running average for these (nb will also need 180 days prior to start of 2010 to calculate initial averages)
# write request specifically for these separately
proxy_base_variables = ['total_precipitation', '2m_temperature']

# can get relevant sets where needed
wind_speed_variables = [
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind',
]

standard_variables = [
    'total_precipitation',
    '2m_dewpoint_temperature',
    '2m_temperature', 
    'skin_temperature',
    'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation',
    'surface_pressure'
]

vegetation_cover = [
    'high_vegetation_cover', 
    'low_vegetation_cover',
    'type_of_high_vegetation', 
    'type_of_low_vegetation'
]

soil_moisture_variables = [
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4'
]

all_vars = proxy_base_variables + wind_speed_variables + standard_variables

client = cdsapi.Client()

def track_runtime(func):

    def tracked_fn(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        run_time = time.time() - start_time

        print(f'[{func.__name__}] run time: {run_time}s')

        return out
    
    return tracked_fn

def parse_date(date: datetime.date) -> str:
    '''
    takes a `date` object and returns the string `yyyy-mm-dd`
    '''

    return f'{date.year}-{date.month}-{date.day}'

@track_runtime
def send_request(
    request: CDS_api_requests.Request, 
    dataset: str ='reanalysis-era5-land', 
    target: str = '', # lands in 
    auto_unzip: bool = False,
    auto_remove: bool = True,
    client=client
    ) -> None:
    '''
    idk what would be sensible to return tbh

    if auto_unzip, then will unzip the file imediately once downloaded // need to implement this
    '''

    if target == '':
        client.retrieve(dataset, request).download()
    else:
        client.retrieve(dataset, request, target)

def download_data(
        requests: list[CDS_api_requests.Request],
        data_name: str,
        extent_name: str = 'world'
        ):
    '''
    downloads data to `./data/<extent_name>/<date>_<data_name>.zip
    '''
    
    for request in requests: 

        year = request['year'][0]
        month = int(request['month'][0])
        download_path = f'./data/{extent_name}/{year}-{month:02d}_{data_name}.zip'

        send_request(request, target = download_path)

def multi_download(
        requests: list[CDS_api_requests.Request],
        data_name: str,
        extent_name: str = 'world',
        max_threads: int = 2
    ):    
    '''
    Downloads all requests in a list of requests - this uses multi-threading to queue 2 requests at a time (CDS will only process one at a time, but if multiple are queued then the waiting time between requests is reduced; I don't think there is any benefit in having more than 2 threads)

    Parameters
    ----------
    requests: list
        List of 'Request's to be downloaded. I would recommend generating this from the 
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
        thread = threading.Thread(target=download_data, args=(job, data_name, extent_name))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

# example code -- downloads and unzips prior data for the alaska range
if __name__ == '__main__':



    extent = ALASKA_RANGE_EXTENT.CDS

    requests = CDS_api_requests.era5_land_request(
        variables = proxy_base_variables, 
        extent = extent,
        years = ['2009'],
        months = ['07'] # [str(i) for i in range(6, 13)] # 6 -> 12 (hopefully)
    )

    