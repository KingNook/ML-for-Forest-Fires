import numpy as np
import cdsapi
import datetime

import time

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

        print(f'[{func.__name__()}] run time: {run_time}s')

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

    if auto_unzip, then will unzip the file imeediately once downloaded // need to implement this
    '''

    if target == '':
        client.retrieve(dataset, request).download()
    else:
        client.retrieve(dataset, request, target)
client.retrieve(dataset, requests[0]).download()
    return None