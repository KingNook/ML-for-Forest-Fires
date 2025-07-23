'''
download monthly data for the input variables
'''

import CDS_api_requests
import unzippify

import request_climate_data as rcd

from extents import ALASKA_RANGE_EXTENT

input_variables = rcd.standard_variables + rcd.wind_speed_variables

requests = CDS_api_requests.era5_land_request(
    variables = input_variables,
    extent = ALASKA_RANGE_EXTENT.CDS, 
    years = ['2014'],
    months = [
        '05', '06', '07', '08', '09', '10', '11', '12'
    ]  
)

rcd.multi_download(requests, 'input_data', 'alaska_main', max_threads=2)

unzippify.unpack_data_folder('./data/alaska_main', remove=True)