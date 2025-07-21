'''
download monthly data for the input variables
'''

import CDS_api_requests
import datetime

import request_climate_data as rcd

from extents import ALASKA_RANGE_EXTENT

input_variables = rcd.standard_variables + rcd.wind_speed_variables

requests = CDS_api_requests.era5_land_request(
    variables = input_variables,
    extent = ALASKA_RANGE_EXTENT.CDS,   
)

rcd.download_data(requests, 'input_data', 'alaska_main')

