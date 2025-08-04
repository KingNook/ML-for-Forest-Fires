'''
download monthly data for the input variables
'''

import CDS_api_requests
import unzippify

import request_climate_data as rcd
from CDS_api_requests import DEFAULT_YEARS, ALL_MONTHS, ALL_DAYS

from extents import ALASKA_RANGE_EXTENT, LA_FORESTS_EXTENT, Extent

'''input_variables = rcd.standard_variables + rcd.wind_speed_variables

requests = CDS_api_requests.era5_land_request(
    variables = input_variables,
    extent = ALASKA_RANGE_EXTENT.CDS, 
    years = ['2014'],  

    
            months = [
        '05', '06', '07', '08', '09', '10', '11', '12'
    ]  
)

rcd.multi_download(requests, 'input_data', 'alaska_main', max_threads=2)

unzippify.unpack_data_folder('./data/alaska_main', remove=True)'''


def request_total_data(
        extent: Extent,
        data_path: str = './data',
        
    ):

    name = extent.name

    input_variables = rcd.standard_variables + rcd.wind_speed_variables
    proxy_variables = rcd.proxy_base_variables

    proxy_requests = CDS_api_requests.era5_land_request(
        variables = proxy_variables,
        extent = extent.CDS,
        years = ['2009'],
        months = ['07', '08', '09', '10', '11', '12']
    )

    rcd.multi_download(proxy_requests, 'proxy_data', f'{name}_prior', max_threads=2)
    unzippify.unpack_data_folder(f'./data/{name}_proxy', remove=True)

    input_requests = CDS_api_requests.era5_land_request(
        variables = input_variables,
        extent = extent.CDS, 
        years = ['2010', '2011', '2012', '2013', '2014']
    )
    
    rcd.multi_download(input_requests, 'input_data', f'{name}_main', max_threads=2)
    unzippify.unpack_data_folder(f'./data/{name}_main', remove=True)
