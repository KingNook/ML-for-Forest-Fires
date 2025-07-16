import cdsapi

import CDS_api_requests

# need whole dataset at once for calculations -- will be calculating running average for these (nb will also need 180 days prior to start of 2010 to calculate initial averages)
# write request specifically for these separately
proxy_base_variables = ['total_precipitation', '2m_temperature']

# can get relevant sets where needed
wind_speed_variables = [
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind',
]

standard_variables = [
    'total_precipitation', ## not sure about this one
    '2m_dewpoint_temperature', # 
    '2m_temperature', 
    'skin_temperature',
    'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation',
    'high_vegetation_cover', 
    'low_vegetation_cover',
    'type_of_high_vegetation', 
    'type_of_low_vegetation',
    'surface_pressure'
]

soil_moisture_variables = [
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4'
]

requests = CDS_api_requests.default_era5_request(proxy_base_variables, years = ['2010'])
dataset = 'reanalysis-era5-land'

client = cdsapi.Client()
client.retrieve(dataset, requests[0]).download()