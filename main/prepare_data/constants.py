'''
Constants -- mostly default settings for various functions
'''

## default configs

# confidence: threshold (will classify > 70 as a fire)
# types: will classify all type '0' fires as positives
DEFAULT_FIRE_CONFIG = {
    'confidence': 70,
    'types': [0]
}

## requests -- default settings

TRAIN_YEARS = ['2010', '2011', '2012', '2013', '2014']
TEST_YEARS = ['2015', '2016', '2017', '2018', '2019']

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

## requests -- standard variable sets (replace these if needed / add your own)
PROXY_VARIABLES = [
    'total_precipitation',
    '2m_temperature'
]

WS_VARIABLES = [
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind',
]

STANDARD_VARIABLES = [
    'total_precipitation',
    '2m_dewpoint_temperature',
    '2m_temperature', 
    'skin_temperature',
    'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation',
    'surface_pressure'
]

## These are time invariant -- request these separately
VEG_COVER_VARIABLES = [
    'high_vegetation_cover', 
    'low_vegetation_cover',
    'type_of_high_vegetation', 
    'type_of_low_vegetation'
]

## THESE ARE BEING KINDA JANKY -- only one will download at a time for some reason
SM_VARIABLES = [
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4'
]

MAIN_VARS = PROXY_VARIABLES + WS_VARIABLES + STANDARD_VARIABLES
PRIOR_VARS = PROXY_VARIABLES