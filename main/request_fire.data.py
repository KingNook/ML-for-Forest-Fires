import pandas as pd

import FIRMS_api_requests as FReq

# path to DIRECTORY that data will be saved in // note that this is included in my gitignore (as i don't want to upload multiple GBs of data)
DATA_PATH = './data/' 

USA_data = FReq.get_country_data(
    source = 'MODIS_SP',
    country = 'USA',
    day_range = '100',
    date = '2010-01-01'
)

USA_data.to_csv(
    DATA_PATH + 'USA_2010_data-1.csv', 
)