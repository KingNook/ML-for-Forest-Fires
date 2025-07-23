'''
calculate the 30 day rolling averages, then functions to read off 90, 180 day averages from that
'''

import open_data

from geodataclass import HourlyData
from request_climate_data import track_runtime

DATA_PATH = './data/alaska_prior'

data = HourlyData(open_data.open_data_dir(DATA_PATH))

@track_runtime
def tot_data(data, n=-1):
    '''
    running 30d totals
    '''

    if n < 0:
        n = len(data) + n + 1

    prev_total = sum(data[:30])
    running_totals = []

    for i in range(n-30):
        old_data = data[i]
        new_data = data[i+30]

        new_total = prev_total - old_data + new_data

        running_totals.append(new_total)       

    return running_totals