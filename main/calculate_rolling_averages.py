'''
calculate the 30 day rolling averages, then functions to read off 90, 180 day averages from that
'''

import open_data

from geodataclass import MonthlyData, Flattened_MonthlyData
from request_climate_data import track_runtime

DATA_PATH = './data/alaska_prior'

data = Flattened_MonthlyData(open_data.open_data_dir(DATA_PATH))

@track_runtime
def tot_data(data, n):
    running_totals = [] ## running 30 day totals

    for i in range(n):

        running_totals.append(
            sum(data[i:i+30])
        )

    return running_totals

print(tot_data(data, len(data)-30)[0])