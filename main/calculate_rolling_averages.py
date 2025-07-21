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
    '''
    running 30d totals
    '''

    prev_total = sum(data[:30])
    running_totals = []

    for i in range(n-30):
        old_data = data[i]
        new_data = data[i+30]

        new_total = prev_total - old_data + new_data

        running_totals.append(new_total)       

    return running_totals

if __name__ == '__main__':
    print(tot_data(data, len(data))[0])