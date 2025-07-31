'''
calculate the 30 day rolling averages, then functions to read off 90, 180 day averages from that
'''
import xarray as xr
from datetime import datetime

from tools import track_runtime
from geodataclass import HourlyData

@track_runtime
def rolling_sum_data(data, prior_data, n=-1):
    '''
    rolling 30 day total (hourly) sum
    '''
    if n < 0:
        n = len(data) + n + 1

    running_totals = []
    if prior_data != None:
        # fill in previous totals for items 1 to 30, then set prev_total to running_totals[-1] (which should be the same as running_totals[29])
        for i in range(n-30):
            old_data = prior_data[i]
            new_data = data[i]
            new_total = prev_total - old_data + new_data
            running_totals.append(new_total)      
    else:
        prev_total = sum(data[:30])

    for i in range(n-30):
        old_data = data[i]
        new_data = data[i+30]

        new_total = prev_total - old_data + new_data

        running_totals.append(new_total)

    return running_totals

def select_var_from_dict(data_dict: dict[str, xr.Dataset], var_name):
    
    for key, val in data_dict.items():
        data_dict[key] = val[var_name]

    return data_dict


def append_rolling_avg(data, prior_data, var_name):

    data = HourlyData(select_var_from_dict(data, var_name), select_var_from_dict(prior_data, var_name))

    rolling_sum = rolling_sum_data(data)

    for sum in rolling_sum:

        None.strftime(HourlyData.key_format)