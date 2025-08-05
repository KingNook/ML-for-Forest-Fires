from time import time
from datetime import datetime, timedelta

def track_runtime(func):

    def tracked_fn(*args, **kwargs):
        start_time = time()
        out = func(*args, **kwargs)
        run_time = time() - start_time

        print(f'[{func.__name__}] run time: {run_time}s')

        return out
    
    return tracked_fn

# pd.to_datetime

def hourlydaterange(start_date, end_date):
    hours = (end_date - start_date).days * 24
    for n in range(0, hours-1):
        n += 1
        yield start_date + timedelta(hours = n)
