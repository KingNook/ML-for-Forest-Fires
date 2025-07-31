from time import time

def track_runtime(func):

    def tracked_fn(*args, **kwargs):
        start_time = time()
        out = func(*args, **kwargs)
        run_time = time() - start_time

        print(f'[{func.__name__}] run time: {run_time}s')

        return out
    
    return tracked_fn