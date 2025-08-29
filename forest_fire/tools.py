'''
File for various miscellaneous scripts

From the old files, contains functions from 
- `tools.py` (done)
- `unzippify.py` (done)
'''

import contextlib
import os
import zipfile
import time

## ========
## tools.py
## ========
def track_runtime(func: function, name = '') -> function:
    '''
    Function wrapper to print the runtime of a function \\
     
    Parameters
    ----------
    func: function
        Function to be timed

    name: str, optional
        Name to be be printed. If no name is provided, this will default to `func.__name__`

    Returns
    -------
    wrapped_fn: function
        Timed function
    '''

    print_name = name if name != '' else func.__name__

    def tracked_fn(*args, **kwargs):
        start_time = time()
        out = func(*args, **kwargs)
        run_time = time() - start_time

        print(f'[{print_name}] run time: {run_time}s')

        return out
    
    return tracked_fn

## ============
## unzippify.py
## ============

@contextlib.contextmanager
def temp_cwd(new_cwd_path: str):
    '''
    Provides a context manager which temporarily changes the current working directory (cwd).

    Parameters
    ----------
    new_dir: str or path-like
        Path to temporary cwd; will not do anything if path is invalid
    
    Usage
    -----
    ```
    with temp_cwd(new_cwd_path):
        ...
    ```
    '''

    old_cwd = os.getcwd()
    os.chdir(new_cwd_path)
    try:
        yield

    finally:
        os.chdir(old_cwd)

def unzip_to_new_dir(data_path: str, file_name: str, remove: bool):
    '''
    Unzips a `.zip` file located at `<data_path>/<file_name>.zip` into a new directory, `<data_path>/<file_name>` \\
    
    Parameters
    ----------
    data_path: str or path-like
        Path to directory containing the `.zip` file; the data will be extracted to a new subdirectory

    file_name: str
        Path to original zip file; this will also be the name of the new directory

    remove: bool
        Whether to remove the `.zip` file after unzipping
    '''

    zip_file_path = os.path.join(data_path, file_name)

    new_dir_path = zip_file_path[:-4]
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)

    with zipfile.ZipFile(zip_file_path) as target:
        target.extractall(new_dir_path)

    if remove:
        os.remove(zip_file_path)

    return new_dir_path

def unpack_folder(data_path: str, remove: bool = False):
    '''
    Unzips all `.zip` files in a folder

    Parameters
    ----------
    data_path: str or path-like
        Path to directory containing `.zip` files

    remove: bool
        Whether to remove `.zip` files after unzipping

    Returns
    -------
    files: list
        List of files in new directory
    '''

    if not os.path.isdir(data_path):
        raise ValueError('Not a valid directory')

    with temp_cwd(data_path):
        files = os.listdir('./')

        for file_name in files:
            if zipfile.is_zipfile(file_name):
                unzip_to_new_dir('.', file_name, remove=remove)
    
    return os.listdir(data_path)