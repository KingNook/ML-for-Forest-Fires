import os
import zipfile

import contextlib

@contextlib.contextmanager
def temp_cwd(new_dir):
    '''
    usage:
    ```
    with temp_cwd(dir_path):
        <code>
    ```
    '''
    old_cwd = os.getcwd()

    os.chdir(new_dir)
    try:
        yield

    finally:
        os.chdir(old_cwd)

def unzip_to_new_dir(data_path, file_name, remove):
    '''
    given a directory + zip file, such that `data_path/file_name` is a zip file, and file_name is `<file>.zip`

    extracts the zip file into a new directory `data_path/<file>/` then possibly removes the old zip file
    returns the path to directory where contents unzipped to
    test
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

def unpack_data_folder(data_path, remove=False):
    '''
    unpacks all .zip files in a folder

    data_path should point to a directory with .zip files

    if `remove=True`, then proceeds to delete all `.zip` files afterwards

    returns new list of files in directory
    '''

    if not os.path.isdir(data_path):

        raise ValueError('Not a valid directory')

    with temp_cwd(data_path):

        files = os.listdir('./')

        for file_name in files:

            if zipfile.is_zipfile(file_name):

                unzip_to_new_dir('.', file_name, remove=remove)

    
    return os.listdir(data_path)


if __name__ == '__main__':

    folder = 'alaska_prior'

    unpack_data_folder(f'./data/{folder}', remove = True)