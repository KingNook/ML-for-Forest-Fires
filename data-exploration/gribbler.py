'''for inspecting .grib files'''
import earthkit.data as ekd

data_directory = 'data/'
data_name = 'large set'

data_path = data_directory + data_name

fs = ekd.from_source('file', data_path)

print(fs.ls())