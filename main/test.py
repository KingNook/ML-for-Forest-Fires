import xarray as xr
from open_data import open_data_dir
from geodataclass import HourlyData, FlattenedData
from compute_wind_speed import append_wind_speed

# path = r'C:\Users\nookh\Github\summer-25\data\alaska_main\2010-03_input_data\data.grib'
# path = r'C:\Users\nookh\Github\summer-25\data\alaska_prior\2009-09_proxy_data\data.grib'


data = append_wind_speed(open_data_dir('./data/alaska_TEST_DATA'))
prior_data = open_data_dir('./data/alaska_prior')

ds = FlattenedData(data, prior_data)

print(ds[500, 7])