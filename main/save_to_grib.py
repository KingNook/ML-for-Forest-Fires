import earthkit.data as ekd
import xarray as xr

xr.set_options(keep_attrs = True)

def xr_to_grib(ds, target):
    ekd.to_target('file', target, ds)

test = xr.open_dataset(
    './data/alaska_prior/2009-06_proxy_data/data.grib', engine='cfgrib'
)

