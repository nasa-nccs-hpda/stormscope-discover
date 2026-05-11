import xarray as xr
import numpy as np

in_file = "data/stormcast_conditioning_20240926_120000.nc"
out_file = "data/stormcast_conditioning_validtime_20240926_120000.nc"

ds = xr.open_dataset(in_file)
arr_name = list(ds.data_vars)[0]
da = ds[arr_name]

print("Before:")
print(da)

# Expect dims: time, lead_time, variable, ...
init_time = da.coords["time"].values[0]
lead_times = da.coords["lead_time"].values

valid_times = init_time + lead_times

# Remove init-time dimension, rename lead_time -> time
da2 = da.isel(time=0)
da2 = da2.assign_coords(time=("lead_time", valid_times))
da2 = da2.swap_dims({"lead_time": "time"})
da2 = da2.drop_vars("lead_time")

da2.name = "fields"

print("After:")
print(da2)

da2.astype("float32").to_netcdf(out_file)

print(f"Saved: {out_file}")