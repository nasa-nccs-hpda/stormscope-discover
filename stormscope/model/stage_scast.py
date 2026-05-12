import os
import numpy as np
import pandas as pd

from earth2studio.data import HRRR, GFS_FX
from earth2studio.models.px import StormCast

os.makedirs("data", exist_ok=True)

time = np.array(["2024-09-26T12:00:00"], dtype="datetime64[ns]")
ts_str = pd.to_datetime(time[0]).strftime("%Y%m%d_%H%M%S")

hrrr_file = f"data/hrrr_{ts_str}.nc"
gfs_file = f"data/stormcast_conditioning_validtime_{ts_str}.nc"

# --------------------------------------------------
# 1. Get StormCast-required HRRR input variables
# --------------------------------------------------

gfs_for_model = GFS_FX(source="aws", cache=True, verbose=True)

package = StormCast.load_default_package()
model = StormCast.load_model(
    package,
    conditioning_data_source=gfs_for_model,
)

hrrr_variables = np.array(model.input_coords()["variable"])

print("StormCast HRRR input variables:")
print(hrrr_variables)

# --------------------------------------------------
# 2. Stage HRRR initial condition
# --------------------------------------------------

hrrr = HRRR(source="aws", cache=True, verbose=True)

da_hrrr = hrrr(time, hrrr_variables)
da_hrrr.name = "fields"
da_hrrr.astype("float32").to_netcdf(hrrr_file)

print(f"Saved local HRRR file: {hrrr_file}")

# --------------------------------------------------
# 3. Stage GFS_FX conditioning
# --------------------------------------------------
# These are the GFS conditioning variables StormCast expects.

gfs_variables = np.array([
    "u10m", "v10m", "t2m", "tcwv", "sp", "msl",
    "u1000", "u850", "u500", "u250",
    "v1000", "v850", "v500", "v250",
    "z1000", "z850", "z500", "z250",
    "t1000", "t850", "t500", "t250",
    "q1000", "q850", "q500", "q250",
])

# Need enough valid times for StormCast rollout.
# For nsteps=4, use at least 0..4 hours.
lead_time = np.array([0, 1, 2, 3, 4, 5], dtype="timedelta64[h]")

gfs = GFS_FX(source="aws", cache=True, verbose=True)

da_gfs = gfs(time, lead_time, gfs_variables)

print("Raw GFS_FX:")
print(da_gfs)

# Convert init_time + lead_time -> valid time
base_time = da_gfs.coords["time"].values[0]
valid_times = base_time + da_gfs.coords["lead_time"].values

da_gfs = da_gfs.isel(time=0)
da_gfs = da_gfs.assign_coords(time=("lead_time", valid_times))
da_gfs = da_gfs.swap_dims({"lead_time": "time"})
da_gfs = da_gfs.drop_vars("lead_time")

da_gfs.name = "fields"
da_gfs.astype("float32").to_netcdf(gfs_file)

print("Converted GFS conditioning:")
print(da_gfs)
print(f"Saved local GFS conditioning file: {gfs_file}")