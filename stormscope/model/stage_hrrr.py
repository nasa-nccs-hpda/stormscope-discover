import os
import numpy as np
import pandas as pd

from earth2studio.data import HRRR, GFS_FX, datasource_to_file
from earth2studio.models.px import StormCast

os.makedirs("data", exist_ok=True)

time = np.array(["2024-09-26T12:00:00"], dtype="datetime64[ns]")
ts_str = pd.to_datetime(time[0]).strftime("%Y%m%d_%H%M%S")
out_file = f"data/hrrr_{ts_str}.nc"
# Use the variables your StormCast run will need.
# Start with the exact same variables requested by the model/workflow.
# Same conditioning source choice as inference
gfs_fx = GFS_FX(
    source="aws",
    cache=True,
    verbose=True,
)
gfs_file = f"data/stormcast_conditioning_{ts_str}.nc"

lead_time = np.array([0,1,2,3,4], dtype="timedelta64[h]")
gfs_variables = variables = np.array([
    "u10m", "v10m", "t2m", "tcwv", "sp", "msl",
    "u1000", "u850", "u500", "u250",
    "v1000", "v850", "v500", "v250",
    "z1000", "z850", "z500", "z250",
    "t1000", "t850", "t500", "t250",
    "q1000", "q850", "q500", "q250",
])

class GFSFXForFile:
    def __init__(self, src, lead_time):
        self.src = src
        self.lead_time = lead_time

    def __call__(self, time, variable):
        return self.src(time, self.lead_time, variable)

gfs_file_source = GFSFXForFile(gfs_fx, lead_time)
datasource_to_file(
    file_name=gfs_file,
    source=gfs_file_source,
    time=time,
    variable=gfs_variables,
    backend="netcdf",
)

print(f"Saved local GFS conditioning file: {out_file}")

package = StormCast.load_default_package()
model = StormCast.load_model(
    package,
    conditioning_data_source=gfs_file_source,
)
# StormCast required initial-condition variables
input_coords = model.input_coords()
variables = np.array(input_coords["variable"])

print("StormCast input variables:")
print(variables)

hrrr = HRRR(
    source="aws",
    cache=True,
    verbose=True,
)

datasource_to_file(
    file_name=out_file,
    source=hrrr,
    time=time,
    variable=variables,
    lead_time=np.array([0], dtype="timedelta64[h]"),
    backend="netcdf",
)

print(f"Saved local HRRR file: {out_file}")