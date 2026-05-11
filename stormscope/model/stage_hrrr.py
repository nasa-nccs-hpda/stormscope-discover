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
conditioning_data_source = GFS_FX(
    source="aws",
    cache=True,
    verbose=True,
)

package = StormCast.load_default_package()
model = StormCast.load_model(
    package,
    conditioning_data_source=conditioning_data_source,
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