import numpy as np
import pandas as pd
from earth2studio.data import HRRR, datasource_to_file



time = np.array(["2024-09-26T12:00:00"], dtype="datetime64[ns]")
ts_str = pd.to_datetime(time[0]).strftime("%Y%m%d_%H%M%S")
out_file = f"data/hrrr_{ts_str}.nc"
# Use the variables your StormCast run will need.
# Start with the exact same variables requested by the model/workflow.
variables = np.array([
    "t2m",
    "u10m",
    "v10m",
    "msl",
    "tcwv",
    "refc",
])

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