from datetime import datetime, timedelta
from earth2studio.data import GFS_FX

# one GFS cycle: 2023-12-05 12Z
init_time = [datetime(2023, 12, 5, 12)]

# exactly these two forecast files: f000 and f001
lead_times = [timedelta(hours=0), timedelta(hours=1)]

# choose variables you want from the GFS lexicon
variables = ["z500"]   # example; can be more than one

src = GFS_FX(source="aws", cache=True, verbose=True)

da = src(
    time=init_time,
    lead_time=lead_times,
    variable=variables,
)

# save to NetCDF
da.to_netcdf("gfs_20231205_12z_f000_f001.nc")
exit()


import os
from datetime import datetime
import numpy as np
from earth2studio.data import GFS_FX, GOES, MRMS, fetch_data, DataArrayFile
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)
import torch


da = GFS_FX().fetch(
    time=np.datetime64(datetime(2023, 12, 5, 12, 00, 0)),
    variable=np.array(["z500"]),
    lead_time=np.array([0,1]),
)
print(da)
exit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
goes_model_name = "6km_60min_natten_cos_zenith_input_eoe_v2"
package = StormScopeBase.load_default_package()
model = StormScopeGOES.load_model(
    package=package,
    conditioning_data_source=GFS_FX(),
    model_name=goes_model_name,
)

variables = model.input_coords()["variable"]
in_coords = model.input_coords()

start_date = [np.datetime64(datetime(2023, 12, 5, 12, 00, 0))]
GOES_INPUT_FILE = "goes_input.nc"
goes_local = DataArrayFile(GOES_INPUT_FILE)
x, x_coords = fetch_data(
    goes_local,
    time=start_date,
    variable=np.array(variables),
    lead_time=in_coords["lead_time"],
    device=device,
)
print("GOES data fetched successfully.")