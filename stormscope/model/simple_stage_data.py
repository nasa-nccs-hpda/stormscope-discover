import numpy as np
from datetime import datetime, timedelta
import torch
import pandas as pd
import xarray as xr

from earth2studio.data import GOES, MRMS, GFS_FX, datasource_to_file, DataArrayFile
from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES, StormScopeMRMS
from earth2studio.models.auto import Package
init_time = [np.datetime64("2024-09-26T12:00:00")]
ts_str = pd.to_datetime(init_time[0]).strftime("%Y%m%d_%H%M%S")

nsteps = 6 # number of forecast steps to produce (e.g. 6 hours out with 1 hour lead time)
#device = "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

goes_model_name = "6km_60min_natten_cos_zenith_input_eoe_v2"
mrms_model_name = "6km_60min_natten_cos_zenith_input_mrms_eoe"

pkg = StormScopeBase.load_default_package()
# Load the package from local disk
# pkg_path = "/stormscope/stormscope-goes-mrms"
# pkg = Package(pkg_path)
print("✓ Package loaded successfully")
try:
    print(f"\nLoading GOES model: {goes_model_name}")
    goes_model = StormScopeGOES.load_model(
        pkg, 
        model_name=goes_model_name, 
        conditioning_data_source=GFS_FX(),
    )
    print("✓ GOES model loaded")
    
except Exception as e:
    print(f"✗ Failed to load GOES model")
    print(f"Error: {e}")

try:
    print(f"\nLoading MRMS model: {mrms_model_name}")
    mrms_model = StormScopeMRMS.load_model(
        pkg, 
        model_name=mrms_model_name, 
        conditioning_data_source=GOES(),
    )
    print("✓ MRMS model loaded")
except Exception as e:
    print(f"✗ Failed to load MRMS model")
    print(f"Error: {e}")
    
print("GOES model input variables:", goes_model.input_coords()["variable"])
print("MRMS model input variables:", mrms_model.input_coords()["variable"])

goes_vars = np.array(goes_model.input_coords()["variable"])
goes_leads = goes_model.input_coords()["lead_time"]

mrms_vars = np.array(["refc"])
mrms_leads = mrms_model.input_coords()["lead_time"]

gfs_vars = np.array(goes_model.conditioning_variables)   # e.g. z500 for this model
gfs_leads = goes_model.input_coords()["lead_time"]
#test_leads = np.array([0, 1, 2], dtype="timedelta64[h]")

datasource_to_file(f"data/goes_input_{ts_str}.nc", GOES(satellite="goes16", scan_mode="C"),
                   time=init_time, variable=goes_vars, lead_time=goes_leads, backend="netcdf")
datasource_to_file(f"data/mrms_input_{ts_str}.nc", MRMS(),
                   time=init_time, variable=mrms_vars, lead_time=mrms_leads, backend="netcdf")
# datasource_to_file("gfs_conditioning.nc", GFS_FX(),
#                    time=init_time, variable=gfs_vars, lead_time=gfs_leads, backend="netcdf")
src = GFS_FX(source="aws", cache=True, verbose=True)
lead_times = [timedelta(hours=int(e)) for e in range(nsteps)]
da = src(
    time=init_time,
    lead_time=lead_times,
    variable=gfs_vars,
)
valid_da_list = []
t0 = init_time[0]
for i, lead in enumerate(da.lead_time.values):
    one = da.sel(lead_time=lead, drop=True)
    vt = t0 + lead_times[i]
    one = one.assign_coords(time=np.array([vt], dtype="datetime64[ns]"))
    valid_da_list.append(one)

da_local = xr.concat(valid_da_list, dim="time")
da_local = da_local.sortby("time")
da_local.to_netcdf(f"data/gfs_conditioning_{ts_str}.nc")
# In offline HPC inference:
# goes_local = DataArrayFile("/data/goes_input.nc")
# mrms_local = DataArrayFile("/data/mrms_input.nc")
# gfs_local  = DataArrayFile("/data/gfs_conditioning.nc")
# Then pass these as model data sources instead of GOES/MRMS/GFS_FX cloud sources.
