import numpy as np
from datetime import datetime
import torch

from earth2studio.data import GOES, MRMS, GFS_FX, datasource_to_file, DataArrayFile
from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES, StormScopeMRMS
from earth2studio.models.auto import Package
init_time = [np.datetime64("2023-12-05T12:00:00")]
#device = "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

goes_model_name = "6km_10min_natten_pure_obs_zenith_6steps"
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

datasource_to_file("goes_input.nc", GOES(satellite="goes16", scan_mode="C"),
                   time=init_time, variable=goes_vars, lead_time=goes_leads, backend="netcdf")
datasource_to_file("mrms_input.nc", MRMS(),
                   time=init_time, variable=mrms_vars, lead_time=mrms_leads, backend="netcdf")
datasource_to_file("gfs_conditioning.nc", GFS_FX(),
                   time=init_time, variable=gfs_vars, lead_time=gfs_leads, backend="netcdf")

# In offline HPC inference:
# goes_local = DataArrayFile("/data/goes_input.nc")
# mrms_local = DataArrayFile("/data/mrms_input.nc")
# gfs_local  = DataArrayFile("/data/gfs_conditioning.nc")
# Then pass these as model data sources instead of GOES/MRMS/GFS_FX cloud sources.
