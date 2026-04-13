import os
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
# from dotenv import load_dotenv

# load_dotenv()

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.data import GFS_FX, GOES, MRMS, fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)

print("All imports successful. StormScope is ready to run!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

goes_model_name = "6km_60min_natten_cos_zenith_input_eoe_v2"
mrms_model_name = "6km_60min_natten_cos_zenith_input_mrms_eoe"

package = StormScopeBase.load_default_package()

# Load GOES model with GFS_FX conditioning (should be set to None for 10min models)
model = StormScopeGOES.load_model(
    package=package,
    conditioning_data_source=GFS_FX(),
    model_name=goes_model_name,
)
model = model.to(device)
model.eval()

# Load MRMS model with GOES conditioning (should be set to None for 10min models)
model_mrms = StormScopeMRMS.load_model(
    package=package,
    conditioning_data_source=GOES(),
    model_name=mrms_model_name,
)
model_mrms = model_mrms.to(device)
model_mrms.eval()

start_date = [np.datetime64(datetime(2023, 12, 5, 12, 00, 0))]
goes_satellite = "goes16"
scan_mode = "C"

variables = model.input_coords()["variable"]
lat_out = model.latitudes.detach().cpu().numpy()
lon_out = model.longitudes.detach().cpu().numpy()

goes = GOES(satellite=goes_satellite, scan_mode=scan_mode)
goes_lat, goes_lon = GOES.grid(satellite=goes_satellite, scan_mode=scan_mode)

# Build interpolators for transforming data to model grid
model.build_input_interpolator(goes_lat, goes_lon)
model.build_conditioning_interpolator(GFS_FX.GFS_LAT, GFS_FX.GFS_LON)

in_coords = model.input_coords()

# Fetch GOES data
x, x_coords = fetch_data(
    goes,
    time=start_date,
    variable=np.array(variables),
    lead_time=in_coords["lead_time"],
    device=device,
)