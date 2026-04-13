"""stormscope_simple_inference.py - Using Earth2Studio's inference utilities"""

import numpy as np
import torch
from datetime import datetime
import xarray as xr

from earth2studio.data import DataArrayFile, GFS_FX, fetch_data
from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES
#from earth2studio.models.auto import Package
#from earth2studio.io import NetCDF4Backend

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 2

# Paths
GOES_INPUT_FILE = "goes_input.nc"
GFS_CONDITIONING_FILE = "gfs_conditioning.nc"
#PKG_PATH = "/stormscope/stormscope-goes-mrms"
GOES_MODEL_NAME = "6km_60min_natten_cos_zenith_input_eoe_v2"
OUTPUT_FILE = "./stormscope_forecast.nc"

print("Loading model and data...")

# Load package and model
#pkg = Package(PKG_PATH)
pkg = StormScopeBase.load_default_package()
gfs_local = DataArrayFile(GFS_CONDITIONING_FILE)
goes_local = DataArrayFile(GOES_INPUT_FILE)

goes_model = StormScopeGOES.load_model(
    pkg,
    model_name=GOES_MODEL_NAME,
    conditioning_data_source=gfs_local
).to(DEVICE)
goes_model.eval()

print(f"✓ Model loaded on {DEVICE}")

# Model-required coordinates
in_coords = goes_model.input_coords()
variables = np.array(in_coords['variable'])

# Get initial time from data
times = goes_local.da.coords["time"].values
start_time = times[0]
print(f"Initial time: {start_time}")

# Build interpolators
# For a local GOES file, we use its lat/lon grid as the input grid.
sample_da = goes_local.da.isel(time=0, variable=0)  # Sample data array to get lat/lon coords

input_lat = sample_da.coords["_lat"].values
input_lon = sample_da.coords["_lon"].values

goes_model.build_input_interpolator(input_lat, input_lon)
goes_model.build_conditioning_interpolator(GFS_FX.GFS_LAT, GFS_FX.GFS_LON)

# Fetch local GOES data through Earth2Studio helper so shapes/coords match model workflow
x, x_coords = fetch_data(
    goes_local,
    time=[start_time],
    variable=np.array(variables),
    lead_time=in_coords["lead_time"],
    device=DEVICE,
)
# Add batch dimension: expected shape [B, T, L, C, H, W]
if x.dim() == 5:
    x = x.unsqueeze(0)
    x_coords["batch"] = np.arange(1)
    x_coords.move_to_end("batch", last=False)

x = x.to(dtype=torch.float32)


print(f"Input tensor shape: {tuple(x.shape)}")
print(f"Variables: {list(variables)}")
print(f"Lead times: {x_coords['lead_time']}")

# Run inference
print(f"\nRunning {NUM_STEPS}-step forecast...")

# Iterative GOES-only inference loop
y, y_coords = x, x_coords
forecast_frames = []
forecast_coords = []

with torch.no_grad():
    for step_idx in range(args.n_steps):
        print(f"Running forecast step {step_idx + 1}/{args.n_steps}")

        # One model step
        y_pred, y_pred_coords = goes_model(y, y_coords)

        # Save raw prediction from this step
        forecast_frames.append(y_pred.detach().cpu())
        forecast_coords.append(y_pred_coords.copy())

        # Advance sliding input window for next step
        y, y_coords = goes_model.next_input(y_pred, y_pred_coords, y, y_coords)

# Concatenate along lead_time if possible
# Each y_pred should contain one future lead block produced by the model.
pred_xr_list = []
for i, (pred_torch, coords) in enumerate(zip(forecast_frames, forecast_coords)):
    pred_np = pred_torch.numpy()

    dims = list(coords.keys())
    pred_da = xr.DataArray(pred_np, dims=dims, coords=coords, name="stormscope_goes")
    pred_da = pred_da.assign_coords(forecast_step=i + 1).expand_dims("forecast_step")
    pred_xr_list.append(pred_da)

out_da = xr.concat(pred_xr_list, dim="forecast_step")

# Mask invalid grid cells if available
if hasattr(goes_model, "valid_mask") and goes_model.valid_mask is not None:
    valid_mask = goes_model.valid_mask.detach().cpu().numpy()
    # Broadcast mask over non-spatial dims
    out_da = out_da.where(valid_mask)

out_da.to_netcdf(OUTPUT_FILE)
print(f"Saved forecast to: {OUTPUT_FILE}")