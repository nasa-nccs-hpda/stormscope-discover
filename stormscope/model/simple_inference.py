"""stormscope_simple_inference.py - Using Earth2Studio's inference utilities"""

import numpy as np
import torch
from datetime import datetime
import xarray as xr

from earth2studio.data import DataArrayFile, GFS_FX, fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeBase, 
    StormScopeGOES,
    StormScopeMRMS,
)

def summarize(name, t):
    t_cpu = t.detach().cpu()

    finite_mask = torch.isfinite(t_cpu)
    n_finite = finite_mask.sum().item()
    total = t_cpu.numel()

    if n_finite > 0:
        finite_vals = t_cpu[finite_mask]
        min_val = finite_vals.min().item()
        max_val = finite_vals.max().item()
    else:
        min_val = "NA"
        max_val = "NA"

    print(
        f"{name}: shape={tuple(t_cpu.shape)}, "
        f"finite={n_finite}/{total}, "
        f"nan={(~finite_mask).sum().item()}, "
        f"min={min_val}, max={max_val}"
    )

def show_coords(name, coords):
    print(f"{name} dims={list(coords.keys())}")
    for k, v in coords.items():
        arr = np.asarray(v)
        preview = arr[: min(3, len(arr))] if arr.ndim == 1 else arr.shape
        print(f"  {k}: shape={arr.shape}, sample={preview}")
#from earth2studio.models.auto import Package
#from earth2studio.io import NetCDF4Backend

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 2

# Paths
GOES_INPUT_FILE = "goes_input.nc"
GOES_MODEL_NAME = "6km_60min_natten_cos_zenith_input_eoe_v2"

MRMS_INPUT_FILE = "mrms_input.nc"
MRMS_MODEL_NAME = "6km_60min_natten_cos_zenith_input_mrms_eoe"

GFS_CONDITIONING_FILE = "gfs_conditioning.nc" #"gfs_20231205_12z_f000_f001.nc" #
OUTPUT_FILE = "./stormscope_forecast.nc"

print("Loading model and data...")

# Load package and model
#pkg = Package(PKG_PATH)
pkg = StormScopeBase.load_default_package()
gfs_local = DataArrayFile(GFS_CONDITIONING_FILE)
goes_local = DataArrayFile(GOES_INPUT_FILE)
mrms_local = DataArrayFile(MRMS_INPUT_FILE)

goes_model = StormScopeGOES.load_model(
    pkg,
    model_name=GOES_MODEL_NAME,
    conditioning_data_source=gfs_local,
).to(DEVICE)
goes_model.eval()
valid_mask = goes_model.valid_mask.detach().cpu().numpy()
print(valid_mask.shape, valid_mask.sum(), valid_mask.sum() / valid_mask.size)
exit()
mrms_model = StormScopeMRMS.load_model(
    pkg,
    model_name=MRMS_MODEL_NAME,
    conditioning_data_source=goes_local,
).to(DEVICE)
mrms_model.eval()
print(f"✓ Model loaded on {DEVICE}")

# Model-required coordinates
in_coords = goes_model.input_coords()
mrms_in_coords = mrms_model.input_coords()
variables = np.array(in_coords['variable'])

# Get initial time from data
times = goes_local.da.coords["time"].values
start_time = times[0]
print(f"Initial time: {start_time}")

# Build interpolators
# For a local GOES file, we use its lat/lon grid as the input grid.
sample_da = goes_local.da.isel(time=0, variable=0)  # Sample data array to get lat/lon coords

goes_lat = sample_da.coords["_lat"].values
goes_lon = sample_da.coords["_lon"].values

goes_model.build_input_interpolator(goes_lat, goes_lon)

# For conditioning data, we also build an interpolator to the same lat/lon grid.
sample = gfs_local.da  # underlying xarray
cond_lat = sample.coords["lat"].values
cond_lon = sample.coords["lon"].values
goes_model.build_conditioning_interpolator(cond_lat, cond_lon)

# Fetch local GOES data through Earth2Studio helper so shapes/coords match model workflow
x, x_coords = fetch_data(
    goes_local,
    time=[start_time],
    variable=np.array(variables),
    lead_time=in_coords["lead_time"],
    device=DEVICE,
)

# For a local MRMS file
sample_da = mrms_local.da.isel(time=0, variable=0)  # Sample data array to get lat/lon coords
mrms_lat = sample_da.coords["lat"].values
mrms_lon = sample_da.coords["lon"].values
mrms_model.build_input_interpolator(mrms_lat, mrms_lon)
mrms_model.build_conditioning_interpolator(goes_lat, goes_lon)

x_mrms, x_coords_mrms = fetch_data(
    mrms_local,
    time=[start_time],
    variable=np.array(["refc"]),
    lead_time=mrms_in_coords["lead_time"],
    device=DEVICE,
)
# Add batch dimension: expected shape [B, T, L, C, H, W]
if x.dim() == 5:
    x = x.unsqueeze(0)
    x_coords["batch"] = np.arange(1)
    x_coords.move_to_end("batch", last=False)

if x_mrms.dim() == 5:
    x_mrms = x_mrms.unsqueeze(0)
    x_coords_mrms["batch"] = np.arange(1)
    x_coords_mrms.move_to_end("batch", last=False)

x = x.to(dtype=torch.float32)
x_mrms = x_mrms.to(dtype=torch.float32)

print(f"Input tensor shape: {tuple(x.shape)}")
print(f"Variables: {list(variables)}")
print(f"Lead times: {x_coords['lead_time']}")

# Run inference
print(f"\nRunning {NUM_STEPS}-step forecast...")

# Iterative GOES & MRMS inference loop
y, y_coords = x, x_coords
y_mrms, y_coords_mrms = x_mrms, x_coords_mrms
forecast_frames = []
forecast_frames_mrms = []
forecast_coords = []
forecast_coords_mrms = []

with torch.no_grad():
    for step_idx in range(NUM_STEPS):
        print(f"Running forecast step {step_idx + 1}/{NUM_STEPS}")

        # One model step
        # summarize("y input to GOES", y)
        # show_coords("y_coords", y_coords)
        y_pred, y_pred_coords = goes_model(y, y_coords)
        # summarize("y_pred GOES", y_pred)
        # show_coords("y_pred_coords", y_pred_coords)

        y_pred_mrms, y_pred_coords_mrms = mrms_model.call_with_conditioning(
            y_mrms, y_coords_mrms, 
            conditioning=y_pred, conditioning_coords=y_pred_coords
        )
        # summarize("y_pred MRMS", y_pred_mrms)
        # show_coords("y_pred_coords_mrms", y_pred_coords_mrms)

        y_next, y_next_coords = goes_model.next_input(y_pred, y_pred_coords, y, y_coords)
        # summarize("y_next GOES", y_next)
        # show_coords("y_next_coords", y_next_coords)

        y_mrms_next, y_mrms_next_coords = mrms_model.next_input(
            y_pred_mrms, y_pred_coords_mrms, y_mrms, y_coords_mrms
        )
        # summarize("y_next MRMS", y_mrms_next)
        # show_coords("y_next_coords_mrms", y_mrms_next_coords)

        # Save raw prediction from this step
        forecast_frames.append(y_pred.detach().cpu())
        forecast_coords.append(y_pred_coords.copy())
        forecast_frames_mrms.append(y_pred_mrms.detach().cpu())
        forecast_coords_mrms.append(y_pred_coords_mrms.copy())

        # Advance sliding input window for next step
        y, y_coords = y_next, y_next_coords
        y_mrms, y_coords_mrms = y_mrms_next, y_mrms_next_coords

# Concatenate along lead_time if possible
# Each y_pred should contain one future lead block produced by the model.
out_lat = goes_model.latitudes.detach().cpu().numpy()
out_lon = goes_model.longitudes.detach().cpu().numpy()
pred_xr_list = []
for i, (pred_torch, coords) in enumerate(zip(forecast_frames, forecast_coords)):
    pred_np = pred_torch.numpy()

    dims = list(coords.keys())
    pred_da = xr.DataArray(pred_np, dims=dims, coords=coords, name="stormscope_goes")
    #pred_da = pred_da.assign_coords(forecast_step=i + 1).expand_dims("forecast_step")
    pred_xr_list.append(pred_da)

out_da = xr.concat(pred_xr_list, dim="lead_time")
out_da.to_netcdf("./stormscope_goes_forecast.nc")

# Concatenate MRMS predictions
pred_xr_list_mrms = []
for i, (pred_torch, coords) in enumerate(zip(forecast_frames_mrms, forecast_coords_mrms)):
    pred_np = pred_torch.numpy()
    dims = list(coords.keys())
    pred_da = xr.DataArray(
        pred_np,
        dims=dims,
        coords=coords,
        name="stormscope_mrms",
    )
    #pred_da = pred_da.assign_coords(forecast_step=i + 1).expand_dims("forecast_step")
    pred_xr_list_mrms.append(pred_da)

out_da_mrms = xr.concat(pred_xr_list_mrms, dim="lead_time")
out_da_mrms.to_netcdf("./stormscope_mrms_forecast.nc")
# # Mask invalid grid cells if available
# if hasattr(goes_model, "valid_mask") and goes_model.valid_mask is not None:
#     valid_mask = goes_model.valid_mask.detach().cpu().numpy()
#     # Broadcast mask over non-spatial dims
#     out_da = out_da.where(valid_mask)
#     out_da_mrms = out_da_mrms.where(valid_mask)

# Build dataset with both predicted variables
ds_out = xr.Dataset(
    {
        "stormscope_goes": out_da,
        "stormscope_mrms": out_da_mrms,
    }
)
# Add lat/lon coordinates as requested
ds_out["out_lat"] = xr.DataArray(out_lat, dims=("y", "x"), 
                             coords={"y": ds_out.y, "x": ds_out.x})
ds_out["out_lon"] = xr.DataArray(out_lon, dims=("y", "x"), 
                             coords={"y": ds_out.y, "x": ds_out.x})

##ds_out.to_netcdf(OUTPUT_FILE)
print(f"Saved forecast to: {OUTPUT_FILE}")