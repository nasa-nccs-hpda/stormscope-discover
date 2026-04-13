"""stormscope_simple_inference.py - Using Earth2Studio's inference utilities"""

import numpy as np
import torch
from datetime import datetime

from earth2studio.data import DataArrayFile
from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES
from earth2studio.models.auto import Package
from earth2studio.io import NetCDF4Backend

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

print(f"✓ Model loaded on {DEVICE}")

# Get initial time from data
init_time = goes_local.get_times()[0]
print(f"Initial time: {init_time}")

# Set up output backend
output_backend = NetCDF4Backend(OUTPUT_FILE)

# Run inference
print(f"\nRunning {NUM_STEPS}-step forecast...")

try:
    # Simple 2-step forecast
    predictions = []
    
    # Get initial input
    input_coords = goes_model.input_coords()
    current_data = goes_local(
        time=init_time,
        variable=np.array(input_coords['variable']),
        lead_time=input_coords['lead_time']
    )
    
    coords = input_coords.copy()
    
    for step in range(NUM_STEPS):
        print(f"  Step {step + 1}/{NUM_STEPS}...")
        
        with torch.no_grad():
            pred = goes_model(x=current_data, coords=coords)
        
        predictions.append(pred)
        
        # Prepare next input if not last step
        if step < NUM_STEPS - 1:
            import xarray as xr
            next_input_t0 = current_data.isel(lead_time=-1)
            next_input_t1 = pred.isel(lead_time=0)
            
            current_data = xr.concat(
                [next_input_t0.expand_dims('lead_time'),
                 next_input_t1.expand_dims('lead_time')],
                dim='lead_time'
            )
    
    # Save results
    import xarray as xr
    all_preds = xr.concat(predictions, dim='lead_time')
    all_preds.to_netcdf(OUTPUT_FILE)
    
    print(f"\n✓ Forecast complete! Saved to {OUTPUT_FILE}")
    print(f"  Output shape: {all_preds.shape}")
    print(f"  Variables: {list(all_preds.coords['variable'].values)}")

except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()