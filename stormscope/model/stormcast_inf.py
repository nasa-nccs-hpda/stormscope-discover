import os
import numpy as np
import earth2studio.run as run
import pandas as pd
import xarray as xr

from earth2studio.data import DataArrayFile, HRRR, GFS_FX
from earth2studio.models.px import StormCast
from earth2studio.perturbation import Zero
from earth2studio.io import NetCDF4Backend

os.makedirs("outputs", exist_ok=True)

date = [np.datetime64("2024-09-26T12:00:00")]
ts_str = pd.to_datetime(date[0]).strftime("%Y%m%d_%H%M%S")
GFS_CONDITIONING_FILE = f"data/stormcast_conditioning_validtime_{ts_str}.nc"
HRRR_FILE = f"data/hrrr_{ts_str}.nc"

# -----------------------------
# 1. StormCast conditioning source
# -----------------------------
# GFS_FX provides GFS forecast fields, not just initial state.
# conditioning_data_source = GFS_FX(
#     source="aws",      # or "ncep"
#     cache=True,
#     verbose=True,
# )
conditioning_data_source = DataArrayFile(GFS_CONDITIONING_FILE)
# class DebugGFSFX:
#     def __init__(self):
#         self.src = GFS_FX(source="aws", cache=True, verbose=True)

#     def __call__(self, time, variable):
#         print("\n===== StormCast requested GFS_FX =====", flush=True)
#         print("time:", time, flush=True)
#         print("variable:", variable, flush=True)

#         # Force stop so you can see the request before downloading
#         raise RuntimeError("Stop here after printing GFS_FX request")

#         # return self.src(time, variable)
# conditioning_data_source = DebugGFSFX()
# -----------------------------
# 2. Load StormCast model
# -----------------------------
package = StormCast.load_default_package()

model = StormCast.load_model(
    package,
    conditioning_data_source=conditioning_data_source,
)

# -----------------------------
# 3. HRRR initial-condition source
# -----------------------------
# Standard StormCast example uses HRRR for mesoscale initialization.
# data = HRRR(
#     source="aws",
#     cache=True,
#     verbose=True,
# )
data = DataArrayFile(HRRR_FILE)

# -----------------------------
# 4. Perturbation
# -----------------------------
# StormCast handles ensemble stochasticity internally.
perturbation = Zero()

# -----------------------------
# 5. NetCDF output
# -----------------------------
out_nc = f"outputs/stormcast_ensemble_gfsfx_{ts_str}.nc"

io = NetCDF4Backend(
    out_nc,
    backend_kwargs={
        "mode": "w",
        "diskless": False,
    },
)
# -----------------------------
# 5.5 diagnostic :: check input data and conditioning variables
# -----------------------------
def print_file_vars(path, label):
    ds = xr.open_dataset(path)
    arr_name = list(ds.data_vars)[0]
    file_vars = ds[arr_name].coords["variable"].values

    print(f"\n{label}: {path}")
    print("array name:", arr_name)
    print("nvars:", len(file_vars))
    print(file_vars)

print_file_vars(HRRR_FILE, "HRRR initial file")
print_file_vars(GFS_CONDITIONING_FILE, "GFS conditioning file")

# Try to reveal what StormCast asks from conditioning source
print("\nStormCast input variables:")
print(model.input_coords()["variable"])

if hasattr(model, "conditioning_coords"):
    print("\nStormCast conditioning variables:")
    print(model.conditioning_coords()["variable"])
# -----------------------------
# 6. Run ensemble
# -----------------------------


nsteps = 5
nensemble = 4
batch_size = 2
out_vars = ["u10m", "v10m", "t2m", "mslp", "refc"]  # specify which variables to save in output
io = run.ensemble(
    date,
    nsteps,
    nensemble,
    model,
    data,
    io,
    perturbation,
    batch_size=batch_size,
    output_coords={
        "variable": np.array(out_vars)
    },
)

print(f"Saved StormCast NetCDF output to: {out_nc}")
