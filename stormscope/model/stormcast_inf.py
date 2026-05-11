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
GFS_CONDITIONING_FILE = f"data/gfs_conditioning_{ts_str}.nc"
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
out_nc = "outputs/stormcast_ensemble_gfsfx.nc"

io = NetCDF4Backend(
    out_nc,
    backend_kwargs={
        "mode": "w",
        "diskless": False,
    },
)
# -----------------------------
# 5.5 diagnostic :: check input data
# -----------------------------
print("Model input variables:")
print(model.input_coords()["variable"])

print("Local HRRR file:")
ds = xr.open_dataset(HRRR_FILE)
print(ds)
arr_name = list(ds.data_vars)[0]
print(ds[arr_name].coords["variable"].values)
# -----------------------------
# 6. Run ensemble
# -----------------------------


nsteps = 4
nensemble = 2
batch_size = 2

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
        "variable": np.array(["t2m", "refc"])
    },
)

print(f"Saved StormCast NetCDF output to: {out_nc}")
