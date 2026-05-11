import numpy as np
import torch

from earth2studio.data import (
    DataArrayFile,
    GFS_FX,
    fetch_data,
)
from earth2studio.models.px import StormCast

# -------------------------------------------------
# Debug wrapper for conditioning source
# -------------------------------------------------

class DebugGFSFX:
    def __init__(self):
        self.src = GFS_FX(
            source="aws",
            cache=True,
            verbose=True,
        )

    def __call__(self, time, variable):

        print("\n===== StormCast requested GFS_FX =====")
        print("time:")
        print(time)

        print("\nvariable:")
        print(variable)

        raise RuntimeError("STOP AFTER PRINT")

        # return self.src(time, variable)


# -------------------------------------------------
# Load StormCast
# -------------------------------------------------

conditioning_data_source = DebugGFSFX()

package = StormCast.load_default_package()

model = StormCast.load_model(
    package,
    conditioning_data_source=conditioning_data_source,
)

# -------------------------------------------------
# Load local HRRR initial condition
# -------------------------------------------------

data = DataArrayFile(
    "data/hrrr_20240926_120000.nc"
)

# -------------------------------------------------
# Fetch model-required inputs
# -------------------------------------------------

time = np.array(
    ["2024-09-26T12:00:00"],
    dtype="datetime64[ns]",
)

input_coords = model.input_coords()

x, coords = fetch_data(
    source=data,
    time=time,
    variable=input_coords["variable"],
    lead_time=np.array([0], dtype="timedelta64[h]"),
    device=torch.device("cuda"),
)

print("\nFetched local HRRR successfully")
print("x shape:", x.shape)

# -------------------------------------------------
# DIRECT StormCast call
# -------------------------------------------------

with torch.no_grad():
    y, y_coords = model(x, coords)

print("SUCCESS")