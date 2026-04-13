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