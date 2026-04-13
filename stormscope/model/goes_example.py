import os
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
# from dotenv import load_dotenv

# load_dotenv()

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.data import GFS_FX, GOES, MRMS, fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)

print("All imports successful. StormScope is ready to run!")