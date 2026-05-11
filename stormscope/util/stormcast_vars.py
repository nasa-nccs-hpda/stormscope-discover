import numpy as np
import pandas as pd
from earth2studio.data import GFS_FX
from earth2studio.models.px import StormCast

class DebugGFSFX:
    def __init__(self):
        self.src = GFS_FX(source="aws", cache=True, verbose=True)

    def __call__(self, time, variable):
        print("\nStormCast requested GFS_FX conditioning:")
        print("time:")
        print(time)
        print("variable:")
        print(variable)
        return self.src(time, variable)


conditioning_data_source = DebugGFSFX()
package = StormCast.load_default_package()
model = StormCast.load_model(
    package,
    conditioning_data_source=conditioning_data_source,
)