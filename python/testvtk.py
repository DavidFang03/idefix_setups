# %%
from idefix2python import readVTK
import numpy as np
import matplotlib.pyplot as plt
import glob

for i in range(2):
    path = f"/home/dfang/Code/idefix_setups/RadialDrift/outputs/Drift_Tau/vtks/part.000{i}.vtk"
    # path = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs/reload_Epstein2_wind/vtks/part.0000.vtk"
    vtk = readVTK(path)
    data = vtk.data
    print(data.keys())

    print(vtk.r)
    print(vtk.theta)
    print(vtk.phi)
    print(data["VX1"])
    print(data["VX2"])
    print(data["VX3"])

# %%
