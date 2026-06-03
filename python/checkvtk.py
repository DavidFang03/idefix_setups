# %%
from idefix2python import readVTK
import numpy as np
import matplotlib.pyplot as plt

# path = "/home/dp316/dp316/dc-fang1/IdefixRuns/RadialDrift/outputs/Drift_Tau/vtks/data.0000.vtk"
# path = "/home/dp316/dp316/dc-fang1/IdefixRuns/RadialDrift/setup_l/part.0000.vtk"
path = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs/dw100_b1e3_1000p/vtks/data.0000.vtk"
vtk = readVTK(path)
data = vtk.data
# print(vtk.dimensions)
print(vtk.geometry)
# print(vars(vtk))
attrs = vars(vtk)
print(attrs.keys())
print(np.shape(vtk.rl))
dr = np.diff(vtk.rl)
print(dr.shape)
print(np.shape(vtk.r))
# print(", ".join("%s: %s" % item for item in attrs.keys()))


# print(vtk.geometry)

# print(np.average(vtk.data["Dust0_RHO"]))
# print(vtk.x)

# for key in data:
#     print(key)

# import glob

# for path in glob.glob(
#     "/home/dp316/dp316/dc-fang1/IdefixRuns/RadialDrift/outputs/Drift_Tau/vtks/*.vtk"
# ):
#     print(path)
#     vtk = readVTK(path)
#     print(np.min(vtk.x))
#     # plt.plot(vtk.x)
#     # plt.savefig("test.png")

# v = readVTK(
#     "/home/dp316/dp316/dc-fang1/IdefixRuns/RadialDrift/outputs/Drift_Tau/vtks/data.0000.vtk"
# )
# plt.plot(v.x, v.data["RHO"][:, :, 0])


# %%
