# %%
from idefix2python import readVTK
import numpy as np
import matplotlib.pyplot as plt

# path = "/home/dp316/dp316/dc-fang1/IdefixRuns/RadialDrift/outputs/Drift_Tau/vtks/data.0000.vtk"
# path = "/home/dp316/dp316/dc-fang1/IdefixRuns/RadialDrift/setup_l/part.0000.vtk"
path = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs/dw100_b1e3_1000p/vtks/part.0000.vtk"
vtk = readVTK(path)
data = vtk.data
# print(vtk.dimensions)
print(vtk.geometry)
print(vtk.z)

num_r = 10
num_theta = 10
num_size = 10

uids_grid = np.arange(num_r * num_theta * num_size).reshape(num_r, num_theta, num_size)

same_r = uids_grid[9, :, :].flatten()
same_angle = uids_grid[:, 9, :].flatten()
same_size = uids_grid[:, :, 0].flatten()
same_pos = [uid for uid in same_r if uid in same_angle]
# uids = list(same_r)
uids = list(same_pos)

# print(vtk.r)
for key in data:
    print(key)
    print(np.shape(data[key]))

print(vtk.r[uids])
print(vtk.theta[uids])
print(data["DRAGCOEFF"][uids])

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
