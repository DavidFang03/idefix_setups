from idefix2python import readVTK
import glob
import matplotlib.pyplot as plt
import numpy as np

# %%
# plt.figure(figsize=(9, 9))
zoom = False


vtklist = sorted(
    glob.glob(
        "/home/dp316/dp316/dc-fang1/IdefixRuns/Test_Epstein/outputs/test_/vtks/data.000*"
    )[::3]
)

partlist = sorted(
    glob.glob("/home/dp316/dp316/dc-fang1/IdefixRuns/Test_Epstein/setup/part.000*")
)[::3]

if len(partlist) == 0:
    raise Exception("no part found")
if len(vtklist) == 0:
    raise Exception("no data found")

rows = range(3)
fig, axs = plt.subplots(len(rows), len(partlist), figsize=(24, 18))


# for i in range(len(vtklist)):
#     vtk = readVTK(vtklist[i])
#     part = readVTK(partlist[i])
#     print(part.data.keys())
#     print(part.data["t_stop"])
#     X = vtk.r
#     for j in rows:
#         jj = 10 * j
#         theta = vtk.theta[jj]
#         t = vtk.t[0]
#         axs[j, i].plot(
#             vtk.theta,
#             np.squeeze(vtk.data["RHO"])[jj, :],
#             label=rf"$\rho \, \mathrm{{at}} \, t={t:.1e} \quad  \theta={theta:.1e}$",
#             marker="+",
#             lw=0.1,
#             alpha=0.2,
#         )

#         axs[j, i].scatter(part.theta, part.data["t_stop"], alpha=0.75, marker="x")

#         axs[j, i].legend(fontsize=12, loc="upper right")
#         axs[j, i].set_xlim(0.2, 0.35)
#         if j == 0:
#             axs[j, i].set_title(f"t={t}")

for i in range(len(vtklist)):
    vtk = readVTK(vtklist[i])
    part = readVTK(partlist[i])
    print(part.data.keys())
    print(part.data["t_stop"])
    X = vtk.r
    for j in rows:
        jj = 10 * j
        theta = vtk.theta[jj]
        t = vtk.t[0]
        axs[j, i].plot(
            vtk.r,
            np.squeeze(vtk.data["RHO"])[:, jj],
            label=rf"$\rho \, \mathrm{{at}} \, t={t:.1e} \quad  \theta={theta:.1e}$",
            marker="+",
            lw=0.1,
            alpha=0.2,
        )

        axs[j, i].scatter(part.r, part.data["t_stop"], alpha=0.75, marker="x")

        axs[j, i].legend(fontsize=12, loc="upper right")
        if zoom:
            axs[j, i].set_xlim(10.8, 11.2)
        for tstop in part.data["t_stop"]:
            axs[j, i].axhline(y=tstop, alpha=0.15, lw=0.2)
        if j == 0:
            axs[j, i].set_title(f"t={t}")

path = "test_epstein.png"
fig.savefig(path, dpi=300)
print(path)

# %%
