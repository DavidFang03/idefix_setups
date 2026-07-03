from idefix2python import readVTK
import glob
import numpy as np
import matplotlib.pyplot as plt


# betas = ("1e3", "4e3", "7e3", "1e4", "4e4", "6e4", "8e4", "1e5")
betas = ("1e3", "4e3", "7e3", "1e4", "4e4", "6e4", "8e4")
betas_float = [float(beta) for beta in betas]
Hideal = 5.0
epsilon = 0.05
num_r = 10
num_theta = 5
num_size = 40


uids_grid = np.arange(num_r * num_theta * num_size).reshape(num_r, num_theta, num_size)
print(uids_grid.shape)

same_r = uids_grid[6, :, :].flatten()
same_angle = uids_grid[:, 2, :].flatten()
same_pos = [uid for uid in same_r if uid in same_angle]
# uids = list(same_pos)
# same_pos = [uid for uid in same_r if uid in same_angle and uid in same_size]
# uids = list(same_r)


def SIZE(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def angle(v):
    return np.pi / 2 - v.data["PART_X2"]


def get_treshold_size(vtk, uids):
    corona_angle = np.atan(Hideal * 0.05)

    ## determine threshold size
    # sizes = size(vtk)[uids]
    # angles = angle(vtk)[uids]
    angles = []
    sizes = []
    vtkuids = vtk.data["uid"]
    for ii in range(len(vtkuids)):
        if vtkuids[ii] in uids:
            sizes.append(vtk.data["SIZE"][ii])
            angles.append(vtk.data["angle"][ii])
    print(len(sizes))
    iis = np.argsort(np.abs(angles - corona_angle))
    print(f"closest particle: {iis[0]} with size {sizes[iis[0]]}")
    print(f"angle difference: {angles[iis[0]] - corona_angle:.1e} ")
    print(f"next closest particle: {iis[1]} with size {sizes[iis[1]]}")
    print(f"angle difference: {angles[iis[1]] - corona_angle:.1e} ")
    if (angles[iis[1]] - corona_angle) * (angles[iis[0]] - corona_angle) < 0:
        result = (sizes[iis[0]] + sizes[iis[1]]) / 2
        print(f"I would say take the average {result}")
    else:
        result = sizes[iis[0]]
        print(f"I would say keep the first: {result}")
    return result, uids[0]


plt.rcParams.update({"text.usetex": True})

fig, axes = plt.subplots(
    1, squeeze=False, layout="constrained", sharex=True, figsize=(10, 24)
)


last_times = []

# for r in range(1, num_r):
#     for theta in range(num_theta - 1):
for r in (num_r - 1,):
    for theta in (num_theta - 1,):
        same_r = uids_grid[r, :, :].flatten()
        same_angle = uids_grid[:, theta, :].flatten()
        same_pos = [uid for uid in same_r if uid in same_angle]
        uids = list(same_pos)
        print(uids)
        sizes = []
        for beta in betas:
            paths = sorted(
                glob.glob(
                    f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v2/dw100_v2_intro_b{beta}_2000p/vtks/part.*.vtk"
                )
            )
            vtk = readVTK(paths[-1])
            firstvtk = readVTK(paths[0])
            vtk.data["PART_X2"] = vtk.theta
            vtk.data["SIZE"] = SIZE(vtk)
            vtk.data["angle"] = angle(vtk)
            size, uid = get_treshold_size(vtk, uids)
            r_ini = firstvtk.r[uid]
            alt_ini = np.tan(np.pi / 2 - firstvtk.theta[uid]) / epsilon
            sizes.append(size)
            last_times.append(vtk.t[0])

        axes[0, 0].plot(betas_float, sizes, marker="x")
        a, b = np.polyfit(np.log(betas_float), np.log(sizes), deg=1)
        expb = np.exp(b)

        betas_float.append(beta)
        s1e4 = 1e4**a * expb
        axes[0, 0].plot(
            betas_float,
            betas_float**a * expb,
            label=rf"Initial position: $({r_ini:.1f},{alt_ini:.1f}H) \quad \frac{{s}}{{{s1e4:.1e}\mathrm{{m}}}}= \left(\frac{{\beta}}{{10 ^ 4}}\right)^{{{a:.3}}}$",
        )


axes[0, 0].set_xlabel(r"$\beta$")
axes[0, 0].set_ylabel("size [m]")
axes[0, 0].set_xscale("log")
axes[0, 0].set_yscale("log")
axes[0, 0].legend(loc="lower left")
# axes[1, 0].plot(betas_float, last_times)


figpath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/fig1.png"
fig.savefig(figpath, dpi=300)
print(figpath)
