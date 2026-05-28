from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity, Fig
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "dw100_b3e3_1000p"

uids = range(0, 10)
uids = "all"


num_r = 10
num_theta = 10
num_size = 20

uids_grid = np.arange(num_r * num_theta * num_size).reshape(num_r, num_theta, num_size)

same_r = uids_grid[9, :, :].flatten()
same_angle = uids_grid[:, 9, :].flatten()
same_size = uids_grid[:, :, 0].flatten()

same_pos = [uid for uid in same_r if uid in same_angle]
# uids = list(same_r)
uids = list(same_pos)

print(uids)

# def z(v):
#     # r = v.data["PART_X1"]
#     # theta = v.data["PART_X2"]
#     r = v.r
#     theta = v.theta
#     return r * np.cos(theta)


def size(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0e3
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def St(v):
    return v.data["TSTOP"] * v.data["PART_X1"] ** (-1.5)


def rhovr(v):
    return v.data["RHO"] * v.data["VX1"]


def z(v):
    return v.data["PART_X1"] * np.cos(v.data["PART_X2"])


def plasmabeta(v):
    P = v.data["PRS"]
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return 8 * np.pi * P / B2


quantities = [
    MapMovie2D("RHO", plot_coords=[0, 1], streamlines=["VX1", "VX2"], uids=uids),
    MapMovie2D("InvDt", plot_coords=[0, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX1", plot_coords=[1, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX2", plot_coords=[2, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX3", plot_coords=[3, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D(
        "BX1",
        plot_coords=[1, 1],
        streamlines=["BX1", "BX2"],
    ),
    MapMovie2D(
        "BX2",
        plot_coords=[2, 1],
        streamlines=["BX1", "BX2"],
    ),
    MapMovie2D(
        "BX3",
        plot_coords=[3, 1],
        streamlines=["BX1", "BX2"],
    ),
    PartQuantity(
        "St",
        r"St",
        uids=uids,
        plot_coords=[0, 2],
        compute=St,
        bounds=[0.01, 0.05],
    ),
    PartQuantity("PART_X1", "$r^\\mathrm{dust}$", uids=uids, plot_coords=[1, 2]),
    PartQuantity(
        "PART_X2",
        r"$\theta^\mathrm{d}$",
        uids=uids,
        plot_coords=[2, 2],
    ),
    PartQuantity(
        "z",
        r"$z^\mathrm{d}$",
        uids=uids,
        plot_coords=[3, 2],
    ),
    PartQuantity(
        "PART_VX1",
        r"$v_r^\mathrm{d}$",
        uids=uids,
        plot_coords=[0, 3],
    ),
    PartQuantity(
        "PART_VX2",
        r"$v_\theta^\mathrm{d}$",
        uids=uids,
        plot_coords=[1, 3],
    ),
    PartQuantity(
        "PART_VX3",
        r"$v_\phi^\mathrm{d}$",
        uids=uids,
        plot_coords=[2, 3],
    ),
    PartQuantity("SIZE", r"size (m)", uids=uids, plot_coords=[3, 3], compute=size),
    MapMovie2D("beta", "$\\beta$", plot_coords=[0, 4], norm="log"),
    # MapMovie2D("rhovr", r"$\rho v_r$", uids=uids, plot_coords=[3, 3], compute=rhovr),
]
for qty in quantities:
    qty.uids = uids
fig1 = Fig(quantities)
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context
runContext = RunContext(task, projectPath, configPath=configPath)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

pipeline.run()
