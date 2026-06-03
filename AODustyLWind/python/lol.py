from idefix2python import (
    RunContext,
    Pipeline,
    MapMovie2D,
    OneComponentOneVariable,
    PartQuantity,
    LocalQuantity,
    Fig,
)
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "dw100_v2_b1e4_2000p"

runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    dataFolder=f"{projectPath}/outputs_v2/{task}",
)
RLine = runContext.gridInfo.X1Line
ThetaLine = runContext.gridInfo.X2Line
Rgrid, Thetagrid = np.meshgrid(RLine, ThetaLine)


# def get_part_index(v):
#     r_arr = v.data["PART_X1"]
#     theta_arr = v.data["PART_X2"]

#     # Calculate absolute physical distance to find the true closest grid cell
#     i = np.array([np.argmin(np.abs(RLine - r)) for r in r_arr])
#     j = np.array([np.argmin(np.abs(ThetaLine - th)) for th in theta_arr])

#     return j, i


def zoom(x1, x2):
    # return x1 < 10, x2 <= np.pi / 2
    return np.ones_like(x1, dtype=bool), x2 <= np.pi / 2


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

print("uids", uids)
# uids = [uids[2]]
# uids = [0]


def size(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0e3
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def St(v):
    R = v.data["PART_X1"] * np.sin(v.data["PART_X2"])
    return v.data["TSTOP"] * R ** (-1.5)


def St_custom(v):
    R = v.data["PART_X1"] * np.sin(v.data["PART_X2"])

    rhos = []
    css = []
    for ii, uid in enumerate(v.data["uid"]):
        rho = v.data["RHO_local"][ii]
        prs = v.data["PRS_local"][ii]
        cs = np.sqrt(prs / rho)
        rhos += [rho]
        css += [cs]
    tstop = v.data["DRAGCOEFF"] / (np.asarray(rhos) * np.asarray(css))
    return tstop * R ** (-1.5)


def rhovr(v):
    return v.data["RHO"] * v.data["VX1"]


def z(v):
    return v.data["PART_X1"] * np.cos(v.data["PART_X2"])


def plasmabeta(v):
    P = v.data["PRS"]
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return 8 * np.pi * P / B2


def massloss(v):
    dR = np.diff(v.rl)
    rho, vr, prs = v.data["RHO"], v.data["VX1"], v.data["PRS"]
    thetam4h = np.pi / 2 - np.atan(4 * 0.05)
    thetap4h = np.pi / 2 + np.atan(4 * 0.05)
    jm4h = np.searchsorted(ThetaLine, thetam4h)
    jp4h = np.searchsorted(ThetaLine, thetap4h)
    jmid = np.searchsorted(ThetaLine, np.pi / 2)

    xiup = np.sum(rho[jp4h, :] * vr[jp4h, :] * dR)
    xidown = np.sum(rho[jm4h, :] * vr[jm4h, :] * dR)
    norm = 2 * np.sum(rho[jmid, :] * np.sqrt(prs[jmid, :] / rho[jmid, :]) * dR)

    # print("rho", v.data["PRS"])
    # print(rho[jp4h, :])
    # print(vr[jp4h, :])
    # print(xiup, xidown, norm)

    xi = (xiup - xidown) / norm
    # print(xi)

    return xi


def plasmabeta_local(v):
    p = v.data["PRS_local"]
    b2 = v.data["BX1_local"] ** 2 + v.data["BX2_local"] ** 2 + v.data["BX3_local"] ** 2
    return 8 * np.pi * p / b2


def plasmabeta_local_polo(v):
    p = v.data["PRS_local"]
    b2 = v.data["BX1_local"] ** 2 + v.data["BX2_local"] ** 2
    return 8 * np.pi * p / b2


def dvr(v):
    return v.data["PART_X1"] - v.data["VX1_local"]


def dvtheta(v):
    # j, i = get_part_index(v)
    return v.data["PART_X2"] - v.data["VX2_local"]


def dvphi(v):
    return v.data["PART_X3"] - v.data["VX3_local"]


# def test_r(v):
#     j, i = get_part_index(v)
#     # return v.r[i]
#     print(np.shape(Rgrid))
#     return Rgrid[j, i]


# def test_theta(v):
#     j, i = get_part_index(v)
#     return Thetagrid[j, i]

localquantities = []
for localkey in ["RHO", "VX1", "VX2", "VX3", "PRS", "BX1", "BX2", "BX3"]:
    localquantities.append(
        LocalQuantity(
            f"{localkey}_local",
            localkey=localkey,
            uids=uids,
            plot_coords=[3, 4],
        )
    )
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
        yscale="log",
        # bounds=[0.01, 0.05],
    ),
    # PartQuantity(
    #     "St_custom",
    #     r"St_custom",
    #     uids=uids,
    #     plot_coords=[0, 2],
    #     compute=St_custom,
    #     yscale="log",
    #     # bounds=[0.01, 0.05],
    # ),
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
        compute=z,
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
    PartQuantity(
        "dvr", r"$v_r^\mathrm{d}-v_r$", uids=uids, plot_coords=[0, 4], compute=dvr
    ),
    PartQuantity(
        "dvtheta",
        r"$v_\theta^\mathrm{d}-v_\theta$",
        uids=uids,
        plot_coords=[1, 4],
        compute=dvtheta,
    ),
    PartQuantity(
        "dvphi",
        r"$v_\phi^\mathrm{d}-v_\phi$",
        uids=uids,
        plot_coords=[2, 4],
        compute=dvphi,
    ),
    MapMovie2D("beta", "$\\beta$", plot_coords=[0, 5], norm="log", compute=plasmabeta),
    PartQuantity(
        "beta_local",
        r"$\beta^\mathrm{local}$",
        uids=uids,
        plot_coords=[1, 5],
        compute=plasmabeta_local,
    ),
    PartQuantity(
        "beta_local",
        r"$\beta_\mathrm{pol}^\mathrm{local}$",
        uids=uids,
        plot_coords=[2, 5],
        compute=plasmabeta_local_polo,
    ),
    PartQuantity("SIZE", r"size (m)", uids=uids, plot_coords=[3, 3], compute=size),
    OneComponentOneVariable(
        "xi", r"$\xi$", plot_coords=[3, 5], compute=massloss, bounds=[None, None]
    ),
]
for qty in quantities:
    qty.uids = uids
fig1 = Fig(localquantities + quantities)
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(
    runContext,
    [fig1],
    zoom=zoom,
)

pipeline.run()
