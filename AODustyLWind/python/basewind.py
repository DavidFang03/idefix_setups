from idefix2python import RunContext, Pipeline, MapMovie2D, OneComponentOneVariable, Fig
import numpy as np
import matplotlib.pyplot as plt

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "clean_wind_100_v2_b1e3"

epsilon = 0.1


def zoom(x1, x2):
    return x1 < 5, np.ones_like(x2, dtype=bool)


def Rm(v):
    return Rgrid ** (-1.5) * 0.05**2 / v.data["eta"]


def fourH(ax, v):
    ax.plot(RLine, 4 * RLine * epsilon, ls="--", color="white", lw=1)
    ax.plot(RLine, -4 * RLine * epsilon, ls="--", color="white", lw=1)


runContext = RunContext(task, projectPath, configPath=configPath)
grid1, grid2 = runContext.gridInfo.grid1, runContext.gridInfo.grid2

RLine = runContext.gridInfo.X1Line
ThetaLine = runContext.gridInfo.X2Line
Rgrid, Thetagrid = np.meshgrid(RLine, ThetaLine)


def rhovr(v):
    return v.data["RHO"] * v.data["VX1"]


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


def circle_min_dt(ax, v):
    """Draws a circle at the location of the minimum time step (maximum InvDt).

    v: The Idefix/Pluto data container
    grid1: 2D array of coordinates (e.g., R or X1)
    grid2: 2D array of coordinates (e.g., Z or X2)
    """
    # 1. Find the flat index of the maximum value
    flat_idx = np.argmax(v.data["InvDt"])

    # 2. Unravel the flat index back into 2D (i, j) coordinates
    i, j = np.unravel_index(flat_idx, v.data["InvDt"].shape)

    # 3. Retrieve the physical coordinates from your grid meshes
    center_x = grid1[i, j]
    center_y = grid2[i, j]

    # 4. Create the Circle patch and add it to the axes
    # Adjust the radius and styling as needed for your simulation scale
    circle = plt.Circle(
        (center_x, center_y), radius=0.1, edgecolor="red", facecolor="none", lw=2
    )
    ax.add_patch(circle)

    # Optional: print the location to your terminal for debugging
    print(
        f"Max InvDt found at index ({i}, {j}) -> Coord: ({center_x:.2f}, {center_y:.2f})"
    )


def ElA(v):
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return B2 / v.data["xA"]


def RmO(v):
    etaO = v.data["eta"]
    Rcyl = Rgrid * np.sin(Thetagrid)
    Omega = Rcyl ** (-1.5)
    H = 0.05 * Rcyl
    return Omega * H**2 / etaO


quantities = [
    MapMovie2D(
        "RHO",
        plot_coords=[0, 1],
        streamlines=["VX1", "VX2"],
    ),
    # MapMovie2D(
    #     "InvDt", plot_coords=[0, 0], streamlines=["VX1", "VX2"], customize=circle_min_dt
    # ),
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
    MapMovie2D(
        "beta",
        "$\\beta$",
        plot_coords=[0, 2],
        norm="log",
        compute=plasmabeta,
        bounds=[None, 2.1e4],
    ),
    OneComponentOneVariable(
        "xi",
        r"$\xi$",
        plot_coords=[1, 2],
        compute=massloss,
        bounds=[None, None],
    ),
    # MapMovie2D(
    #     "Rm",
    #     "Rm",
    #     plot_coords=[0, 3],
    #     norm="log",
    #     compute=Rm,
    #     bounds=[1, 1e12],
    # ),
    # MapMovie2D("rhovr", r"$\rho v_r$",, plot_coords=[3, 3], compute=rhovr),
]

fig1 = Fig(quantities)

# Initialize context

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(
    runContext,
    [fig1],
    # zoom=zoom,
    no_movie=True,
)

pipeline.run()
