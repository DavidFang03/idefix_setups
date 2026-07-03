from idefix2python import RunContext, Pipeline, MapMovie2D, Fig, OneComponentOneVariable
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
# projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
# task = "cw_20_b1e4"
task = "clean_wind_100_v2_b1e4"

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "dw100_v2_intro_b8e4_2000p"


# runContext = RunContext(task, projectPath, configPath=configPath)
runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    dataFolder=f"{projectPath}/outputs_v2/{task}/vtks",
    iniPath=f"{projectPath}/inputs_v2/{task}.ini",
)

eps = 0.05


def T(v):
    data = v.data
    return data["PRS"] / data["RHO"]


def compute_mach_p(v):
    data = v.data
    cs2 = data["PRS"] / data["RHO"]
    return np.sqrt(data["VX1"] ** 2 + data["VX2"] ** 2) / cs2


def cs(v):
    data = v.data
    return np.sqrt(data["PRS"] / data["RHO"])


def Rm(v):
    # d = v.data
    r, theta = np.meshgrid(v.r, v.theta)
    R = r * np.sin(theta)
    return R ** (-1.5) * eps**2 / v.data["eta"]


def ElA(v):
    d = v.data
    # return d["Am"]
    r, theta = np.meshgrid(v.r, v.theta)
    R = r * np.sin(theta)
    B2 = d["BX1"] ** 2 + d["BX2"] ** 2 + d["BX3"] ** 2
    return B2 * R**1.5 / (d["RHO"] * d["Am"])


def massloss(v):
    dR = np.diff(v.rl)
    dTheta = np.diff(v.thetal)
    R = v.r
    ThetaLine = v.theta
    rho, vr, prs = v.data["RHO"], v.data["VX1"], v.data["PRS"]
    vtheta = v.data["VX2"]
    thetam4h = np.pi / 2 - np.atan(5 * 0.05)
    thetap4h = np.pi / 2 + np.atan(5 * 0.05)
    jm4h = np.searchsorted(ThetaLine, thetam4h)
    jp4h = np.searchsorted(ThetaLine, thetap4h)
    ir10 = np.searchsorted(R, 40)
    rmax = 50
    ir50 = np.searchsorted(R, rmax)
    jmid = np.searchsorted(ThetaLine, np.pi / 2)

    xiup = np.sum(rho[jp4h, ir10:ir50] * vtheta[jp4h, ir10:ir50] * dR[ir10:ir50])
    xidown = np.sum(rho[jm4h, ir10:ir50] * vtheta[jm4h, ir10:ir50] * dR[ir10:ir50])
    norm = 2 * np.sum(rho[jmid, :] * np.sqrt(prs[jmid, :] / rho[jmid, :]) * dR)

    # norm = (
    #     -2
    #     * np.pi
    #     * R[ir50]
    #     * np.sum(
    #         rho[jm4h:jp4h, ir50] * vr[jm4h:jp4h, ir50] * R[ir50] * dTheta[jm4h:jp4h]
    #     )
    # )
    # norm = 1

    xi = (xiup - xidown) / norm
    # print(xi)

    return xi


# def massloss(v):
#     # Grids
#     R = v.r
#     ThetaLine = v.theta
#     dR = np.diff(v.rl)
#     dTheta = np.diff(v.thetal)

#     # Extract data arrays
#     rho = v.data["RHO"]
#     vr = v.data["VX1"]
#     vtheta = v.data["VX2"]

#     # Define boundaries based on scale height (H/R = 0.05)
#     h_r = 0.05
#     thetam4h = np.pi / 2 - np.atan(5 * h_r)
#     thetap4h = np.pi / 2 + np.atan(5 * h_r)

#     # Find indices
#     jm4h = np.searchsorted(ThetaLine, thetam4h)
#     jp4h = np.searchsorted(ThetaLine, thetap4h)
#     ir10 = np.searchsorted(R, 40)
#     ir50 = np.searchsorted(R, 50)

#     # Slices for readability
#     r_slice = slice(ir10, ir50)

#     # Shell elements: r * dr
#     r_element = R[r_slice] * dR[r_slice]

#     # Calculate Theta Fluxes (including 2*pi*sin(theta)*r*dr physical scaling)
#     # Upper boundary
#     xiup = (
#         2
#         * np.pi
#         * np.sin(ThetaLine[jp4h])
#         * np.sum(rho[jp4h, r_slice] * vtheta[jp4h, r_slice] * r_element)
#     )
#     # Lower boundary
#     xidown = (
#         2
#         * np.pi
#         * np.sin(ThetaLine[jm4h])
#         * np.sum(rho[jm4h, r_slice] * vtheta[jm4h, r_slice] * r_element)
#     )

#     # Net mass loss escaping the wedge boundaries
#     xi_net = xiup - xidown

#     # Optional: If you ever want to turn normalization back on properly
#     theta_slice = slice(jm4h, jp4h)
#     sin_theta = np.sin(ThetaLine[theta_slice])
#     norm = (
#         2
#         * np.pi
#         * (R[ir50] ** 2)
#         * np.sum(
#             rho[theta_slice, ir50]
#             * vr[theta_slice, ir50]
#             * sin_theta
#             * dTheta[theta_slice]
#         )
#     )

#     # norm = 1.0

#     return xi_net / norm


inferno = {"cmap": "inferno"}
quantities = [
    MapMovie2D(
        "RHO",
        plot_coords=[0, 0],
        streamlines=["VX1", "VX2"],
    ),
    MapMovie2D(
        "InvDt", plot_coords=[0, 1], streamlines=["VX1", "VX2"], bounds=[1e1, 1e3]
    ),
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
        "Mach_p",
        plot_coords=[4, 0],
        title="Poloidal Mach Number",
        compute=compute_mach_p,
        contours=[1],
        contour_color="green",
    ),
    MapMovie2D(
        "Rm",
        r"$\mathrm{Rm}$",
        plot_coords=[0, 2],
        bounds=[1e-1, 1e12],
        norm="log",
        compute=Rm,
        style_kwargs=inferno,
    ),
    MapMovie2D(
        "ElA",
        r"$\Lambda_\mathrm{A}$",
        plot_coords=[1, 2],
        bounds=[1e-1, 1e12],
        norm="log",
        compute=ElA,
        style_kwargs=inferno,
    ),
    MapMovie2D(
        "T",
        "Temperature",
        plot_coords=[2, 2],
        compute=T,
        bounds=[1e-4, 1e-1],
        norm="log",
        style_kwargs=inferno,
    ),
    OneComponentOneVariable(
        "xi",
        r"$\xi$",
        plot_coords=[4, 1],
        compute=massloss,
        # bounds=[-1e-2, 0],
        # ymin=-1e-2,
        # ymax=1e-2,
    ),
]
fig1 = Fig(quantities)
# fig1 = Fig(
#     [
#         MapMovie2D("InvDt", plot_coords=[0, 0], streamlines=["VX1", "VX2"]),
#     ]
# )
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context


# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

pipeline.run()
