from idefix2python import RunContext, Pipeline, MapMovie2D, Fig
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
# projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
# task = "cw_20_b1e4"
task = "clean_wind_100_b2e4"

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
runContext = RunContext(task, projectPath, configPath=configPath)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

pipeline.run()
