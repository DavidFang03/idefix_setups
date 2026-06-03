from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity, Fig
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "clean_wind_100_b3e3"


def zoom(x1, x2):
    return x1 < 5, np.ones_like(x2, dtype=bool)
    return x1 < 10, x2 <= np.pi / 2


def rhovr(v):
    return v.data["RHO"] * v.data["VX1"]


def plasmabeta(v):
    P = v.data["PRS"]
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return 8 * np.pi * P / B2


quantities = [
    MapMovie2D(
        "RHO",
        plot_coords=[0, 1],
        streamlines=["VX1", "VX2"],
    ),
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
    MapMovie2D(
        "beta",
        "$\\beta$",
        plot_coords=[0, 2],
        norm="log",
        compute=plasmabeta,
        bounds=[None, 2.1e4],
    ),
    # MapMovie2D("rhovr", r"$\rho v_r$",, plot_coords=[3, 3], compute=rhovr),
]

fig1 = Fig(quantities)

# Initialize context
runContext = RunContext(task, projectPath, configPath=configPath)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(
    runContext,
    [fig1],
    zoom=zoom,
)

pipeline.run()
