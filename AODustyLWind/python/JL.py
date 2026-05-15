from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity, Fig
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "dw_JL"


def z(v):
    r = v.data["PART_X1"]
    theta = v.data["PART_X2"]
    return r * np.cos(theta)


def St(v):
    r = v.x
    return v.data["TSTOP"] * r ** (-1.5)


uids = "all"

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
    PartQuantity("St", r"St", uids=uids, plot_coords=[0, 2], compute=St, yscale="log"),
    PartQuantity("PART_X1", "PART_X1", uids=uids, plot_coords=[1, 2]),
    PartQuantity(
        "PART_X2",
        r"$\theta^\mathrm{d}$",
        uids=uids,
        plot_coords=[2, 2],
    ),
]
for qty in quantities:
    qty.uids = "all"
fig1 = Fig(quantities)
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context
runContext = RunContext(task, projectPath, configPath=configPath)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

pipeline.run()
