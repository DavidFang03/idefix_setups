from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity, Fig
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/Idefix2Python/config/config.json"
task = "dw100_RD"


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
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
        uids=uids,
    ),
    # MapMovie2D(
    #     "VX1", r"$v_x$", plot_coords=[1, 0], streamlines=["VX1", "VX2"], uids=uids
    # ),
    MapMovie2D(
        "VX3", r"$v_\phi$", plot_coords=[2, 0], streamlines=["VX1", "VX2"], uids=uids
    ),
    MapMovie2D(
        "BX1",
        r"BX1",
        plot_coords=[0, 2],
        streamlines=["BX1", "BX2"],
        uids=uids,
    ),
    MapMovie2D(
        "BX2",
        r"BX2",
        plot_coords=[1, 2],
        streamlines=["BX1", "BX2"],
        uids=uids,
    ),
    # PartQuantity(
    #     "PART_X2",
    #     "PART_X2",
    #     uids=uids,
    #     plot_coords=[1, 1],
    # ),
    PartQuantity("St", r"St", uids=uids, plot_coords=[0, 1], compute=St, yscale="log"),
    PartQuantity("PART_X1", "PART_X1", uids=uids, plot_coords=[1, 1]),
    PartQuantity(
        "PART_X2",
        r"$\theta^\mathrm{d}$",
        uids=uids,
        plot_coords=[2, 1],
    ),
]
fig1 = Fig(quantities)
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context
runContext = RunContext(task, projectPath, configPath=configPath)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

pipeline.run()
