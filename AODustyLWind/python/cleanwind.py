from idefix2python import RunContext, Pipeline, MapMovie2D, Fig
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "clean_wind_100_b1e4"
# task = "clean_wind_20_b1e4_axis"


def compute_mach_p(data):
    cs2 = data["PRS"] / data["RHO"]
    return np.sqrt(data["VX1"] ** 2 + data["VX2"] ** 2) / cs2


quantities = [
    MapMovie2D(
        "RHO",
        plot_coords=[0, 0],
        streamlines=["VX1", "VX2"],
    ),
    MapMovie2D("InvDt", plot_coords=[0, 1], streamlines=["VX1", "VX2"]),
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
