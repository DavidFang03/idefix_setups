from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/Idefix2Python/config/config.json"
task = "clean_reload_wind"


def compute_mach_p(data):
    return np.sqrt(data["VX1"] ** 2 + data["VX2"] ** 2) / data["cs"]


custom_fields2D = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
        # uids="all",
        uids="all",
    ),
    MapMovie2D(
        "VX1", r"$v_x$", plot_coords=[1, 0], streamlines=["VX1", "VX2"], uids="all"
    ),
]


def z(v):
    r = v.data["PART_X1"]
    theta = v.data["PART_X2"]
    return r * np.cos(theta)


pqs = [
    PartQuantity(
        "PART_X1",
        "PART_X1",
        uids="all",
        # uids="all",
        plot_coords=[0, 0],
    ),
    PartQuantity(
        "PART_X2",
        "PART_X2",
        uids="all",
        plot_coords=[0, 1],
    ),
    PartQuantity("z", "z", uids="all", plot_coords=[0, 2], compute=z),
]


# Initialize context
runContext = RunContext(task, projectPath, configPath=configPath)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, partQuantities=pqs, movies2D=custom_fields2D)

pipeline.run()
