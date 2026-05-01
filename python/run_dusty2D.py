from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/Idefix2Python/config/config.json"
task = "test_dustless"


def compute_mach_p(data):
    return np.sqrt(data["VX1"] ** 2 + data["VX2"] ** 2) / data["cs"]


custom_fields2D = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
        particles="all",
    ),
]

partQuantities = [PartQuantity("PART_X1", r"$r_\mathrm{dust}$")]


# Initialize context
runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    partFolder="/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/setup_l",
)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, movies2D=custom_fields2D, partQuantities=partQuantities)

pipeline.run()
