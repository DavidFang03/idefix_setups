from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyWind"
task = "restart_dust"

custom_fields2D = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
        particles="all",
    ),
    # MapMovie2D(
    #     "Dust0_RHO",
    #     r"$\rho$",
    #     plot_coords=[0, 1],
    #     title="Density",
    #     streamlines=["VX1", "VX2"],
    #     particles="all",
    # ),
]

# Initialize context
runContext = RunContext(
    task,
    projectPath,
    partFolder="/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/setup_l",
)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, movies2D=custom_fields2D)

pipeline.run()
