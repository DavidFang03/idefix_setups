from idefix2python import RunContext, Pipeline, MapMovie2D, Fig

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/Idefix2Python/config/config.json"
task = "clean_wind_100"


quantities = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
    ),
    MapMovie2D("VX1", r"$v_x$", plot_coords=[1, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX2", r"$v_\theta$", plot_coords=[2, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX3", r"$v_\phi$", plot_coords=[3, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D(
        "BX1",
        r"BX1",
        plot_coords=[1, 1],
        streamlines=["BX1", "BX2"],
    ),
    MapMovie2D(
        "BX2",
        r"BX2",
        plot_coords=[2, 1],
        streamlines=["BX1", "BX2"],
    ),
    MapMovie2D(
        "BX3",
        r"BX3",
        plot_coords=[3, 1],
        streamlines=["BX1", "BX2"],
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
