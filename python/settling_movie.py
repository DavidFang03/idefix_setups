from idefix2python import RunContext, Pipeline, LineMovie1D

configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/Idefix2Python/config/config.json"

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/VerticalSettling"
task = "Settling_Tau"

custom_LineMovie1Ds = [
    LineMovie1D(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
    )
]

runContext = RunContext(task, projectPath, configPath=configPath)
pipeline = Pipeline(runContext, movies1D=custom_LineMovie1Ds)

pipeline.run()
