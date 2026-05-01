from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/Idefix2Python/config/config.json"
task = "Drift_Tau"


custom_spaceTimeHeatmaps = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
    )
]

runContext = RunContext(task, projectPath, configPath=configPath)

pipeline = Pipeline(runContext, spaceTimeHeatmaps=custom_spaceTimeHeatmaps)

pipeline.run()
