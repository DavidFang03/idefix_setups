from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap
import utilitaries

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/Idefix2Python/config/config.json"
task = "Drift_Tau"


def analytical_trajectory(t):
    Stokes0 = 1
    fluid = utilitaries.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=Stokes0)
    r0 = 2
    return utilitaries.integrate(fluid.vrDrift, r0, t)


custom_spaceTimeHeatmaps = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
        ref_function=analytical_trajectory,
    )
]

runContext = RunContext(task, projectPath, configPath=configPath)
pipeline = Pipeline(runContext, spaceTimeHeatmaps=custom_spaceTimeHeatmaps)

pipeline.run()
