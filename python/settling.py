from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap
import utilities

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/VerticalSettling"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/Idefix2Python/config/config.json"
task = "Settling_Tau"


def analytical_trajectory(t):
    Stokes0 = 1
    z0 = 0.1
    fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=Stokes0, r0=2, z0=z0)
    return utilities.solve_2nd_order_ode(fluid.azSettling, z0, 0, t)


custom_spaceTimeHeatmaps = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
        norm="log",
        cmap="inferno",
        ref_function=analytical_trajectory,
    )
]

runContext = RunContext(task, projectPath, configPath=configPath)
pipeline = Pipeline(runContext, spaceTimeHeatmaps=custom_spaceTimeHeatmaps)

pipeline.run()
