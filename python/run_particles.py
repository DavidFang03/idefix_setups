from idefix2python import RunContext, Pipeline, PartQuantity
import utilities

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift"
task = "DriftL_Tau"


def analytical_drift(t_array):
    Stokes0 = 1
    fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=Stokes0)
    r0 = 2
    return utilities.integrate(fluid.vrDrift, r0, t_array)


custom_partQuantities = [
    PartQuantity(
        "PART_X1", "PART_X1", plot_coords=[0, 0], ref_function=analytical_drift
    )
]

runContext = RunContext(
    task,
    projectPath,
    partFolder="/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/setup_l",
)
pipeline = Pipeline(
    runContext,
    partQuantities=custom_partQuantities,
)

pipeline.run()
