from idefix2python import RunContext, Pipeline, MapMovie2D, PartQuantity, Fig
import numpy as np

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/Idefix2Python/config/config.json"
task = "r_Epstein_wind_1e-4"


def z(v):
    r = v.data["PART_X1"]
    theta = v.data["PART_X2"]
    return r * np.cos(theta)


def St(v):
    r = v.x
    return v.data["TSTOP"] * r ** (-1.5)


uids = "all"

quantities = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
        uids=uids,
    ),
    # MapMovie2D(
    #     "VX1", r"$v_x$", plot_coords=[1, 0], streamlines=["VX1", "VX2"], uids=uids
    # ),
    MapMovie2D(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[1, 0],
        streamlines=["VX1", "VX2"],
        # uids=uids,
    ),
    # PartQuantity(
    #     "PART_X1",
    #     "PART_X1",
    #     uids=uids,
    #     # uids=uids,
    #     plot_coords=[0, 1],
    # ),
    # PartQuantity(
    #     "PART_X2",
    #     "PART_X2",
    #     uids=uids,
    #     plot_coords=[1, 1],
    # ),
    PartQuantity("St", r"St", uids=uids, plot_coords=[0, 1], compute=St, vmax=2),
    # MapMovie2D("Dust0_RHO", r"$\rho^d$", plot_coords=[0, 1], uids=uids),
    # MapMovie2D("Dust0_VX1", r"$v_r^d$", plot_coords=[1, 1], uids=uids),
    # MapMovie2D("Dust0_VX2", r"$v_\theta^d$", plot_coords=[2, 1], uids=uids),
    # MapMovie2D("Dust0_VX3", r"$v_\phi^d$", plot_coords=[3, 1], uids=uids),
    # MapMovie2D("InvDt", r"$dt^{-1}$", plot_coords=[2, 0], uids=uids),
]
fig1 = Fig(quantities)
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context
runContext = RunContext(task, projectPath, configPath=configPath)

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

pipeline.run()
