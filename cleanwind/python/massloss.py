from idefix2python import RunContext, Pipeline, MapMovie2D, Fig
import numpy as np
from common import float_to_latex, plasmabeta
import matplotlib

matplotlib.use("pdf")

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
# task = "clean_wind_100_v2_b1e4"
task = "lr_wind_v2_b1e4"

runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
)

eps = float(runContext.inidata["Setup"]["epsilon"])
betamid = float(runContext.inidata["Setup"]["beta"])


def title(ax, v):
    fig = ax.get_figure()
    fig.suptitle(rf"$t={float_to_latex(v.t[0] / (2 * np.pi))}$ yr")


inferno = {"cmap": "inferno"}
quantities = [
    MapMovie2D(
        "RHO",
        r"\rho",
        title="Gas density",
        plot_coords=[0, 0],
        streamlines=["VX1", "VX2"],
        customize=title,
    ),
    MapMovie2D(
        "beta",
        r"$\beta$",
        title=r"Plasma $\beta$ (poloidal)",
        plot_coords=[0, 1],
        streamlines=["BX1", "BX2"],
        compute=plasmabeta,
        bounds=[1e-1, betamid],
        norm="log",
        contours=[1],
    ),
]
fig1 = Fig(quantities)

pipeline = Pipeline(runContext, [fig1])

pipeline.run()
