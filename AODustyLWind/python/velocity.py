from idefix2python import RunContext, Pipeline, MapMovie2D, OneComponentOneVariable, Fig
import numpy as np
import matplotlib.pyplot as plt

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "clean_wind_100_v2_b1e4"
   
runContext = RunContext(task, projectPath, configPath=configPath,
 custom_name="Machs")

def zoom(x1, x2):
    return x1 < 5, np.ones_like(x2, dtype=bool)


def compute_mach_p(v):
    data=v.data
    cs2 = data["PRS"] / data["RHO"]
    vp2 = data["VX1"] ** 2 + data["VX2"] ** 2
    return np.sqrt(vp2 / cs2)

def compute_alfven_mach_p(v):
    data=v.data
    va2 = (data["BX1"]**2+data["BX2"]**2) / data["RHO"]
    vp2 = data["VX1"] ** 2 + data["VX2"] ** 2
    return np.sqrt(vp2 / va2)


quantities = [
    MapMovie2D(
        "Mach_p",
        r"$\mathcal{M}$",
        plot_coords=[0, 0],
        title="Mach Number (poloidal)",
        compute=compute_mach_p,
        contours=[1],
        contour_color="green",
    ),
    MapMovie2D(
        "Mach_A",
        r"$\mathcal{M}_\mathrm{A}$",
        plot_coords=[0, 1],
        title="Alfvén Mach number (poloidal)",
        compute=compute_alfven_mach_p,
        contours=[1],
        contour_color="green",
    ),
]

fig1 = Fig(quantities)


pipeline = Pipeline(
    runContext,
    [fig1],
    # zoom=zoom,
    # no_movie=True,
)

pipeline.run()
