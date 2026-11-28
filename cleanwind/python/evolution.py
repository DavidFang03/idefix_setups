import os
import numpy as np

import appl

RUNS_FOLDER_PATH = str(os.getenv("RUNS_FOLDER_PATH"))

project = "cleanwind"
runName = "hr2_b1e4"

runContext = appl.Context(
    runName=runName,
    projectPath=f"{RUNS_FOLDER_PATH}/{project}",
    code="idefix",
)


def plasmabeta(v):
    P = v["PRS"].data
    B2 = v["BX1"].data ** 2 + v["BX2"].data ** 2 + v["BX3"].data ** 2
    return 8 * np.pi * P / B2


streamlines = ["VX1", "VX2"]
inferno = {"cmap": "inferno"}
quantities = [
    appl.MapMovie2D(
        "RHO",
        r"$\rho$",
        title="Gas density",
        plot_coords=[0, 0],
        bounds=[1e-8, 1e-3],
        norm="log",
        # streamlines=streamlines,
        style_kwargs={"cmap": "inferno"},
    ),
    # appl.MapMovie2D(
    #     "T",
    #     r"$T$",
    #     title=r"Temperature",
    #     plot_coords=[0, 1],
    #     streamlines=streamlines,
    #     compute=wd.temperature,
    #     # bounds=[1e-4, 1e-4],
    #     norm="log",
    # ),
    appl.MapMovie2D(
        "CS",
        r"$c_s$",
        title=r"Sound speed",
        plot_coords=[0, 1],
        streamlines=streamlines,
        # bounds=[1e-4, 1e-4],
        norm="log",
    ),
    appl.MapMovie2D(
        "VX1",
        r"VX1",
        plot_coords=[0, 2],
        streamlines=streamlines,
        bounds=[-1e-2, 1e-2],
        style_kwargs={"cmap": "coolwarm"},
        # xmin=75,
        # xmax=10,
        # ymin=-10,
        # ymax=10,
    ),
    appl.MapMovie2D(
        "VX2",
        r"$VX2$",
        plot_coords=[0, 3],
        streamlines=streamlines,
        style_kwargs={"cmap": "coolwarm"},
        bounds=[-1e-2, 1e-2],
    ),
    appl.MapMovie2D(
        "VX3",
        r"VX3",
        plot_coords=[0, 4],
        streamlines=streamlines,
        bounds=[1e-1, 1],
    ),
    appl.MapMovie2D(
        "EPHI",
        r"$E_\phi$",
        plot_coords=[1, 0],
        streamlines=streamlines,
    ),
    appl.MapMovie2D(
        "beta",
        r"$\beta$",
        plot_coords=[1, 1],
        # streamlines=streamlines,
        compute=plasmabeta,
    ),
    appl.MapMovie2D(
        "INVDT",
        r"InvDt",
        title="CFL",
        plot_coords=[0, 5],
        norm="log",
    ),
]

fig1 = appl.Fig(quantities)


# Initialize context

# Inject ONLY the 2D fields into the pipeline
pipeline = appl.Pipeline(runContext, [fig1])

if __name__ == "__main__":
    pipeline.run()
