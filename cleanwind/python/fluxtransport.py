import os

import appl
import common
import numpy as np

RUNS_FOLDER_PATH = str(os.getenv("RUNS_FOLDER_PATH"))

project = "cleanwind"
runName = "mr_relax_b1e3"

runContext = appl.Context(
    runName=runName,
    projectPath=f"{RUNS_FOLDER_PATH}/{project}",
    code="idefix",
    custom_name="ft",
)

grid = runContext.grid
r = grid.X1_med
theta = grid.X2_med

zmax = 10
Z = np.linspace(-zmax, zmax, 500)


def vpsi(ax, ds, on, fn):
    x = 4.0
    ft = common.FluxTransport(grid, ds).at(x, Z)

    ax.plot(Z, ft.advection_term, label="advection")
    ax.plot(Z, ft.ohmic_term, label="ohmic")
    ax.plot(Z, ft.ambipolar_term, label="ambipolar")

    ax.legend()


quantities = [
    appl.CustomQuantity(
        "vB",
        r"$v_B$",
        plot_coords=[0, 0],
        customize=vpsi,
        xlabel="$z$",
    ),
]

fig1 = appl.Fig(quantities)

if __name__ == "__main__":
    pipeline = appl.Pipeline(runContext, [fig1])

    pipeline.run()
