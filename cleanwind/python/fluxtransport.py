import os

import appl
import common
import numpy as np

RUNS_FOLDER_PATH = str(os.getenv("RUNS_FOLDER_PATH"))

project = "cleanwind"
runName = "mr_relax2_b1e3"

runContext = appl.Context(
    runName=runName,
    projectPath=f"{RUNS_FOLDER_PATH}/{project}",
    code="idefix",
    custom_name="ft",
)

epsilon = runContext.inidata["Setup"]["epsilon"]
grid = runContext.grid
r = grid.X1_med
theta = grid.X2_med

zmax = 3.0
Z = np.linspace(-zmax, zmax, 500)

x = 5.0


def vpsi(ax, ds, on, fn):
    ft = common.FluxTransport(grid, ds).at(x, Z)

    H = x * epsilon

    ax.plot(Z / H, ft.advection_term, label="Advection")
    ax.plot(Z / H, ft.ohmic_term, label="Ohmic")
    ax.plot(Z / H, ft.ambipolar_term, label="Ambipolar")
    ax.plot(Z / H, ft.total, label="Total", color="black", ls="--")
    ax.plot(Z / H, ft.from_idefix, label="from idefix", color="red", ls="--")

    ax.set_ylim(-0.01, 0.04)
    ax.set_xlim(-zmax / H, zmax / H)

    ax.set_ylabel(r"$v_\psi$")

    ax.legend()
    ax.grid()


quantities = [
    appl.CustomQuantity(
        "vB",
        r"$v_\psi$",
        title=f"Flux transport terms at $R={x}$ au",
        plot_coords=[0, 0],
        customize=vpsi,
        xlabel="$z/H$",
    ),
]

fig1 = appl.Fig(quantities)

if __name__ == "__main__":
    pipeline = appl.Pipeline(runContext, [fig1])

    pipeline.run()
