from idefix2python import (
    RunContext,
    Pipeline,
    Fig,
    PartQuantity,
    SpaceTimeHeatmap,
    MapMovie2D,
    LineMovie1D,
    LocalQuantity,
)
import utilities
from pathlib import Path
import numpy as np

projectPath = "/home/dfang/Code/idefix_setups/VerticalSettling"
task = "Settling_Size_clean"
uids = None
# uids = "all"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/
# In this example, the vtks/ folder contains both part*.vtk and data*.vtk


# def analytical_trajectory(t):
#     z0 = 0.1
#     fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=1, z0=z0)
#     return utilities.solve_2nd_order_ode(fluid.azSettling, z0, 0, t)
runContext = RunContext(task, projectPath, show_ini=True)


class analytical_trajectory:
    def __init__(self, tau):
        self.tau = tau
        self.plot_kwargs = {
            "ls": "--",
            "color": "lightgreen",
            "lw": 0.5,
            "alpha": 0.75,
            "label": "Predicted",
        }

    def __call__(self, t):
        tau = self.tau
        z0 = 0.0
        r0 = 2
        fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=tau, z0=z0)
        return utilities.integrate(fluid.vrDrift, r0, t)


RLine = runContext.gridInfo.X1Line


def drag_force(v):
    return -(v.data["PART_VX1"] - v.data["vr_local"]) / v.data["TSTOP"]


def grav_force(v):
    return -1 / v.data["PART_X1"] ** 2


def centrifugal_force(v):
    return v.data["PART_VX2"] ** 2 / v.data["PART_X1"]


def total_force(v):
    return drag_force(v) + grav_force(v) + centrifugal_force(v)


def vphi_mb(v):
    R = v.x
    Omega = np.pow(R, -1.5)
    return v.data["VX2"] - R * Omega


def prs(v):
    return v.data["cs"] * v.data["RHO"]


def cs(v):
    return np.sqrt(v.data["PRS"] / v.data["RHO"])


def ylim(ax, v):
    ax.set_ylim(0, 1e-1)


quantities = [
    LineMovie1D(
        "VX1",
        r"$v_r$",
        plot_coords=[0, 1],
        uids=uids,
    ),
    LineMovie1D(
        "VX2",
        r"$v_\phi$",
        plot_coords=[0, 2],
        uids="all",
        # bounds=[-1e-4, 1e-4],
    ),
    LineMovie1D(
        "VX2_mb",
        r"$v_\phi- v_\mathrm{K}$",
        plot_coords=[2, 2],
        compute=vphi_mb,
        uids="all",
    ),
    LineMovie1D(
        "VX3",
        r"$v_z$",
        plot_coords=[0, 3],
        # uids="all",
        # bounds=[-1e-4, 1e-4],
    ),
    LineMovie1D("RHO", r"$\rho$", plot_coords=[0, 4], uids="all", customize=ylim),
    LineMovie1D(
        "cs",
        r"$c_s$",
        plot_coords=[0, 5],
        uids=uids,
    ),
    # LineMovie1D(
    #     "PRS",
    #     r"$P$",
    #     title="Pressure (zoomed)",
    #     plot_coords=[0, 6],
    #     uids=uids,
    #     bounds=[1e-3, 5e-3],
    # ),
]

# quantities = [
#     PartQuantity(
#         "PART_X1",
#         r"$r^\mathrm{part}$",
#         plot_coords=[1, 1],
#         style_kwargs={"lw": 2, "color": "black"},
#         ref_function=analytical_trajectory(1),
#     ),
#     # MapMovie2D(
#     #     "Dust0_RHO",
#     #     r"$\rho^\mathrm{dust}$",
#     #     plot_coords=[0, 0],
#     #     # uids="all",
#     # ),
#     # MapMovie2D(
#     #     "VX1",
#     #     r"$v_r$",
#     #     plot_coords=[0, 1],
#     #     uids=uids,
#     #     # uids="all",
#     #     # bounds=[-1e-4, 1e-4],
#     # ),
#     # MapMovie2D(
#     #     "VX2",
#     #     r"$v_\theta$",
#     #     plot_coords=[0, 2],
#     #     # uids="all",
#     #     # bounds=[-1e-4, 1e-4],
#     # ),
#     # MapMovie2D(
#     #     "VX3",
#     #     r"$v_\phi$",
#     #     plot_coords=[0, 3],
#     #     # uids="all",
#     #     # bounds=[0, 2],
#     # ),
#     # MapMovie2D(
#     #     "RHO",
#     #     r"$\rho$",
#     #     plot_coords=[0, 4],
#     #     # uids="all",
#     #     # bounds=[0, 0.1],
#     # ),
#     # MapMovie2D(
#     #     "cs",
#     #     r"$c_s$",
#     #     plot_coords=[0, 5],
#     # ),
# ]

fig0 = Fig(
    quantities,
    suptitle="Dust density: pressureless fluid, particles, and an analytical trajectory",
)


if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0], no_movie=True)
    pipeline.run()
