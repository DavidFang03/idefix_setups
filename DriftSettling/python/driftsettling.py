from idefix2python import (
    RunContext,
    Pipeline,
    Fig,
    PartQuantity,
    SpaceTimeHeatmap,
    MapMovie2D,
    LineMovie1D,
)
import utilities
from pathlib import Path
import numpy as np

projectPath = "/home/dfang/Code/idefix_setups/DriftSettling"
task = "Drift_Tau"
uids = "all"
uids = None
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/
# In this example, the vtks/ folder contains both part*.vtk and data*.vtk


# def analytical_trajectory(t):
#     z0 = 0.1
#     fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=1, z0=z0)
#     return utilities.solve_2nd_order_ode(fluid.azSettling, z0, 0, t)


class analytical_trajectory:
    def __init__(self, tau):
        self.tau = tau
        self.plot_kwargs = {
            "ls": "--",
            "color": "white",
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


analytical_trajectory.plot_kwargs = {"ls": "--", "color": "cyan", "lw": 2}

# print(analytical_trajectory(0))


def dv(v):
    dx = v.r[1] - v.r[0]
    rpart = v.data["PART_X1"]
    if np.isnan(rpart):
        rpart = [2]
    print(v.theta)
    print(rpart)
    ii = int(rpart / dx)
    drift = v.data["PART_VX1"] - [v.data["VX1"][ii]]
    return drift


# quantities = [
#     # PartQuantity(
#     #     "PART_X1",
#     #     r"$r^\mathrm{part}$",
#     #     plot_coords=[0, 0],
#     #     style_kwargs={"lw": 2, "color": "black"},
#     # ),
#     SpaceTimeHeatmap(
#         "Dust0_RHO",
#         r"$\rho^\mathrm{dust}$",
#         plot_coords=[0, 0],
#         # uids="all",
#         ref_function=analytical_trajectory,
#     ),
#     # SpaceTimeHeatmap(
#     #     "VX1",
#     #     r"$v_r$",
#     #     plot_coords=[0, 1],
#     #     uids="all",
#     #     ref_function=analytical_trajectory,
#     # ),
#     LineMovie1D(
#         "VX1",
#         r"$v_r$",
#         plot_coords=[0, 1],
#         # uids="all",
#         bounds=[-1e-4, 1e-4],
#     ),
#     LineMovie1D(
#         "VX2",
#         r"$v_\theta$",
#         plot_coords=[0, 2],
#         # uids="all",
#         # bounds=[-1e-4, 1e-4],
#     ),
#     LineMovie1D(
#         "VX3",
#         r"$v_\phi$",
#         plot_coords=[0, 3],
#         # uids="all",
#         # bounds=[-1e-4, 1e-4],
#     ),
#     LineMovie1D(
#         "RHO",
#         r"$\rho$",
#         plot_coords=[0, 4],
#         # uids="all",
#     ),
#     # LineMovie1D(
#     #     "cs",
#     #     r"$\rho$",
#     #     plot_coords=[0, 2],
#     #     uids="all",
#     # ),
#     # PartQuantity("dv", "dv", plot_coords=[1,0], compute=dv)
# ]

quantities = [
    # PartQuantity(
    #     "PART_X1",
    #     r"$r^\mathrm{part}$",
    #     plot_coords=[1, 1],
    #     style_kwargs={"lw": 2, "color": "black"},
    #     ref_function=analytical_trajectory(1),
    # ),
    MapMovie2D(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        # uids="all",
    ),
    MapMovie2D(
        "Dust0_VX1",
        r"$v_r^\mathrm{dust}$",
        plot_coords=[1, 0],
        # uids="all",
    ),
    MapMovie2D(
        "Dust0_VX2",
        r"$v_\theta^\mathrm{dust}$",
        plot_coords=[2, 0],
        # uids="all",
    ),
    MapMovie2D(
        "Dust0_VX3",
        r"$v_\phi^\mathrm{dust}$",
        plot_coords=[3, 0],
        # uids="all",
    ),
    # SpaceTimeHeatmap(
    #     "VX1",
    #     r"$v_r$",
    #     plot_coords=[0, 1],
    #     uids="all",
    #     ref_function=analytical_trajectory,
    # ),
    MapMovie2D(
        "VX1",
        r"$v_r$",
        plot_coords=[0, 1],
        uids=uids,
        # uids="all",
        # bounds=[-1e-4, 1e-4],
    ),
    MapMovie2D(
        "VX2",
        r"$v_\theta$",
        plot_coords=[0, 2],
        # uids="all",
        # bounds=[-1e-4, 1e-4],
    ),
    MapMovie2D(
        "VX3",
        r"$v_\phi$",
        plot_coords=[0, 3],
        # uids="all",
        # bounds=[0, 2],
    ),
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 4],
        # uids="all",
        # bounds=[0, 0.1],
    ),
    MapMovie2D(
        "cs",
        r"$c_s$",
        plot_coords=[0, 5],
    ),
    # PartQuantity("dv", "dv", plot_coords=[1,0], compute=dv)
]

fig0 = Fig(
    quantities,
    suptitle="Dust density: pressureless fluid, particles, and an analytical trajectory",
)

runContext = RunContext(
    task,
    projectPath,
)

if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0])
    pipeline.run()
