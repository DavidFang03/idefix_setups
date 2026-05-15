from idefix2python import RunContext, Pipeline, Fig, SpaceTimeHeatmap, PartQuantity
import utilities
from dotenv import load_dotenv
import os
from scipy.integrate import solve_ivp
import numpy as np

load_dotenv()  # This loads variables from .env into os.environ
RUNS_FOLDER_PATH = os.getenv("RUNS_FOLDER_PATH")

projectPath = f"{RUNS_FOLDER_PATH}/ThomasDrift"
task = "DriftL_Size_CFLg"

uids = [0, 1, 2]
betas = [1, 0.1, 0.01]


class analytical_trajectory:
    def __init__(self, beta):
        self.beta = beta
        self.plot_kwargs = {
            "ls": "--",
            "color": "white",
            "lw": 0.5,
            "alpha": 0.75,
            "label": "Predicted",
        }
        z0 = 0
        r0 = 2
        fluid = utilities.Fluid(
            cs0=0.05,
            csSlope=-0.5,
            sigma0=0.125,
            sigmaSlope=-0.5,
            beta=self.beta,
            r0=r0,
            z0=z0,
            drag="epstein",
        )
        self.sol = solve_ivp(
            fluid.vrDrift,
            [0, 800],
            [r0],
            dense_output=True,
            # method="RK45",
            atol=1e-15,
            rtol=1e-12,
            method="DOP853",
        ).sol

    def __call__(self, t):
        return self.sol(t)[0, :]


sols = [analytical_trajectory(beta) for beta in betas]


class DiffCompute:
    def __init__(self, sol_func):
        self.sol_func = sol_func

    def __call__(self, v):
        d = np.abs(v.data["PART_X1"] - self.sol_func(v.t))
        return d


class SolPred:
    def __init__(self, sol_func):
        self.sol_func = sol_func

    def __call__(self, v):
        return self.sol_func(v.t) + 0 * v.data["PART_X1"]


if __name__ == "__main__":
    qties = []
    for jj, beta in enumerate(betas):
        uid = [uids[jj]]
        sol = sols[jj]
        sp = SpaceTimeHeatmap(
            f"Dust{jj}_RHO",
            r"$\rho^\mathrm{dust}$",
            plot_coords=[0, jj],
            title=rf"$\beta \equiv s \rho^\mathrm{{part}} = {beta}$",
            cmap="inferno",
            norm="linear",
            ref_function=sol,
            uids=uid,
            parts_kwargs={"color": "blue"},
        )

        z_part = PartQuantity(
            f"diffx_{jj}",
            r"errors",
            plot_coords=[1, jj],
            compute=DiffCompute(sols[jj]),
            uids=uid,
            plot_kwargs={"color": "blue"},
        )

        x_zoom = PartQuantity(
            "PART_X1",
            r"PART_X1",
            plot_coords=[2, jj],
            uids=uid,
            plot_kwargs={"color": "blue"},
        )
        xpred_zoom = PartQuantity(
            f"pred_{jj}",
            f"pred_{jj}",
            plot_coords=[3, jj],
            compute=SolPred(sol),
            uids=uid,
            plot_kwargs={"color": "red"},
        )
        qties += [sp, z_part, x_zoom, xpred_zoom]

    fig0 = Fig(qties, suptitle="Particle trajectory")

    runContext = RunContext(
        task,
        projectPath,
    )
    pipeline = Pipeline(runContext, [fig0])

    pipeline.run()
