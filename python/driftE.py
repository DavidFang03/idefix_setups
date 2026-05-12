from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap, PartQuantity
import utilities
from dotenv import load_dotenv
import os
from scipy.integrate import solve_ivp


load_dotenv()  # This loads variables from .env into os.environ
RUNS_FOLDER_PATH = os.getenv("RUNS_FOLDER_PATH")

projectPath = f"{RUNS_FOLDER_PATH}/ThomasDrift"
beta = 1e-2
task = "DriftL_2048_Size1e-2_custom"


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
            [0, 750],
            [r0],
            dense_output=True,
            # method="LSODA",
            method="DOP853",
        ).sol

    def __call__(self, t):
        return self.sol(t)[0, :]


sol = analytical_trajectory(beta)


custom_spaceTimeHeatmaps = []
custom_spaceTimeHeatmaps.append(
    SpaceTimeHeatmap(
        f"Dust{0}_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title=rf"$\beta \equiv s \rho^\mathrm{{part}} = {beta}$",
        cmap="inferno",
        norm="linear",
        ref_function=sol,
        uids="all",
    ),
)

SpaceTimeHeatmap.suptitle = "Particle trajectory over gas density"


def diff(v):
    return v.data["PART_X1"] - sol(v.t)


z_part = PartQuantity("diffx", r"errors", plot_coords=[0, 0], compute=diff)


runContext = RunContext(
    task,
    projectPath,
)
pipeline = Pipeline(
    runContext, spaceTimeHeatmaps=custom_spaceTimeHeatmaps, partQuantities=[z_part]
)

pipeline.run()
