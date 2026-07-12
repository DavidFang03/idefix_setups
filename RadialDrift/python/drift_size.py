from idefix2python import (
    RunContext,
    Pipeline,
    Fig,
    SpaceTimeHeatmap,
    OneComponentOneVariable,
)
from scipy.integrate import solve_ivp
import utilities
import numpy as np
import matplotlib.pyplot as plt

projectPath = "/home/dfang/Code/idefix_setups/RadialDrift"
task = "Drift_Size_clean"

runContext = RunContext(task, projectPath, pdf_mode=False,)

beta = runContext.outputTypes_info["particles"].testData["DRAGCOEFF"][0]
print(beta)

rho0 = 6.0e-10
rhos = 1.0
au = 1.5e11
size = beta * (rho0 * au) / rhos


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
        # self.sol = solve_ivp(
        #     fluid.vrDrift,
        #     [0, 750],
        #     [r0],
        #     dense_output=True,
        #     # method="LSODA",
        #     method="DOP853",
        # ).sol
        self.sol = solve_ivp(
            fluid.drift_system,
            [0, 750],
            [r0, 0],
            dense_output=True,
            # method="LSODA",
            method="DOP853",
        ).sol

    def __call__(self, t):
        return self.sol(t)[0, :]


atraj = analytical_trajectory(beta)
atraj.plot_kwargs = {"lw": 2, "color": "darkorange", "alpha": 1}


def diffdust(v):
    rdust = v.x[np.argmax(v.data["Dust0_RHO"])]
    dustrho = np.clip(v.data["Dust0_RHO"], a_min=1e-5, a_max=None) - 1e-5
    rdust = np.sum(v.x * dustrho) / np.sum(dustrho)

    return (np.abs(rdust - atraj(v.t))) / atraj(v.t)


def diffpart(v):
    return (np.abs(v.data["PART_X1"] - atraj(v.t))) / atraj(v.t)

def St(v):
    return v.data["TSTOP"]*v.data["PART_X1"]**(-1.5)

def ylim(ax, v):
    ax.set_ylim(1e-4, 1e-1)
    ax.set_ylabel(
        r"$r-r_\mathrm{pred}$",
    )
    ax.set_xlabel(
        r"$t$",
    )


def label_func(uid, vtk):
    return "Particle"


quantities = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        title="",
        plot_coords=[0, 0],
        uids="all",
        ref_function=atraj,
        style_kwargs={"cmap": "inferno"},
        norm="log",
        bounds=[1e-4, 1e-2],
        xlabel="",
        ylabel="$r$ [au]",
        label_func=label_func,
        parts_kwargs={"lw": 3, "color": "aqua"},
    ),
    OneComponentOneVariable(
        "diffdust",
        # r"$\rho^\mathrm{dust}$",
        # title=rf"$s={size}\,\mathrm{{m}}$",
        plot_coords=[1, 0],
        compute=diffdust,
        customize=ylim,
        style_kwargs={"label": "Pressureless fluid", "color": "magenta"},
    ),
    OneComponentOneVariable(
        "diffpart",
        # r"$r-r_\mathrm{pred}$",
        # title=rf"$s={size}\,\mathrm{{m}}$",
        plot_coords=[1, 0],
        compute=diffpart,
        yscale="log",
        ylabel=r"Relative error",
        xlabel=r"$t$ [yr]",
        style_kwargs={"label": "Particle", "color": "aqua"},
    ),
    OneComponentOneVariable(
        "St",
        # r"$r-r_\mathrm{pred}$",
        # title=rf"$s={size}\,\mathrm{{m}}$",
        compute=St,
        plot_coords=[2, 0],
        yscale="log",
        ylabel=r"St",
        xlabel=r"$t$ [yr]",
        style_kwargs={"label": "Particle", "color": "aqua"},
    ),
]

default_text_color = plt.rcParams['text.color']

fig0 = Fig(
    quantities,
    suptitle=rf"$s={size}\,\mathrm{{m}}$",
        suptitle_kwargs={"x":0,
                "y":1.15,
                "bbox":dict(facecolor="none", edgecolor="white"),
                "ha":"left"},
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1.2, 1.2]},
)


if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0], no_movie=True)
    pipeline.run()
