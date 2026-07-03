from idefix2python import (
    RunContext,
    Pipeline,
    LineMovie1D,
    PartQuantity,
    SpaceTimeHeatmap,
    Fig,
)
import numpy as np
from scipy.integrate import solve_ivp

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/MyDustyWave"
task = "dwave_n10_v2"
iniPath = f"/home/dp316/dp316/dc-fang1/IdefixRuns/MyDustyWave/inputs/{task}.ini"

runContext = RunContext(
    task,
    projectPath,
    iniPath=iniPath,
    # pdf_mode=True,
)
n = int(runContext.inidata["Setup"]["n"])
print(n)
# n = 1

tau = 0.1
Omega = 1
S = 1.5
vs = 1

kx = 2 * np.pi * n / 1
kappa = np.sqrt(2 * Omega * (2 * Omega - S))
omega = np.sqrt(kappa**2 + (kx * vs) ** 2)
tmax = 20
perturbation_amplitude = 1e-2


def ref_density(v):
    return 1 + perturbation_amplitude * np.cos(kx * v.x - omega * v.t[0])


def f(t, Y):
    x, y, dotx, doty = Y
    tvx = np.abs(perturbation_amplitude * omega / kx)
    tvy = np.abs(perturbation_amplitude * (S - 2 * Omega) / kx)

    v_gas_y = -S * x + tvy * np.sin(omega * t + kx * x)

    ddotx = 2 * Omega * (doty + S * x) - 1 / tau * (
        dotx - tvx * np.cos(omega * t + kx * x)
    )
    ddoty = -2 * Omega * dotx - 1 / tau * (doty - v_gas_y)
    # ddoty = -2 * Omega * dotx - 1 / tau * (
    # doty + tvy * np.sin(omega * t + kx * x)
    # )
    return [dotx, doty, ddotx, ddoty]


sol = solve_ivp(
    f,
    [0, tmax],
    [0, 0, 0, 0],
    dense_output=True,
    # method="BDF",
    method="RK45",
    # method="DOP853",
    # max_step=perturbation_amplitude / kx,
    atol=1e-9,
    rtol=1e-9,
).sol


def predx(T):
    T = np.asarray(T)
    return sol(T)[0, :]


def diffx1(v):
    data = v.data
    return data["PART_X1"] - sol(v.t)[0]


def diffx1_relat(v):
    data = v.data
    return np.abs((data["PART_X1"] - sol(v.t)[0]) / sol(v.t)[0])


def labels(ax, vtk):
    ax.set_xlabel(r"$x$ [au]")
    ax.set_ylabel(r"$\rho/\rho_0$")


def ref_density(v):
    return 1 + perturbation_amplitude * np.cos(kx * v.x - omega * v.t[0])


def ref_vx1(v):
    return perturbation_amplitude * omega / kx * np.cos(kx * v.x - omega * v.t[0])


def ref_vx2(v):
    return -S * v.x - perturbation_amplitude * (2 * Omega - S) / kx * np.sin(
        kx * v.x - omega * v.t[0]
    )


custom_LineMovie1Ds = [
    # Density
    LineMovie1D(
        "RHO",
        r"$\rho$",
        title="Gas density",
        plot_coords=[0, 0],
    ),
    LineMovie1D(
        "RHO_ref",
        plot_coords=[0, 0],
        compute=ref_density,
        style_kwargs={"ls": "--", "lw": 1.0, "label": "Reference"},
        xlabel=r"$x$",
        ylabel=r"$\rho$",
    ),
    # Radial velocity
    LineMovie1D(
        "VX1",
        r"$v_x$",
        title="Radial velocity",
        plot_coords=[1, 0],
    ),
    LineMovie1D(
        "VX1_ref",
        plot_coords=[1, 0],
        compute=ref_vx1,
        style_kwargs={"ls": "--", "lw": 1.0, "label": "Reference"},
        xlabel=r"$x$",
        ylabel=r"$v_x$",
    ),
    # Azimuthal velocity
    LineMovie1D(
        "VX2",
        r"$v_y$",
        title="Azimuthal velocity",
        plot_coords=[2, 0],
    ),
    LineMovie1D(
        "VX2_ref",
        plot_coords=[2, 0],
        compute=ref_vx2,
        style_kwargs={"ls": "--", "lw": 1.0, "label": "Reference"},
        xlabel=r"$x$",
        ylabel=r"$v_y$",
    ),
]
pqs = [
    PartQuantity(
        "PART_X1",
        r"$x^\mathrm{part}$",
        title="Particle position",
        plot_coords=[0, 0],
        uids="all",
        ref_function=predx,
        alpha=0.3,
        xlabel="",
        xmax=1,
        xmin=0,
    ),
    PartQuantity(
        "PART_X1_diff_relat",
        r"Relative error",
        title="Relative error",
        plot_coords=[1, 0],
        uids="all",
        compute=diffx1_relat,
        yscale="log",
    ),
]


pipeline = Pipeline(
    runContext,
    figs=[
        Fig(custom_LineMovie1Ds, row_height=9),
        Fig(
            pqs,
            row_height=2,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1.2]},
        ),
    ],
)


pipeline.run()
