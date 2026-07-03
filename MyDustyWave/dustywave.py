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
task = "dwave_n10"
iniPath = f"/home/dp316/dp316/dc-fang1/IdefixRuns/MyDustyWave/inputs/{task}.ini"

runContext = RunContext(task, projectPath,iniPath=iniPath)
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

def ref_density(v):
    return 1+ 1e-3 * np.cos(kx * v.x + omega*v.t[0])



def f(t, Y):
    x, y, dotx, doty = Y
    tvx = np.abs(1e-3 * omega / kx)
    tvy = np.abs(1e-3 * (S - 2 * Omega) / kx)

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
    method="BDF",
    # method="DOP853",
    # max_step=1e-3 / kx,
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


# print(predx(np.linspace(1, 10)))


def vzdiff(v):
    print(np.shape(v.data["VX3"]))
    print(np.shape(v.x))
    return v.data["VX3"] + S * v.x


# custom_LineMovie1Ds = [
#     SpaceTimeHeatmap("RHO", r"$\rho$", title="Density", plot_coords=[0, 0]),
#     SpaceTimeHeatmap("VX1", r"$v_x$", plot_coords=[1, 0], uids="all"),
#     SpaceTimeHeatmap("VX2", r"$v_y$", plot_coords=[2, 0]),
#     SpaceTimeHeatmap("VX3", r"$v_z$", plot_coords=[3, 0]),
# ]
custom_LineMovie1Ds = [
    LineMovie1D("RHO", r"$\rho$", title="Density", plot_coords=[0, 0]),
    LineMovie1D("RHO_ref", plot_coords=[0, 0], compute=ref_density, style_kwargs={"alpha":1, "lw":0.5, "ls":"--", "label":"Reference"}),
    LineMovie1D("VX1", r"$v_x$", plot_coords=[1, 0], uids="all"),
    LineMovie1D("VX2", r"$v_y$", plot_coords=[2, 0]),
    LineMovie1D("VX3", r"$v_z$", plot_coords=[3, 0]),
]

pqs = [
    PartQuantity(
        "PART_X1",
        r"$x^\mathrm{part}$",
        plot_coords=[0, 0],
        uids="all",
        ref_function=predx,
        alpha=0.3,
    ),
    # PartQuantity(
    #     "PART_X1_diff",
    #     r"$x^\mathrm{part}$",
    #     plot_coords=[0, 0],
    #     uids="all",
    #     compute=diffx1,
    # ),
    PartQuantity(
        "PART_X1_diff_relat",
        r"Relative error",
        title="Relative error",
        plot_coords=[1, 0],
        uids="all",
        compute=diffx1_relat,
        yscale="log"
    ),
]

sps = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        uids="all",
        compute=diffx1,
        ymin=-0.002,
        ymax=0.002,
    )
]

# fig = Fig(custom_LineMovie1Ds+pqs)

# pipeline = Pipeline(runContext, partQuantities=pqs, movies1D=custom_LineMovie1Ds)
pipeline = Pipeline(runContext, figs=[Fig(custom_LineMovie1Ds),Fig(pqs)])
# pipeline = Pipeline(runContext, spaceTimeHeatmaps=sps, partQuantities=pqs)

pipeline.run()
