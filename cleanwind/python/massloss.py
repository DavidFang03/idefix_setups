from idefix2python import (
    RunContext,
    Pipeline,
    MapMovie2D,
    Fig,
    OneComponentOneVariable,
    SpaceTimeHeatmap,
    LineMovie1D,
)
import numpy as np
from common import float_to_latex, plasmabeta, WindyDisk
import matplotlib.pyplot as plt

discrete_cmap = plt.get_cmap("turbo_r").resampled(30)

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
# task = "clean_wind_100_v2_b1e4"
task = "lr_wind_v7_b1e4"


runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    custom_name="vB",
)

wd = WindyDisk(runContext.inidata, runContext.gridInfo)


eps = float(runContext.inidata["Setup"]["epsilon"])
betamid = float(runContext.inidata["Setup"]["beta"])

RLine = runContext.gridInfo.X1Line
Rgrid = runContext.gridInfo.grid1


def title(ax, v):
    fig = ax.get_figure()
    fig.suptitle(rf"$t={float_to_latex(v.t[0] / (2 * np.pi))}$ yr")


def totaddmass(v):
    dr, dtheta = np.meshgrid(np.diff(v.rl), np.diff(v.thetal))
    return np.sum(v.data["addedMass"] * Rgrid * dr * dtheta)


def grid(ax, v):
    ax.grid()


inferno = {"cmap": "inferno"}
quantities = [
    MapMovie2D(
        "RHO",
        r"\rho",
        title="Gas density",
        plot_coords=[0, 0],
        streamlines=["VX1", "VX2"],
        customize=title,
        # xmax=10,
        # ymin=-10,
        # ymax=10,
    ),
    MapMovie2D(
        "beta",
        r"$\beta$",
        title=r"Plasma $\beta$ (poloidal)",
        plot_coords=[0, 1],
        streamlines=["BX1", "BX2"],
        compute=plasmabeta,
        bounds=[1e-1, betamid],
        norm="log",
        contours=[1],
    ),
    MapMovie2D(
        "addedMass",
        r"$m_+$",
        title=r"Added mass",
        plot_coords=[2, 0],
        norm="log",
    ),
    OneComponentOneVariable(
        "TotalAddedMass",
        r"$M_+$",
        plot_coords=[2, 1],
        compute=totaddmass,
        yscale="log",
    ),
    OneComponentOneVariable(
        "totalmass",
        r"$M_\mathrm{disk}$",
        plot_coords=[2, 2],
        compute=wd.totalmass,
        yscale="log",
    ),
    OneComponentOneVariable(
        "Macc",
        r"$\dot M_\mathrm{acc}$",
        plot_coords=[1, 0],
        compute=wd.Macc,
        # yscale="log",
        ymin=-3e-4,
        ymax=3e-4,
    ),
    LineMovie1D(
        "zetap",
        r"$\zeta_+$",
        plot_coords=[0, 2],
        compute=wd.massloss_up,
        customize=grid,
        xlabel="$R$",
        yscale="log",
    ),
    LineMovie1D(
        "Ephi0",
        r"$E_\phi(R, \pi/2)$",
        plot_coords=[1, 2],
        compute=wd.Ephi_midplane,
        xlabel="$R$",
    ),
    LineMovie1D(
        "vB",
        r"$v_B$",
        plot_coords=[1, 3],
        compute=wd.vB,
        xlabel="$R$",
    ),
    LineMovie1D(
        "Bz0",
        r"$B_z(R,\pi/2)$",
        plot_coords=[0, 3],
        compute=wd.bz_midplane,
        xlabel="$R$",
        yscale="log",
    ),
    LineMovie1D(
        "mid_beta",
        r"$\beta_\mathrm{mid}$",
        plot_coords=[2, 3],
        compute=wd.midplane_beta,
        customize=grid,
        xlabel="$R$",
        yscale="log",
    ),
    SpaceTimeHeatmap(
        "Psimid",
        r"$\Psi_\mathrm{mid}(R,t)$",
        plot_coords=[1, 1],
        compute=wd.flux_function,
        # xqty=wd.RLine,
        rotate=True,
        # compute=wd.flux_function,
        # bounds=[0, 0.005],
        style_kwargs={"cmap": discrete_cmap},
        # xmax=10,
    ),
]
fig1 = Fig(quantities)


pipeline = Pipeline(runContext, [fig1])

pipeline.run()
