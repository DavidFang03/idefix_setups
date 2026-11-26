import matplotlib.pyplot as plt
import numpy as np
from appl import (
    Context,
    Fig,
    LineMovie1D,
    MapMovie2D,
    Pipeline,
    SpaceTimeHeatmap,
)
from common import WindyDisk, float_to_latex, plasmabeta

discrete_cmap = plt.get_cmap("turbo_r").resampled(40)

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
task = "mr_relax_b1e3"


runContext = Context(
    task,
    projectPath,
    custom_name="vB",
)

wd = WindyDisk(runContext.inidata, runContext.grid)


beta0 = float(runContext.inidata["Setup"]["beta"])
eps = float(runContext.inidata["Setup"]["epsilon"])
betamid = float(runContext.inidata["Setup"]["beta"])
densityFloor = float(runContext.inidata["Setup"]["densityFloor"])

RLine = runContext.grid.X1


def title(ax, v, nb):
    fig = ax.get_figure()
    fig.suptitle(rf"$t={float_to_latex(runContext.times_all[nb] / (2 * np.pi))}$ yr")


# def totaddmass(v):
#     dr, dtheta = np.meshgrid(np.diff(v.rl), np.diff(v.thetal))
#     return np.sum(v.data["addedMass"] * Rgrid * dr * dtheta)


def grid(ax, v, on):
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
        bounds=[densityFloor, 1],
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
        bounds=[1, betamid],
        norm="log",
        contours=[1],
    ),
    # MapMovie2D(
    #     "addedMass",
    #     r"$m_+$",
    #     title=r"Added mass",
    #     plot_coords=[2, 0],
    #     norm="log",
    # ),
    # OneComponentOneVariable(
    #     "TotalAddedMass",
    #     r"$M_+$",
    #     plot_coords=[2, 1],
    #     compute=totaddmass,
    #     yscale="log",
    # ),
    # OneComponentOneVariable(
    #     "totalmass",
    #     r"$M_\mathrm{disk}$",
    #     plot_coords=[2, 2],
    #     compute=wd.totalmass,
    #     yscale="log",
    # ),
    # OneComponentOneVariable(
    #     "Macc",
    #     r"$\dot M_\mathrm{acc}$",
    #     plot_coords=[1, 0],
    #     compute=wd.Macc,
    #     # yscale="log",
    #     ymin=-3e-4,
    #     ymax=3e-4,
    # ),
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

xmax = 10
quantities = [
    MapMovie2D(
        "BX1",
        plot_coords=[1, 1],
        bounds=[-1e-4, 1e-4],
        xmax=20,
        ymin=-20,
        ymax=20,
    ),
    MapMovie2D(
        "BX2",
        plot_coords=[1, 2],
        bounds=[-1e-4, 1e-4],
        xmax=20,
        ymin=-20,
        ymax=20,
    ),
    MapMovie2D(
        "BX3",
        plot_coords=[1, 3],
        bounds=[-1e-4, 1e-4],
        xmax=20,
        ymin=-20,
        ymax=20,
    ),
    # MapMovie2D(
    #     "RHO",
    #     r"$\rho$",
    #     title="Gas density",
    #     plot_coords=[0, 0],
    #     streamlines=["VX1", "VX2"],
    #     customize=title,
    #     bounds=[densityFloor, 1e-4],
    #     style_kwargs={"cmap": "inferno"},
    #     norm="log",
    #     # xmax=10,
    #     # ymin=-10,
    #     # ymax=10,
    # ),
    # MapMovie2D(
    #     "beta",
    #     r"$\beta$",
    #     title=r"Plasma $\beta$ (poloidal)",
    #     plot_coords=[0, 1],
    #     streamlines=["BX1", "BX2"],
    #     compute=plasmabeta,
    #     bounds=[1, betamid],
    #     norm="log",
    #     contours=[1],
    # ),
    # LineMovie1D(
    #     "zetap",
    #     r"$\zeta_+$",
    #     title="Wind mass loss (up)",
    #     plot_coords=[0, 2],
    #     compute=wd.massloss_up,
    #     customize=grid,
    #     xlabel="$R$",
    #     yscale="log",
    #     ymin=1e-7,
    #     ymax=1e-4,
    # ),
    SpaceTimeHeatmap(
        "Psimid",
        r"$\Psi_\mathrm{mid}(R,t)$",
        title="Magnetic flux",
        plot_coords=[0, 0],
        compute=wd.flux_function,
        # xqty=wd.RLine,
        xmax=xmax,
        # norm="log",
        rotate=True,
        bounds=[5e-4, 1e-2 * 1e4 / beta0 * np.pow(xmax / 10, 0.5)],
        style_kwargs={"cmap": discrete_cmap},
        # yscale="log",
        # xmax=10,
    ),
    LineMovie1D(
        "vB",
        r"$v_B$",
        plot_coords=[1, 0],
        compute=wd.vB,
        xlabel="$R$",
        ymin=-0.001,
        ymax=0.003,
    ),
    # LineMovie1D(
    #     "Bz0",
    #     r"$B_z(R,\pi/2)$",
    #     title="Midplane $B_z$",
    #     plot_coords=[0, 3],
    #     compute=wd.bz_midplane,
    #     xlabel="$R$",
    #     yscale="log",
    #     ymin=1e-6,
    #     ymax=1e-3,
    # ),
]

fig1 = Fig(quantities)

if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig1])

    pipeline.run()
