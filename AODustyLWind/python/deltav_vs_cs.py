from idefix2python import (
    RunContext,
    Pipeline,
    MapMovie2D,
    OneComponentOneVariable,
    PartQuantity,
    LocalQuantity,
    Fig,
)
import numpy as np

from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "dw100_v2_intro_b1e4_2000p"

runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    dataFolder=f"{projectPath}/outputs_v2/{task}/vtks",
    iniPath=f"{projectPath}/inputs_v2/{task}.ini",
    custom_name="supersonic_drift",
    # pdf_mode=True,
)


parts_cmap = plt.get_cmap("cool")


num_r = int(runContext.inidata["Particles"]["num_r"][0])
num_theta = int(runContext.inidata["Particles"]["num_theta"][0])
num_size = int(runContext.inidata["Particles"]["num_size"][0])


uids_grid = np.arange(num_r * num_theta * num_size).reshape(num_r, num_theta, num_size)
print(uids_grid.shape)

same_r = uids_grid[-2, :, :].flatten()
same_angle = uids_grid[:, 3, :].flatten()
same_size = uids_grid[:, :, 10:20].flatten()

same_pos = [uid for uid in same_r if uid in same_angle]
uids = list(same_pos)


def float_to_latex(num: float) -> str:
    """
    Converts a float (including scientific notation like 4e3)
    into a LaTeX formatted string: $4 \cdot 10^3$.
    """
    if num == 0:
        return "0"

    # Get the base-10 exponent and the mantissa
    exponent = int(np.floor(np.log10(abs(num))))
    mantissa = num / (10**exponent)

    # Clean up trailing zeros or convert float integers (like 4.0 to 4)
    mantissa = int(mantissa) if mantissa.is_integer() else round(mantissa, 4)

    # If the mantissa is exactly 1, we usually just write 10^x instead of 1 \cdot 10^x
    if mantissa == 1:
        return f"10^{{{exponent}}}"
    if mantissa == -1:
        return f"-10^{{{exponent}}}"

    # Format as a LaTeX inline np string
    return f"{mantissa:.2} \\cdot 10^{{{exponent}}}"


cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_scientific("%.2e")
cbformat.set_powerlimits((-2, 12))
cbformat.set_useMathText(True)


def colorbar(mappable, loc):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size="4%", pad="25%")
    cbar = fig.colorbar(mappable, cax=cax, location=loc, pad="-100%")
    plt.sca(last_axes)
    return cbar


def cb(ax, vtk):
    ax.axhline(y=1, color=plt.rcParams["text.color"], lw=1, ls="--")
    sizes = size(vtk)

    # 1. Define the same LogNorm used for the coloring
    norm = LogNorm(vmin=np.min(sizes), vmax=np.max(sizes))

    mappable = ScalarMappable(norm=norm, cmap=parts_cmap)
    mappable.axes = ax
    # fig = ax.figure
    # cbar = fig.colorbar(mappable, pad=0.05, location="left")
    cbar = colorbar(mappable, loc="right")
    cbar.ax.set_title("Dust size [m]", pad=15)

    # cbar.ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    # cbar.ax.yaxis.set_ticks_position("right")
    # cbar.ax.yaxis.set_label_position("right")

    return cbar


def size(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def colors(vtk):
    sizes = size(vtk)
    norm = LogNorm(vmin=np.min(sizes), vmax=np.max(sizes))

    return parts_cmap(norm(sizes))


def delta_vr(v):
    return np.abs(v.data["PART_VX1"] - v.data["vr_local"]) / v.data["cs_local"]


def delta_vtheta(v):
    return np.abs(v.data["PART_VX2"] - v.data["vtheta_local"]) / v.data["cs_local"]


def delta_vphi(v):
    return np.abs(v.data["PART_VX3"] - v.data["vphi_local"]) / v.data["cs_local"]


def cs_local(v):
    return np.sqrt(v.data["P_local"] / v.data["RHO_local"])


quantities = [
    LocalQuantity(
        "vr_local",
        localkey="VX1",
        uids=uids,
        plot_coords=[0, 0],
        yscale="log",
    ),
    LocalQuantity(
        "vtheta_local",
        localkey="VX2",
        uids=uids,
        plot_coords=[0, 1],
        yscale="log",
    ),
    LocalQuantity(
        "vphi_local",
        localkey="VX3",
        uids=uids,
        plot_coords=[0, 2],
        yscale="log",
    ),
    LocalQuantity(
        "P_local",
        localkey="PRS",
        uids=uids,
        plot_coords=[0, 3],
        yscale="log",
    ),
    LocalQuantity(
        "RHO_local",
        localkey="RHO",
        uids=uids,
        plot_coords=[0, 3],
        yscale="log",
    ),
    PartQuantity(
        "cs_local",
        compute=cs_local,
        uids=uids,
        plot_coords=[0, 4],
        yscale="log",
    ),
]


quantities2 = [
    PartQuantity(
        "delta_vr",
        uids=uids,
        plot_coords=[0, 0],
        yscale="log",
        compute=delta_vr,
        xlabel="$t$ [yr]",
        ylabel=r"$\Delta v_r/c_\mathrm{s}$",
        title=r"$\Delta v_r/c_\mathrm{s}$",
    ),
    PartQuantity(
        "delta_vtheta",
        uids=uids,
        plot_coords=[0, 1],
        yscale="log",
        compute=delta_vtheta,
        xlabel="$t$ [yr]",
        ylabel=r"$\Delta v_\theta/c_\mathrm{s}$",
        title=r"$\Delta v_\theta/c_\mathrm{s}$",
    ),
    PartQuantity(
        "delta_vphi",
        uids=uids,
        plot_coords=[0, 2],
        yscale="log",
        compute=delta_vphi,
        xlabel="$t$ [yr]",
        ylabel=r"$\Delta v_\phi/c_\mathrm{s}$",
        title=r"$\Delta v_\phi/c_\mathrm{s}$",
        customize=cb,
    ),
]
for qty in quantities + quantities2:
    qty.parts_color = colors
fig1 = Fig(quantities)
fig2 = Fig(quantities2)

pipeline = Pipeline(
    runContext,
    [fig1, fig2],
    # scatter_particles=True,
)

pipeline.run()
