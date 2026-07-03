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
    custom_name="escape_phi",
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
    print(len(sizes))
    norm = LogNorm(vmin=np.min(sizes), vmax=np.max(sizes))

    return parts_cmap(norm(sizes))


def velocity(vtk):
    return np.sqrt(
        0
        # + vtk.data["PART_VX1"] ** 2
        # + vtk.data["PART_VX2"] ** 2
        + vtk.data["PART_VX3"] ** 2
    )


def escape_velocity(vtk):
    d = np.sqrt(
        vtk.data["PART_X1"] ** 2 + vtk.data["PART_X2"] ** 2 + vtk.data["PART_X3"] ** 2
    )
    return np.sqrt(2 / d)


def normalized_velocity(vtk):
    return velocity(vtk) / escape_velocity(vtk)


quantities = [
    PartQuantity(
        "v",
        uids=uids,
        plot_coords=[0, 0],
        yscale="log",
        compute=velocity,
        xlabel="$t$ [yr]",
        ylabel=r"$v$",
    ),
    PartQuantity(
        "ev",
        uids=uids,
        plot_coords=[0, 1],
        yscale="log",
        compute=escape_velocity,
        xlabel="$t$ [yr]",
        ylabel=r"$v_\mathrm{esc}$",
    ),
]


quantities2 = [
    PartQuantity(
        "evn",
        uids=uids,
        title="Azimuthal velocity",
        plot_coords=[0, 0],
        yscale="log",
        compute=normalized_velocity,
        xlabel="$t$ [yr]",
        ylabel=r"$v_\phi/v_\mathrm{esc}$",
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
