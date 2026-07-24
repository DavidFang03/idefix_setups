from idefix2python import readVTK
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import inifix
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap

cmap = "viridis"
plt.style.use("dark_background")

# cmap = Colormap("crameri:devon").to_mpl()

cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_scientific("%.2e")
cbformat.set_powerlimits((-2, 12))
cbformat.set_useMathText(True)


def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    loc = "right"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size="4%", pad="5%")
    cbar = fig.colorbar(mappable, cax=cax, location=loc, format=cbformat)
    plt.sca(last_axes)
    return cbar


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
    return f"{mantissa} \\cdot 10^{{{exponent}}}"


betas = ["1e4", "4e3"]
betas_float = []


def SIZE(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def angle(v):
    return np.pi / 2 - v.data["PART_X2"]


def get_treshold_size(vtk, uids):
    corona_angle = np.atan(Hideal * 0.05)

    angles = []
    sizes = []
    vtkuids = vtk.data["uid"]
    for ii in range(len(vtkuids)):
        if vtkuids[ii] in uids:
            # sizes.append(vtk.data["TSTOP"][ii])
            sizes.append(vtk.data["TSTOP"][ii]*vtk.data["PART_X1"][ii]**(-1.5))
            angles.append(vtk.data["angle"][ii])

    iis = np.argsort(np.abs(angles - corona_angle))

    if (angles[iis[1]] - corona_angle) * (angles[iis[0]] - corona_angle) < 0:
        result = (sizes[iis[0]] + sizes[iis[1]]) / 2
    else:
        result = sizes[iis[0]]
    return result, uids[0]


plt.rcParams.update({"text.usetex": True})


last_times = []


def plot_and_annotate_lines(ax, Xline, Hideal, epsilon):
    """
    Plots lines relative to Hideal and annotates them at the far right,
    perfectly inclined to match the visual slope of each line.
    """
    line_configs = [
        (Hideal - 2, f"${Hideal - 2}H$"),
        (Hideal - 1, f"${Hideal - 1}H$"),
        (Hideal, f"${Hideal}H$ (Corona)"),
        (Hideal + 1, f"${Hideal + 1}H$"),
        (Hideal + 2, f"${Hideal + 2}H$"),
    ]

    for H_val, label_text in line_configs:
        Y_line = H_val * Xline * epsilon

        ax.plot(
            Xline,
            Y_line,
            color="white",
            alpha=0.5,
        )

        x_points = Xline[-2:]
        y_points = Y_line[-2:]

        trans = ax.transData.transform
        p1 = trans((x_points[0], y_points[0]))
        p2 = trans((x_points[1], y_points[1]))

        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

        ax.annotate(
            label_text,
            xy=(Xline[-1], Y_line[-1]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            rotation=angle,  # Rotates text to match the line slope
            rotation_mode="anchor",  # Ensures rotation happens *before* alignment
        )


for beta in betas:
    fig, axes = plt.subplots(
        1,
        squeeze=False,
        figsize=(8, 6),
    )

    paths = sorted(
        glob.glob(
            f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v2/dw100_v2_thin_b{beta}_2000p/vtks/part.*.vtk"
        )
    )
    iniPath = Path(
        f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/inputs_v2/dw100_v2_thin_b{beta}_2000p.ini"
    )
    with iniPath.open("rb") as fh:
        inidata = inifix.load(fh, sections="require")
    Hideal = inidata["Setup"]["Hideal"]
    epsilon = inidata["Setup"]["epsilon"]
    num_r = inidata["Particles"]["num_r"][0]
    num_theta = inidata["Particles"]["num_theta"][0]
    num_size = inidata["Particles"]["num_size"][0]

    uids_grid = np.arange(num_r * num_theta * num_size).reshape(
        num_r, num_theta, num_size
    )

    vtk = readVTK(paths[-1])
    firstvtk = readVTK(paths[1])
    vtk.data["PART_X2"] = vtk.theta
    vtk.data["PART_X1"] = vtk.r
    vtk.data["SIZE"] = SIZE(vtk)
    vtk.data["angle"] = angle(vtk)

    r_range = range(2, num_r)
    theta_range = range(num_theta)

    X_grid = np.zeros((len(r_range), len(theta_range)))
    Z_grid = np.zeros((len(r_range), len(theta_range)))
    Size_grid = np.zeros((len(r_range), len(theta_range)))

    for i, r in enumerate(r_range):
        for j, theta in enumerate(theta_range):
            same_r = uids_grid[r, :, :].flatten()
            same_angle = uids_grid[:, theta, :].flatten()
            same_pos = [uid for uid in same_r if uid in same_angle]
            uids = list(same_pos)

            size, uid = get_treshold_size(vtk, uids)
            r_ini = firstvtk.r[uid]
            theta_ini = firstvtk.theta[uid]

            X_grid[i, j] = r_ini * np.sin(theta_ini)
            Z_grid[i, j] = r_ini * np.cos(theta_ini)
            Size_grid[i, j] = size

            last_times.append(vtk.t[0])

    mesh2 = axes[0, 0].pcolormesh(
        X_grid,
        Z_grid,
        Size_grid,
        shading="auto",
        cmap=cmap,
        edgecolors="none",
        rasterized=True,
        # antialiased=True,
    )

    axes[0, 0].scatter(
        X_grid,
        Z_grid,
        c=Size_grid,
        cmap=cmap,
        # marker="x",
        s=10,
        linewidths=1,  # Thicker line width for the "edge" effect
        edgecolors="white",
        alpha=0.5,
        vmin=Size_grid.min(),
        vmax=Size_grid.max(),
        label="Particle initial position",
    )

    xmin, xmax = X_grid.min(), X_grid.max()
    zmin, zmax = Z_grid.min(), Z_grid.max()
    Xline = np.linspace(0, 1.15 * xmax)
    for ax in axes.flatten():
        plot_and_annotate_lines(ax, Xline, int(Hideal), epsilon)
        ax.set_aspect("equal")
        ax.set_xlim(0.9 * xmin, 1.15 * xmax)
        ax.set_ylim(0.9 * zmin, 1.1 * zmax)
        ax.set_ylabel(r"$z$ [au]")
        ax.legend(loc="upper left")

    cbar = colorbar(mesh2)
    cbar.set_label(r"$\mathrm{St}_\mathrm{crit}$ [m]")

    axes[0, 0].set_xlabel(r"$x$ [au]")
    fig.suptitle(rf"$\beta_\mathrm{{mid}} = {float_to_latex(float(beta))}$", y=0.8)

    figpath = f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/thin_{beta}.png"
    fig.savefig(figpath, dpi=400, bbox_inches="tight")
    print(figpath)
    fig.clf()
