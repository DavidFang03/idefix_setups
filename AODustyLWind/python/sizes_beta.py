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
ext = "png"
# cmap = Colormap("crameri:devon").to_mpl()

cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_scientific("%.2e")
cbformat.set_powerlimits((-2, 12))
cbformat.set_useMathText(True)


def get_vtks_path(betamid):
    return f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v2/dw100_v2_thin_tv_b{betamid}_2000p/vtks"


def add_mesh_labels(ax, X, Z, data, fmt=".2f", color="black", fontsize=8):
    """Overlays text labels onto each cell of a pcolormesh plot

    where X and Z represent cell centers.
    """
    # Loop through rows and columns of the data grid
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Since X and Z are already centers, use them directly
            x_center = X[i, j]
            z_center = Z[i, j]

            val = data[i, j]

            # Skip labeling NaNs if they exist in your data
            if not np.isnan(val):
                ax.text(
                    x_center,
                    z_center,
                    f"{val:{fmt}}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=fontsize,
                )


def colorbar(mappable, loc="right"):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
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


# betas = ["1e4", "4e3"]
betas = (
    # "1e3",
    "4e3",
    "7e3",
    "1e4",
    "4e4",
    "6e4",
    "8e4",
)

betas_float = [float(beta) for beta in betas]


def SIZE(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def angle(v):
    return np.pi / 2 - v.data["PART_X2"]


def get_treshold_size(vtk, uids, firstvtk, thetas):
    corona_angle = np.atan(Hideal * epsilon)

    angles = []
    sizes = []
    vtkuids = vtk.data["uid"]
    for ii in range(len(vtkuids)):
        R = firstvtk.r[ii] * np.sin(firstvtk.theta[ii])
        stokes = firstvtk.data["TSTOP"][ii] * R ** (-1.5)
        print(stokes)

        sizes.append(stokes)
        # sizes.append(vtk.data["SIZE"][ii])
        # angles.append(vtk.data["angle"][ii])
        angles.append(np.pi / 2 - thetas[ii])

    iis = np.argsort(np.abs(angles - corona_angle))

    print(len(iis))
    if (
        len(iis) > 1
        and (angles[iis[1]] - corona_angle) * (angles[iis[0]] - corona_angle) < 0
    ):
        result = (sizes[iis[0]] + sizes[iis[1]]) / 2
    elif len(iis) > 0:
        result = sizes[iis[0]]
    else:
        result = None

    print(result)
    return result, uids[0]


plt.rcParams.update({"text.usetex": True})


last_times = []


def get_theta_including_last(vtklist):
    firstvtk = readVTK(vtklist[0])
    thetas = []
    times = []
    firstuids = firstvtk.data["uid"]
    thetas_t = np.array(firstvtk.theta)
    for vtkpath in vtklist[::10] + [vtklist[-1]]:
        vtk = readVTK(vtkpath)
        uids = vtk.data["uid"]
        for ii, uid in enumerate(uids):
            if uid in firstuids:
                thetas_t[uid] = vtk.theta[ii]
        times.append(vtk.t[0])
        thetas.append(thetas_t.copy())
    return thetas, times


def plot_and_annotate_lines(ax, Xline, Hideal, epsilon):
    """
    Plots lines relative to Hideal and annotates them at the far right,
    perfectly inclined to match the visual slope of each line.
    """
    line_configs = [
        # (Hideal - 2, f"${Hideal - 2}H$"),
        # (Hideal - 1, f"${Hideal - 1}H$"),
        (Hideal, f"${Hideal}H$ (Corona)"),
        (Hideal + 1, f"${Hideal + 1}H$"),
        (Hideal + 2, f"${Hideal + 2}H$"),
    ]

    for H_val, label_text in line_configs:
        Y_line = H_val * Xline * epsilon

        ax.plot(
            Xline,
            Y_line,
            color="black",
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


table = []


def pimp(axes, X_grid, Z_grid, Hideal, epsilon):
    xmin, xmax = X_grid.min(), X_grid.max()
    zmin, zmax = Z_grid.min(), Z_grid.max()
    Xline = np.linspace(0, 1.15 * xmax)
    for ax in axes.flatten():
        plot_and_annotate_lines(ax, Xline, int(Hideal), epsilon)
        ax.set_aspect("equal")
        ax.set_xlim(0.9 * xmin, 1.15 * xmax)
        ax.set_ylim(0.9 * zmin, 1.1 * zmax)
        ax.set_ylabel(r"$z$ [au]")
    axes[0, 0].set_xlabel(r"$x$ [au]")


for beta in betas:
    fig, axes = plt.subplots(
        1,
        squeeze=False,
        figsize=(8, 6),
    )

    vtks_path = get_vtks_path(beta)
    paths = sorted(glob.glob(f"{vtks_path}/part.*.vtk"))
    iniPath = Path(
        f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/inputs_v2/dw100_v2_thin_tv_b{beta}_2000p.ini"
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
    firstvtk = readVTK(paths[0])
    secondvtk = readVTK(paths[1])
    vtk.data["PART_X2"] = vtk.theta
    vtk.data["SIZE"] = SIZE(vtk)
    vtk.data["angle"] = angle(vtk)

    thetas, times = get_theta_including_last(paths)
    thetas = np.asarray(thetas)

    r_range = range(2, num_r)
    theta_range = range(num_theta)

    X_grid = np.zeros((len(r_range), len(theta_range)))
    Z_grid = np.zeros((len(r_range), len(theta_range)))
    r_grid = np.zeros((len(r_range), len(theta_range)))
    theta_grid = np.zeros((len(r_range), len(theta_range)))
    Size_grid = np.zeros((len(r_range), len(theta_range)))

    for i, r in enumerate(r_range):
        for j, theta in enumerate(theta_range):
            same_r = uids_grid[r, :, :].flatten()
            same_angle = uids_grid[:, theta, :].flatten()
            same_pos = [uid for uid in same_r if uid in same_angle]
            uids = list(same_pos)

            figangle, axangle = plt.subplots()
            print(np.shape(thetas))
            for uid in uids:
                axangle.plot(times, thetas[:, uid])
            figangle.suptitle(beta)
            figangle.savefig(f"angles{beta}.png")

            size, uid = get_treshold_size(vtk, uids, secondvtk, thetas[-1])
            r_ini = firstvtk.r[uid]
            theta_ini = firstvtk.theta[uid]
            r_grid[i, j] = r_ini
            theta_grid[i, j] = theta_ini
            X_grid[i, j] = r_ini * np.sin(theta_ini)
            Z_grid[i, j] = r_ini * np.cos(theta_ini)
            Size_grid[i, j] = size

    last_times.append(vtk.t[0])

    table.append(Size_grid)

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
        edgecolors="black",
        alpha=0.5,
        vmin=Size_grid.min(),
        vmax=Size_grid.max(),
        label="Particle initial position",
    )

    pimp(axes, X_grid, Z_grid, Hideal, epsilon)
    for ax in axes.flatten():
        ax.legend(loc="upper left")

    cbar = colorbar(mesh2)
    cbar.set_label(r"$s_\mathrm{crit}$ [m]")

    fig.suptitle(rf"$\beta_\mathrm{{mid}} = {float_to_latex(float(beta))}$", y=0.8)

    # figpath = f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/thin_{beta}.{ext}"
    # fig.savefig(figpath, dpi=400, bbox_inches="tight")
    # print(figpath)
    fig.clf()

fitting_value = r"$B_\mathrm{local}^2$"
fitting_value = r"$B_\mathrm{local}^2$"


def get_local_beta(r, theta, betamid):
    vtks_path = get_vtks_path(betamid)
    paths = sorted(glob.glob(f"{vtks_path}/data.*.vtk"))
    vtk = readVTK(paths[1])
    i = np.searchsorted(vtk.r, r)
    j = np.searchsorted(vtk.theta, theta)
    # print(vtk.data["PRS"].shape)
    p = vtk.data["PRS"][i, j, 0]
    B2 = (
        vtk.data["BX1"][i, j, 0] ** 2
        + vtk.data["BX2"][i, j, 0] ** 2
        + vtk.data["BX3"][i, j, 0] ** 2
    )
    return 1 / B2
    # return 8 * np.pi * p / B2


fig0, axes0 = plt.subplots(
    1,
    squeeze=False,
    figsize=(8, 8),
)
ax = axes0[0, 0]
grid_slope = np.zeros(Size_grid.shape)
grid_beta = np.zeros(Size_grid.shape)
for i, r in enumerate(r_range):
    for j, theta in enumerate(theta_range):
        same_r = uids_grid[r, :, :].flatten()
        same_angle = uids_grid[:, theta, :].flatten()
        same_pos = [uid for uid in same_r if uid in same_angle]
        uids = list(same_pos)

        r_ini = r_grid[i, j]
        theta_ini = theta_grid[i, j]
        alt_ini = np.tan(np.pi / 2 - theta_ini) / epsilon

        list_sizes = []
        local_betas = []
        for ii, beta in enumerate(betas):
            Size_grid = table[ii]
            list_sizes.append(Size_grid[i, j])
            local_beta = get_local_beta(r_ini, theta_ini, betamid=beta)
            local_betas.append(local_beta)

        a, b = np.polyfit(np.log(local_betas), np.log(list_sizes), deg=1)
        # a, b = np.polyfit(np.log(betas_float), np.log(list_sizes), deg=1)
        expb = np.exp(b)

        grid_slope[i, j] = np.nan
        if alt_ini < 5 or r_ini < 15:
            continue
        # if alt_ini > 5:
        grid_slope[i, j] = a
        s1e4 = 1e4**a * expb

        if i % 3 == 0 and j % 3 == 0:
            (line,) = ax.plot(
                local_betas,
                local_betas**a * expb,
                label=rf"Initial position: $({r_ini:.1f},{alt_ini:.1f}H) \quad \frac{{s}}{{{s1e4:.1e}\mathrm{{m}}}}= \left(\frac{{B_\mathrm{{local}}^2}}{{10 ^ 4}}\right)^{{{a:.3}}}$",
                # label=rf"Initial position: $({r_ini:.1f},{alt_ini:.1f}H) \quad \frac{{s}}{{{s1e4:.1e}\mathrm{{m}}}}= \left(\frac{{\beta_\mathrm{{mid}}}}{{10 ^ 4}}\right)^{{{a:.3}}}$",
            )
            # (line,) = ax.plot(
            #     betas_float,
            #     betas_float**a * expb,
            #     label=rf"Initial position: $({r_ini:.1f},{alt_ini:.1f}H) \quad \frac{{s}}{{{s1e4:.1e}\mathrm{{m}}}}= \left(\frac{{\beta_\mathrm{{mid}}}}{{10 ^ 4}}\right)^{{{a:.3}}}$",
            # )

            ax.scatter(local_betas, list_sizes, color=line.get_color())


ax.set_xlabel(r"$B_\mathrm{local}^{-2}$")
# ax.set_xlabel(r"$\beta_\mathrm{mid}$")
ax.set_ylabel(r"$s_\mathrm{crit}$ [m]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc="upper right")

figpath = f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/sizes_beta.{ext}"
fig0.savefig(figpath, dpi=300, bbox_inches="tight")
print(figpath)


figmesh, axmesh = plt.subplots(
    1,
    squeeze=False,
    figsize=(8, 6),
)

cmap = Colormap("chrisluts:bop_blue").to_mpl()


mesh2 = axmesh[0, 0].pcolormesh(
    X_grid,
    Z_grid,
    grid_slope,
    shading="auto",
    cmap=cmap,
    edgecolors="none",
    rasterized=True,
    # antialiased=True,
)
pimp(axmesh, X_grid, Z_grid, Hideal, epsilon)
cbar = colorbar(mesh2)
cbar.ax.set_title(r"$\sigma$")
axmesh[0, 0].set_title(r"$s_\mathrm{crit} \propto \beta_\mathrm{mid}^{\sigma}$")
axmesh[0, 0].set_title(r"$s_\mathrm{crit} \propto B_\mathrm{local}^{-2\sigma}$")
# y4h = H_val * Xline * epsilon
# axmesh[0,0].

add_mesh_labels(
    axmesh[0, 0], X_grid, Z_grid, grid_slope, fmt=".2f", color="white", fontsize=8
)

pathfinal = f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/sizes_beta_mesh.{ext}"
figmesh.savefig(pathfinal, bbox_inches="tight")
print(pathfinal)
