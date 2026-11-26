import os

import appl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def size(v):
    beta = v["dragcoeff"].data
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos


# def add_line_colorbar(ax, vtk, on):
#     sizes = size(vtk)
#     unique_sizes = np.unique(sizes)
#     num_colors = len(unique_sizes)

#     # 1. Create a discrete colormap with exactly the right number of steps
#     discrete_cmap = plt.get_cmap("managua", num_colors)

#     # 2. Create boundaries so each unique value gets its own centered bin
#     # We create bins centered around the unique values
#     half_step = (
#         np.sqrt(unique_sizes[1] * unique_sizes[0]) / 2 if num_colors > 1 else 0.5
#     )
#     boundaries = np.append(unique_sizes - half_step, unique_sizes[-1] + half_step)

#     # 3. Setup the norm and scalar mappable
#     norm = matplotlib.colors.BoundaryNorm(boundaries, ncolors=discrete_cmap.N)
#     sm = matplotlib.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
#     sm.set_array([])

#     # 4. Plot the colorbar
#     fig = ax.get_figure()
#     cbformat = matplotlib.ticker.ScalarFormatter()
#     cbformat.set_scientific("%.2e")
#     cbformat.set_powerlimits((-2, 12))
#     cbformat.set_useMathText(True)

#     cbar = fig.colorbar(
#         sm,
#         ax=ax,
#         orientation="horizontal",
#         pad=0.05,
#         ticks=unique_sizes,  # Places the tick marks exactly in the middle of each color block
#         format=cbformat,
#     )
#     from matplotlib.ticker import FuncFormatter

#     def format_math_text(x, pos):
#         if x == 0:
#             return "$0$"
#         # Get scientific notation components (e.g., "1.03e-03" -> base="1.03", exponent="-03")
#         s = f"{x:.2e}"
#         base, exponent = s.split("e")
#         # Strip leading zeros/plus signs from exponent
#         exponent = int(exponent)
#         return f"${base} \\times 10^{{{exponent}}}$"

#     cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_math_text))

#     # Control the label spacing so they don't overlap
#     n_labels = len(unique_sizes)
#     # if n_labels > 15:
#     #     skip = (
#     #         n_labels // 8
#     #     )  # Adjusted slightly to give the math text more breathing room
#     #     visible_ticks = unique_sizes[::skip]
#     #     cbar.set_ticks(visible_ticks)

#     cbar.set_label("Sizes [m]")


def format_math_text(x, pos):
    if x == 0:
        return "$0$"
    s = f"{x:.2e}"
    base, exponent = s.split("e")
    return f"${base} \\times 10^{{{int(exponent)}}}$"


def add_line_colorbar(ax, vtk, nb, pipeline, log_scale=True, max_ticks=6):
    sizes = np.asarray(size(vtk)) * 1e6
    unique_sizes = np.unique(sizes)
    num_colors = len(unique_sizes)

    # 1. Choose continuous vs. discrete mapping based on count
    if num_colors == 1:
        val = unique_sizes[0]
        cmap = plt.get_cmap("managua", 1)
        norm = matplotlib.colors.BoundaryNorm([val - 0.5, val + 0.5], ncolors=1)
        ticks = [val]
    elif num_colors <= 15 and not log_scale:
        # True discrete bins using midpoints between adjacent sorted values
        cmap = plt.get_cmap("managua", num_colors)
        midpoints = 0.5 * (unique_sizes[:-1] + unique_sizes[1:])
        boundaries = np.concatenate(
            [
                [unique_sizes[0] - (midpoints[0] - unique_sizes[0])],
                midpoints,
                [unique_sizes[-1] + (unique_sizes[-1] - midpoints[-1])],
            ]
        )
        norm = matplotlib.colors.BoundaryNorm(boundaries, ncolors=cmap.N)
        ticks = unique_sizes
    else:
        # Continuous colormap for 40 - 1000+ values
        cmap = plt.get_cmap("managua")
        if log_scale:
            positive = sizes[sizes > 0]
            norm = matplotlib.colors.LogNorm(vmin=positive.min(), vmax=positive.max())
            ticks = matplotlib.ticker.LogLocator(numticks=max_ticks)
        else:
            norm = matplotlib.colors.Normalize(vmin=sizes.min(), vmax=sizes.max())
            ticks = matplotlib.ticker.MaxNLocator(nbins=max_ticks)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 2. Create horizontal colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        ticks=ticks,
    )

    # 3. Format xaxis (not yaxis!) for horizontal orientation
    # cbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_math_text))
    # cbar.ax.tick_params(
    #     axis="x", rotation=30
    # )  # Prevents scientific notation collisions
    cbar.set_label("Sizes [microns]")
    return cbar


def colors(ds):
    sizes = size(ds)
    parts_cmap = plt.get_cmap("managua")
    return parts_cmap(sizes / np.max(sizes))


def scatter(ax, ds, nb, pipeline):
    r, theta, sizes = ds["x1"].data, ds["x2"].data, size(ds) * 1e6
    nb_particles = len(r)
    x, y = r * np.sin(theta), r * np.cos(theta)

    sc = ax.scatter(x, y, c=colors(ds), s=0.1 * 1e5 / nb_particles, alpha=0.5)
    # fig = ax.get_figure()
    # fig.colorbar(sc, label="size (um)")

    ax.set_aspect("equal")
    # ax.set_xlim(runContext.grid.xmin, runContext.grid.xmax)
    # ax.set_ylim(runContext.grid.ymin, runContext.grid.ymax)


if __name__ == "__main__":
    RUNS_FOLDER_PATH = str(os.getenv("RUNS_FOLDER_PATH"))

    project = "dustywind"
    runName = "1e6_b1e3"

    runContext = appl.Context(
        runName=runName,
        projectPath=f"{RUNS_FOLDER_PATH}/{project}",
    )

    num_r = int(runContext.inidata["Particles"]["num_r"])
    num_theta = int(runContext.inidata["Particles"]["num_theta"])
    num_size = int(runContext.inidata["Particles"]["num_size"])

    uids_grid = np.arange(num_r * num_theta * num_size).reshape(
        num_r, num_theta, num_size
    )

    same_r = uids_grid[0, :, :].flatten()
    same_angle = uids_grid[:, 0, :].flatten()
    same_size = uids_grid[:, :, 0:2].flatten()

    same_pos = [uid for uid in same_r if uid in same_angle]
    uids = same_pos
    # uids = list(uids_grid.flatten()[::10])
    # same_pos = [uid for uid in same_r if uid in same_angle and uid in same_size]
    # uids = list(same_r)
    # uids = [list(same_pos)[0]]
    # uids = list(same_size)
    # uids = "all"

    eps = float(runContext.inidata["Setup"]["epsilon"])
    betamid = float(runContext.inidata["Setup"]["beta"])
    densityFloor = float(runContext.inidata["Setup"]["densityFloor"])

    # def title(ax, v, nb):
    #     fig = ax.get_figure()
    #     fig.suptitle(rf"$t={float_to_latex(runContext.times_all[nb] / (2 * np.pi))}$ yr")

    # def totaddmass(v):
    #     dr, dtheta = np.meshgrid(np.diff(v.rl), np.diff(v.thetal))
    #     return np.sum(v.data["addedMass"] * Rgrid * dr * dtheta)

    quantities = [
        appl.MapMovie2D(
            "RHO",
            r"$\rho$",
            title="Gas density",
            plot_coords=[0, 0],
            streamlines=["VX1", "VX2"],
            bounds=[densityFloor * 1000, 1e-2],
            style_kwargs={"cmap": "inferno"},
            customize=add_line_colorbar,
            ymin=0,
            ymax=50,
            norm="log",
            uids=uids,
            parts_color=colors,
        ),
        appl.CustomQuantity(
            "scatter",
            # r"$Size distribution$",
            title="Size distribution",
            plot_coords=[0, 1],
            customize=scatter,
            # ymin=0,
            # ymax=20,
            # xmin=0,
            # xmax=40,
            xmin=0,
            xmax=100,
            ymin=0,
            ymax=50,
        ),
    ]

    # for qty in quantities:
    #     qty.uids = uids
    #     qty.parts_color = colors
    fig1 = appl.Fig(quantities)
    # fig1.axes[0, 0].xmin = 0

    custom_fields2D = []

    # Initialize context
    # Inject ONLY the 2D fields into the pipeline
    pipeline = appl.Pipeline(
        runContext,
        [fig1],
    )
    pipeline.run()
