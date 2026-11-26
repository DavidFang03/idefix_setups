import os

import appl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import formulas


def format_math_text(x, pos):
    if x == 0:
        return "$0$"
    s = f"{x:.2e}"
    base, exponent = s.split("e")
    return f"${base} \\times 10^{{{int(exponent)}}}$"


def add_line_colorbar(ax, ds, on=None, log_scale=True, max_ticks=6):
    sizes = np.asarray(formulas.size(ds)) * 1e6
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
    sizes = formulas.size(ds)
    parts_cmap = plt.get_cmap("managua")
    return parts_cmap(sizes / np.max(sizes))


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


def the_greatest_detector_of_the_death_that_kills_ever_created(
    pipeline, yearsmax, Hmax=5
):
    particles = pipeline.particles

    previndex = -2
    R0 = np.sqrt(
        particles.abscissas[previndex, :] ** 2 + particles.ordinates[previndex, :] ** 2
    )
    Rlast = np.sqrt(particles.abscissas[-1, :] ** 2 + particles.ordinates[-1, :] ** 2)

    HLast = particles.ordinates[-1, :] / particles.abscissas[-1, :]
    # delta_t is the time elapsed between the 2 latest outputs.
    delta_t = runContext.times_all[-1] - runContext.times_all[previndex]

    # When does the particle might escape the grid i.e R > Rmax?
    Rmax = 120
    # Not putting 100 because a particle might be deactivated exactly be already beyond?
    # t_out = delta_t * (Rmax - R0) / (Rlast - R0)
    diff = Rlast - R0
    t_out = np.full_like(diff, -np.inf, dtype=float)
    np.divide(delta_t * (Rmax - R0), diff, out=t_out, where=(diff > 0))

    uids = ((t_out / (2 * np.pi)) > yearsmax) | (HLast < Hmax)

    return np.arange(len(uids))[uids]


def histstuff(ax):
    ax.legend()


yearsmax = 1e3


def get_particles_below_vesc(ds):
    v = np.sqrt(ds["vx1"].data ** 2 + ds["vx2"].data ** 2 + ds["vx3"].data ** 2)
    remaining = v / formulas.escape_velocity(ds) < 1.0
    return remaining


def histsize(ax, ds, on, pipeline):
    dst = runContext.datatest_particles
    # Initial distribution
    sizes = formulas.size(dst)
    # Get sorted unique sizes and their exact counts
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    # Plot one point per discrete size (use 'o-' for line + markers, or 'o' for points only)
    ax.plot(unique_sizes, counts, "o-", markersize=4, label="Initial ditribution")

    # Using interpolation
    uids = the_greatest_detector_of_the_death_that_kills_ever_created(
        pipeline, yearsmax
    )
    duids = dst["uid"].data
    mask = np.isin(duids, uids)
    unique_sizes, counts = np.unique(sizes[mask], return_counts=True)
    ax.plot(
        unique_sizes,
        counts,
        "o-",
        markersize=4,
        label=f"After interpolating trajectory, guess these particles will still be inside the grid at {yearsmax:.0e} yr",
    )

    # At this timestep, which ones have reach escape velocity?
    sizes = formulas.size(ds)

    remaining = get_particles_below_vesc(ds)
    sizes_remaining = sizes[remaining]
    unique_sizes, counts = np.unique(sizes_remaining, return_counts=True)
    ax.plot(
        unique_sizes,
        counts,
        "o-",
        markersize=4,
        label=f"Particles that haven't reached escape velocity yet at {runContext.years_all[on]:.1e} yr",
        color="green",
    )

    ax.set_xlabel(r"Sizes [m]")
    ax.set_ylabel(r"Number of particles")
    ax.set_ylim(1e3, 1.1 * num_theta * num_r)
    histstuff(ax)
    ax.grid(which="minor")


def hist_st(ax, ds, on, pipeline):
    bins = num_size
    bins = np.logspace(-6, 0.5, bins)

    dst = runContext.datatest_particles
    st = formulas.St(dst)
    ax.hist(
        st,
        label="Initial ditribution",
        bins=bins,
        histtype="step",
    )

    st = formulas.St(ds)

    # uids = the_greatest_detector_of_the_death_that_kills_ever_created(
    #     pipeline, yearsmax
    # )
    # duids = ds["uid"].data
    # mask = np.isin(duids, uids)
    # ax.hist(
    #     st[mask],
    #     label=f"After interpolating trajectory, guess these particles will still be inside the grid at {yearsmax:.0e} yr",
    #     bins=bins,
    #     histtype="step",
    # )

    remaining = get_particles_below_vesc(ds)
    ax.hist(
        st[remaining],
        label=f"Particles that haven't reached escape velocity yet at {runContext.years_all[on]:.1e} yr",
        bins=bins,
        histtype="step",
        color="green",
    )

    ax.set_xlabel(r"Stokes number")
    ax.set_ylabel(r"Number of particles")
    ax.set_ylim(1, 1e6)
    histstuff(ax)
    ax.grid()


quantities = [
    # appl.PartQuantity(
    #     "v_vesc",
    #     r"$v / v_\mathrm{esc}$",
    #     title="Size distribution",
    #     plot_coords=[0, 1],
    #     compute=v_vesc,
    #     uids=list(uids_grid.flatten()[::50]),
    # ),
    appl.CustomQuantity(
        "histsize",
        r"size",
        title="Size distribution",
        plot_coords=[0, 0],
        customize=histsize,
        xscale="log",
        yscale="log",
    ),
    appl.CustomQuantity(
        "hist_st",
        r"st",
        title="Stokes number distribution",
        plot_coords=[0, 1],
        customize=hist_st,
        xscale="log",
        yscale="log",
    ),
]

uids_grid = np.arange(num_r * num_theta * num_size).reshape(num_r, num_theta, num_size)

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

fig1 = appl.Fig(quantities)

pipeline = appl.Pipeline(
    runContext, [fig1], particles_requested=True, custom_name="stats"
)


def v_vesc(ds):
    v = np.sqrt(ds["vx1"].data ** 2 + ds["vx2"].data ** 2 + ds["vx3"].data ** 2)
    return v / formulas.escape_velocity(ds)


if __name__ == "__main__":
    pipeline.run()
