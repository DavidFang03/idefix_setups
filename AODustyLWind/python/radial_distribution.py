from idefix2python import (
    RunContext,
    Pipeline,
    MapMovie2D,
    OneComponentOneVariable,
    PartQuantity,
    LocalQuantity,
    OneComponentOneVariable,
    Fig,
)
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "dw100_v2_intro_b1e4_2000p"

runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    dataFolder=f"{projectPath}/outputs_v2/{task}/vtks",
    iniPath=f"{projectPath}/inputs_v2/{task}.ini",
)
RLine = runContext.gridInfo.X1Line
ThetaLine = runContext.gridInfo.X2Line
Rgrid, Thetagrid = np.meshgrid(RLine, ThetaLine)


def zoom(x1, x2):
    # return x1 < 10, x2 <= np.pi / 2
    return np.ones_like(x1, dtype=bool), x2 <= np.pi / 2


num_r = int(runContext.inidata["Particles"]["num_r"][0])
num_theta = int(runContext.inidata["Particles"]["num_theta"][0])
num_size = int(runContext.inidata["Particles"]["num_size"][0])

uids_grid = np.arange(num_r * num_theta * num_size).reshape(num_r, num_theta, num_size)

same_r = uids_grid[9, :, :].flatten()
same_angle = uids_grid[:, 2, :].flatten()
same_size = uids_grid[:, :, 0:3].flatten()

same_pos = [uid for uid in same_r if uid in same_angle]
# uids = list(same_r)
uids = list(same_pos)
# uids = [uids[2]]
# uids = [0]


def fourH(ax, vtk):
    ax.axhline(y=np.atan(4 * 0.05), color="red")


def fiveH(ax, vtk):
    ax.axhline(y=np.atan(5 * 0.05), color="red")


def add_line_colorbar(ax, vtk):
    sizes = size(vtk)
    unique_sizes = np.unique(sizes)
    num_colors = len(unique_sizes)

    # 1. Create a discrete colormap with exactly the right number of steps
    discrete_cmap = plt.get_cmap("managua", num_colors)

    # 2. Create boundaries so each unique value gets its own centered bin
    # We create bins centered around the unique values
    half_step = (unique_sizes[1] - unique_sizes[0]) / 2 if num_colors > 1 else 0.5
    boundaries = np.append(unique_sizes - half_step, unique_sizes[-1] + half_step)

    # 3. Setup the norm and scalar mappable
    norm = mcolors.BoundaryNorm(boundaries, ncolors=discrete_cmap.N)
    sm = cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])

    # 4. Plot the colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        pad=0.05,
        ticks=unique_sizes,  # Places the tick marks exactly in the middle of each color block
    )
    cbar.set_label("Sizes [m]")


def size(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def colors(vtk):
    sizes = size(vtk)
    parts_cmap = plt.get_cmap("managua")
    return parts_cmap(sizes / np.max(sizes))


def angle(v):
    return (np.pi / 2 - v.data["PART_X2"]) * 180 / np.pi


# def inclination_with_size(v):
#     nb_particles_with_same_angle = len(uids_grid[:,0, :])
#     print("nb_particles_with_same_angle", nb_particles_with_same_angle) # should be 10
#     data = {}
#     for size in v.data["DRAGCOEFF"]:
#         if size not in data:
#             data[size]=[]

#         for ll in nb_particles_with_same_angle:
#             for uids_with_same_angle in uids_grid[:, ll, :]:
#                 angle_in_question =
#                 for uid in uids_with_same_angle:

sizeqty = PartQuantity(
    "SIZE",
    r"size (m)",
    uids=uids,
    plot_coords=[0, 2],
    compute=size,
    # customize=add_line_colorbar,
)
quantities = [
    PartQuantity(
        "angle",
        r"$\frac{\pi}{2} - \theta^\mathrm{dust}$",
        title="Inclination",
        uids=uids,
        plot_coords=[0, 0],
        customize=fiveH,
        compute=angle,
        parts_color=colors,
    ),
    sizeqty,
    PartQuantity(
        "inclination_with_angle",
        r"inclination [deg]",
        uids=uids,
        plot_coords=[0, 1],
        xqty=sizeqty,
        compute=angle,
        bounds=[10, 70],
        yscale="log",
    ),
]
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [Fig(quantities)], scatter_particles=True)

pipeline.run()
