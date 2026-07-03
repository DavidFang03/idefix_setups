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

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
task = "dw100_v3_thin_tv_b4e4_2000p"

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
print(uids_grid.shape)

same_r = uids_grid[5, :, :].flatten()
same_angle = uids_grid[:, 3, :].flatten()
same_size = uids_grid[:, :, 10:20].flatten()


same_pos = [uid for uid in same_r if uid in same_angle]
# same_pos = [uid for uid in same_r if uid in same_angle and uid in same_size]
# uids = list(same_r)
# uids = [list(same_pos)[0]]
# uids = list(same_size)
# uids = "all"
uids = list(uids_grid.flatten()[::10])

print("uids", uids)
# uids = [uids[2]]
# uids = [0]
Hideal = float(runContext.inidata["Setup"]["Hideal"])


def fourH(ax, vtk):
    corona_angle = np.atan(Hideal * 0.05)
    ax.axhline(y=corona_angle, color="red")
    print(vtk.data["DRAGCOEFF"])

    ## determine threshold size
    # sizes = size(vtk)[uids]
    # angles = angle(vtk)[uids]
    # angles = []
    # sizes = []
    # vtkuids = vtk.data["uid"]
    # for ii in range(len(vtkuids)):
    #     if vtkuids[ii] in uids:
    #         sizes.append(vtk.data["SIZE"][ii])
    #         angles.append(vtk.data["angle"][ii])
    # print(len(sizes))
    # iis = np.argsort(np.abs(angles - corona_angle))
    # print(f"closest particle: {iis[0]} with size {sizes[iis[0]]}")
    # print(f"angle difference: {angles[iis[0]] - corona_angle:.1e} ")
    # print(f"next closest particle: {iis[1]} with size {sizes[iis[1]]}")
    # print(f"angle difference: {angles[iis[1]] - corona_angle:.1e} ")
    # if (angles[iis[1]] - corona_angle) * (angles[iis[0]] - corona_angle) < 0:
    #     print(f"I would say take the average {(sizes[iis[0]] + sizes[iis[1]]) / 2}")
    # else:
    #     print(f"I would say keep the first: {sizes[iis[0]]}")


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
    cbformat = matplotlib.ticker.ScalarFormatter()
    cbformat.set_scientific("%.2e")
    cbformat.set_powerlimits((-2, 12))
    cbformat.set_useMathText(True)

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        pad=0.05,
        ticks=unique_sizes,  # Places the tick marks exactly in the middle of each color block
        format=cbformat,
    )
    from matplotlib.ticker import FuncFormatter

    def format_math_text(x, pos):
        if x == 0:
            return "$0$"
        # Get scientific notation components (e.g., "1.03e-03" -> base="1.03", exponent="-03")
        s = f"{x:.2e}"
        base, exponent = s.split("e")
        # Strip leading zeros/plus signs from exponent
        exponent = int(exponent)
        return f"${base} \\times 10^{{{exponent}}}$"

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_math_text))

    # Control the label spacing so they don't overlap
    n_labels = len(unique_sizes)
    # if n_labels > 15:
    #     skip = (
    #         n_labels // 8
    #     )  # Adjusted slightly to give the math text more breathing room
    #     visible_ticks = unique_sizes[::skip]
    #     cbar.set_ticks(visible_ticks)

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


def St(v):
    R = v.data["PART_X1"] * np.sin(v.data["PART_X2"])
    return v.data["TSTOP"] * R ** (-1.5)


def St_custom(v):
    R = v.data["PART_X1"] * np.sin(v.data["PART_X2"])

    rhos = []
    css = []
    for ii, uid in enumerate(v.data["uid"]):
        rho = v.data["RHO_local"][ii]
        prs = v.data["PRS_local"][ii]
        cs = np.sqrt(prs / rho)
        rhos += [rho]
        css += [cs]
    tstop = v.data["DRAGCOEFF"] / (np.asarray(rhos) * np.asarray(css))
    return tstop * R ** (-1.5)


def rhovr(v):
    return v.data["RHO"] * v.data["VX1"]


def z(v):
    return v.data["PART_X1"] * np.cos(v.data["PART_X2"])


def plasmabeta(v):
    P = v.data["PRS"]
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return 8 * np.pi * P / B2


def massloss(v):
    dR = np.diff(v.rl)
    dTheta = np.diff(v.thetal)
    R = v.r
    rho, vr, prs = v.data["RHO"], v.data["VX1"], v.data["PRS"]
    vtheta = v.data["VX2"]
    thetam4h = np.pi / 2 - np.atan(5 * 0.05)
    thetap4h = np.pi / 2 + np.atan(5 * 0.05)
    jm4h = np.searchsorted(ThetaLine, thetam4h)
    jp4h = np.searchsorted(ThetaLine, thetap4h)
    ir10 = np.searchsorted(R, 10)
    jmid = np.searchsorted(ThetaLine, np.pi / 2)

    xiup = np.sum(rho[jp4h, ir10:] * vtheta[jp4h, ir10:] * dR[ir10:])
    xidown = np.sum(rho[jm4h, ir10:] * vtheta[jm4h, ir10:] * dR[ir10:])
    # norm = 2 * np.sum(rho[jmid, :] * np.sqrt(prs[jmid, :] / rho[jmid, :]) * dR)

    norm = (
        -2
        * np.pi
        * R[-1]
        * np.sum(rho[jm4h:jp4h, -1] * vr[jm4h:jp4h, -1] * R[-1] * dTheta[jm4h:jp4h])
    )

    # print("rho", v.data["PRS"])
    # print(rho[jp4h, :])
    # print(vr[jp4h, :])
    # print(xiup, xidown, norm)

    xi = (xiup - xidown) / norm
    # print(xi)

    return xi


def plasmabeta_local(v):
    p = v.data["PRS_local"]
    b2 = v.data["BX1_local"] ** 2 + v.data["BX2_local"] ** 2 + v.data["BX3_local"] ** 2
    return 8 * np.pi * p / b2


def plasmabeta_local_polo(v):
    p = v.data["PRS_local"]
    b2 = v.data["BX1_local"] ** 2 + v.data["BX2_local"] ** 2
    return 8 * np.pi * p / b2


def dvr(v):
    return v.data["PART_X1"] - v.data["VX1_local"]


def dvtheta(v):
    # j, i = get_part_index(v)
    return v.data["PART_X2"] - v.data["VX2_local"]


def dvphi(v):
    return v.data["PART_X3"] - v.data["VX3_local"]


def ElA(v):
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return v.data["Am"] / B2


def RmO(v):
    etaO = v.data["eta"]
    Rcyl = Rgrid * np.sin(Thetagrid)
    Omega = Rcyl ** (-1.5)
    H = 0.05 * Rcyl
    return Omega * H**2 / etaO


def angle(v):
    return np.pi / 2 - v.data["PART_X2"]


localquantities = []
for localkey in ["VX1", "VX2", "VX3", "PRS", "BX1", "BX2", "BX3"]:
    localquantities.append(
        LocalQuantity(
            f"{localkey}_local",
            localkey=localkey,
            uids=uids,
            plot_coords=[3, 4],
        )
    )
quantities = [
    # LocalQuantity(
    #     "RHO_local",
    #     localkey="RHO",
    #     uids=uids,
    #     plot_coords=[2, 6],
    #     yscale="log",
    # ),
    MapMovie2D(
        "RHO",
        plot_coords=[0, 1],
        streamlines=["VX1", "VX2"],
        uids=uids,
        customize=add_line_colorbar,
    ),
    MapMovie2D("InvDt", plot_coords=[0, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX1", plot_coords=[1, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX2", plot_coords=[2, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D("VX3", plot_coords=[3, 0], streamlines=["VX1", "VX2"]),
    MapMovie2D(
        "BX1",
        plot_coords=[1, 1],
        streamlines=["BX1", "BX2"],
    ),
    MapMovie2D(
        "BX2",
        plot_coords=[2, 1],
        streamlines=["BX1", "BX2"],
    ),
    MapMovie2D(
        "BX3",
        plot_coords=[3, 1],
        streamlines=["BX1", "BX2"],
    ),
    PartQuantity(
        "St",
        r"St",
        uids=uids,
        plot_coords=[0, 2],
        compute=St,
        yscale="log",
        # bounds=[0.01, 0.05],
    ),
    # PartQuantity(
    #     "St_custom",
    #     r"St_custom",
    #     uids=uids,
    #     plot_coords=[0, 2],
    #     compute=St_custom,
    #     yscale="log",
    #     # bounds=[0.01, 0.05],
    # ),
    PartQuantity("PART_X1", "$r^\\mathrm{dust}$", uids=uids, plot_coords=[1, 2]),
    PartQuantity(
        "angle",
        r"$\frac{\pi}{2} - \theta^\mathrm{dust}$",
        uids=uids,
        plot_coords=[2, 2],
        customize=fourH,
        compute=angle,
    ),
    PartQuantity(
        "z",
        r"$z^\mathrm{d}$",
        uids=uids,
        compute=z,
        plot_coords=[3, 2],
    ),
    PartQuantity(
        "PART_VX1",
        r"$v_r^\mathrm{d}$",
        uids=uids,
        plot_coords=[0, 3],
    ),
    PartQuantity(
        "PART_VX2",
        r"$v_\theta^\mathrm{d}$",
        uids=uids,
        plot_coords=[1, 3],
    ),
    PartQuantity(
        "PART_VX3",
        r"$v_\phi^\mathrm{d}$",
        uids=uids,
        plot_coords=[2, 3],
    ),
    PartQuantity(
        "dvr", r"$v_r^\mathrm{d}-v_r$", uids=uids, plot_coords=[0, 4], compute=dvr
    ),
    PartQuantity(
        "dvtheta",
        r"$v_\theta^\mathrm{d}-v_\theta$",
        uids=uids,
        plot_coords=[1, 4],
        compute=dvtheta,
    ),
    PartQuantity(
        "dvphi",
        r"$v_\phi^\mathrm{d}-v_\phi$",
        uids=uids,
        plot_coords=[2, 4],
        compute=dvphi,
    ),
    MapMovie2D("beta", "$\\beta$", plot_coords=[0, 5], norm="log", compute=plasmabeta),
    PartQuantity(
        "beta_local",
        r"$\beta^\mathrm{local}$",
        uids=uids,
        plot_coords=[1, 5],
        compute=plasmabeta_local,
        yscale="log",
    ),
    PartQuantity(
        "beta_polo_local",
        r"$\beta_\mathrm{pol}^\mathrm{local}$",
        uids=uids,
        plot_coords=[2, 5],
        compute=plasmabeta_local_polo,
        yscale="log",
    ),
    PartQuantity(
        "SIZE",
        r"size (m)",
        uids=uids,
        plot_coords=[3, 3],
        compute=size,
        customize=add_line_colorbar,
        yscale="log",
    ),
    OneComponentOneVariable(
        "xi",
        r"$\xi$",
        plot_coords=[3, 5],
        compute=massloss,
        yscale="log",
        # bounds=[-1e-2, 0],
        # ymin=-1e-2,
        # ymax=1e-2,
    ),
    MapMovie2D(
        "Rm",
        "$\\mathrm{Rm}_\\mathrm{O}$",
        plot_coords=[0, 6],
        norm="log",
        compute=RmO,
        bounds=[1e-1, 1e12],
        style_kwargs={"cmap": "inferno"},
    ),
    MapMovie2D(
        "ElA",
        "$\\Lambda_\\mathrm{A}$",
        plot_coords=[1, 6],
        compute=ElA,
        bounds=[1e-1, 1e12],
        style_kwargs={"cmap": "inferno"},
    ),
    LocalQuantity(
        "RHO_local",
        localkey="beta",
        uids=uids,
        plot_coords=[2, 6],
        yscale="log",
    ),
]
for qty in quantities:
    qty.uids = uids
    qty.parts_color = colors
fig1 = Fig(localquantities + quantities)
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(
    runContext,
    [fig1],
    zoom=zoom,
    # scatter_particles=True,
)

pipeline.run()
