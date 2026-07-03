import glob
import math
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import inifix
from cmap import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from idefix2python import readVTK

EXT = "png"
CMAP_MESH = "viridis"
CMAP_SLOPE = Colormap("chrisluts:bop_blue").to_mpl()

CBFORMAT = matplotlib.ticker.ScalarFormatter()
CBFORMAT.set_scientific("%.2e")
CBFORMAT.set_powerlimits((-2, 12))
CBFORMAT.set_useMathText(True)

plt.rcParams.update({"text.usetex": True})
plt.style.use("dark_background")

root_name = "dw100_v2_thin"


METRIC_CONFIG = {
    "0": {"label": r"$\beta_\mathrm{mid}$", "legend": r"\beta_\mathrm{{mid}}"},
    "A": {"label": r"$\beta_\mathrm{local}$", "legend": r"\beta_\mathrm{{local}}"},
    "B": {"label": r"$1/B^2$", "legend": r"{1/B^2}"},
    "C": {"label": r"${u_\mathrm{p}^2}$", "legend": r"{u_\mathrm{p}^2}"},
    "D": {"label": r"$r$ [au]", "legend": r"r"},
}

# "0", "A", "B", "C", or "D"
ACTIVE_OPTION = "B"

X_AXIS_LABEL = METRIC_CONFIG[ACTIVE_OPTION]["label"]
X_AXIS_LEGEND_STRING = METRIC_CONFIG[ACTIVE_OPTION]["legend"]


def get_x_axis_metric(r, theta, hydro_vtk, betamid):
    """Calculates the target property for the X-axis using pre-loaded hydro or model data."""
    if ACTIVE_OPTION == "0":
        # Convert string representations like '1e4' into floats
        return float(betamid)

    # For local metrics, pull fluid snapshot attributes
    p, b2, vp2, rho = get_local_hydro_data(r, theta, hydro_vtk)

    if np.isnan(p) or b2 == 0:
        return np.nan

    if ACTIVE_OPTION == "A":
        return 8 * np.pi * p / b2
    elif ACTIVE_OPTION == "B":
        return 1 / b2
    elif ACTIVE_OPTION == "C":
        return vp2
    elif ACTIVE_OPTION == "D":
        return r
    else:
        return np.nan


######################################################################


def get_vtks_path(betamid):
    return f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v2/{root_name}_b{betamid}_2000p/vtks"


def colorbar(mappable, loc="right"):
    """Appends an aligned colorbar to the current axes."""
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size="4%", pad="5%")
    cbar = fig.colorbar(mappable, cax=cax, location=loc, format=CBFORMAT)
    plt.sca(last_axes)
    return cbar


def float_to_latex(num: float) -> str:
    """Converts a float into a LaTeX formatted string."""
    if num == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(num))))
    mantissa = num / (10**exponent)
    mantissa = int(mantissa) if mantissa.is_integer() else round(mantissa, 4)

    if mantissa == 1:
        return f"10^{{{exponent}}}"
    if mantissa == -1:
        return f"-10^{{{exponent}}}"
    return f"{mantissa} \\cdot 10^{{{exponent}}}"


def get_theta_trajectory(vtklist, target_uids):
    """Securely extracts trajectory paths matching target particle unique IDs dynamically."""
    if not target_uids:
        return np.array([]), []

    max_uid = max(target_uids)
    thetas = []
    times = []

    # Target indices safely sampled across simulation snapshots
    sampled_paths = vtklist[::10]
    if vtklist[-1] not in sampled_paths:
        sampled_paths.append(vtklist[-1])

    for vtkpath in sampled_paths:
        vtk = readVTK(vtkpath)
        uids = vtk.data["uid"]

        # Build dynamic runtime mapping for this exact snapshot file
        current_uid_map = {uid: idx for idx, uid in enumerate(uids) if uid <= max_uid}

        thetas_t = np.full(max_uid + 1, np.nan)
        for uid in target_uids:
            if uid in current_uid_map:
                idx = current_uid_map[uid]
                thetas_t[uid] = vtk.theta[idx]

        times.append(vtk.t[0])
        thetas.append(thetas_t)

    return np.asarray(thetas), times


def calculate_cell_critical_size(second_vtk, cell_uids, final_thetas, Hideal, epsilon):
    """Finds the critical Stokes inside a single spatial cell."""
    corona_angle = np.arctan(Hideal * epsilon)

    sizes = []
    angles = []
    uids_found = []
    all_stokes = {}

    sec_uid_map = {uid: idx for idx, uid in enumerate(second_vtk.data["uid"])}

    for uid in cell_uids:
        if uid not in sec_uid_map:
            continue
        if uid >= len(final_thetas) or np.isnan(final_thetas[uid]):
            continue

        idx_sec = sec_uid_map[uid]
        R = second_vtk.r[idx_sec] * np.sin(second_vtk.theta[idx_sec])
        stokes = second_vtk.data["DRAGCOEFF"][idx_sec]
        # stokes = second_vtk.data["TSTOP"][idx_sec] * (R**-1.5)

        # rho0 = 6.0e-10
        # rhos = 1.0
        # au = 1.5e11
        # stokes = second_vtk.data["DRAGCOEFF"][idx_sec] * (rho0 * au) / rhos

        all_stokes[uid] = stokes
        sizes.append(stokes)
        angles.append(np.pi / 2 - final_thetas[uid])
        uids_found.append(uid)

    if not sizes:
        return np.nan, None, {}

    angles = np.array(angles)
    sizes = np.array(sizes)
    uids_found = np.array(uids_found)

    sort_idx = np.argsort(np.abs(angles - corona_angle))
    crit_uid = uids_found[sort_idx[0]] if len(sort_idx) > 0 else None

    if (
        len(sort_idx) > 1
        and (angles[sort_idx[1]] - corona_angle) * (angles[sort_idx[0]] - corona_angle)
        < 0
    ):
        result = (sizes[sort_idx[0]] + sizes[sort_idx[1]]) / 2
    elif len(sort_idx) > 0:
        result = sizes[sort_idx[0]]
    else:
        result = np.nan

    return result, crit_uid, all_stokes


def plot_and_annotate_lines(ax, Xline, Hideal, epsilon):
    """
    Plots lines relative to Hideal and annotates them
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
            alpha=0.3,
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
            color="white",
        )


def massloss(v, rmin=15, rmax=30, angle=None):
    dR = np.diff(v.rl)
    dTheta = np.diff(v.thetal)
    RLine = v.r
    ThetaLine = v.theta
    R, Theta = np.meshgrid(RLine, ThetaLine)
    rho, prs = v.data["RHO"][:, :, 0].T, v.data["PRS"][:, :, 0].T
    vr, vtheta = v.data["VX1"][:, :, 0].T, v.data["VX2"][:, :, 0].T
    vz = np.cos(Theta) * vr - np.sin(Theta) * vtheta
    if angle is None:
        angle = np.atan(5 * 0.05)
    thetam4h = np.pi / 2 - angle
    thetap4h = np.pi / 2 + angle
    jm4h = np.searchsorted(ThetaLine, thetam4h)
    jp4h = np.searchsorted(ThetaLine, thetap4h)

    ir10 = np.searchsorted(RLine, rmin)
    ir50 = np.searchsorted(RLine, rmax)
    jmid = np.searchsorted(ThetaLine, np.pi / 2)

    xiup = np.sum(rho[jp4h, ir10:ir50] * vz[jp4h, ir10:ir50] * dR[ir10:ir50])
    # xidown = np.sum(rho[jm4h, ir10:ir50] * vz[jm4h, ir10:ir50] * dR[ir10:ir50])

    xidown = 0
    norm = 2 * np.sum(rho[jmid, :] * np.sqrt(prs[jmid, :] / rho[jmid, :]) * dR)

    # norm = (
    #     -2
    #     * np.pi
    #     * R[ir50]
    #     * np.sum(
    #         rho[jm4h:jp4h, ir50] * vr[jm4h:jp4h, ir50] * R[ir50] * dTheta[jm4h:jp4h]
    #     )
    # )
    # norm = 1

    xi = (xiup - xidown) / norm
    # print(xi)

    return np.abs(xi)


def get_local_hydro_data(r, theta, hydro_vtk):
    r_mesh = np.atleast_1d(hydro_vtk.r)
    th_mesh = np.atleast_1d(hydro_vtk.theta)

    if r_mesh.ndim == 1 and th_mesh.ndim == 1:
        R, TH = np.meshgrid(r_mesh, th_mesh, indexing="ij")
    else:
        R, TH = r_mesh, th_mesh

    distance = (R - r) ** 2 + (TH - theta) ** 2
    idx_2d = np.unravel_index(np.argmin(distance), distance.shape)

    i, j = idx_2d[0], idx_2d[1]

    p = hydro_vtk.data["PRS"][i, j, 0]
    b2 = (
        hydro_vtk.data["BX1"][i, j, 0] ** 2
        + hydro_vtk.data["BX2"][i, j, 0] ** 2
        + hydro_vtk.data["BX3"][i, j, 0] ** 2
    )
    vp2 = (
        # hydro_vtk.data["VX1"][i, j, 0] ** 2
        +(hydro_vtk.data["VX2"][i, j, 0] ** 2)
        # + hydro_vtk.data["VX3"][i, j, 0] ** 2
    )
    rho = hydro_vtk.data["RHO"][i, j, 0] ** 2
    return p, b2, vp2, rho


def run_analysis():
    betas = ("7e3", "1e4", "4e4", "6e4", "8e4")

    iniPath = Path(
        f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/inputs_v2/{root_name}_b{betas[0]}_2000p.ini"
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
    r_range = range(2, num_r)
    theta_range = range(num_theta)

    print("Building geometry baseline layout from master reference...")
    ref_paths = sorted(glob.glob(f"{get_vtks_path(betas[0])}/part.*.vtk"))
    master_first_vtk = readVTK(ref_paths[0])
    master_uid_to_idx = {
        uid: idx for idx, uid in enumerate(master_first_vtk.data["uid"])
    }

    X_grid = np.zeros((len(r_range), len(theta_range)))
    Z_grid = np.zeros((len(r_range), len(theta_range)))
    r_grid = np.zeros((len(r_range), len(theta_range)))
    theta_grid = np.zeros((len(r_range), len(theta_range)))

    for i, r in enumerate(r_range):
        for j, theta in enumerate(theta_range):
            cell_uids = uids_grid[r, j, :].flatten()
            valid_uid = None
            for uid in cell_uids:
                if uid in master_uid_to_idx:
                    valid_uid = uid
                    break

            if valid_uid is not None:
                idx = master_uid_to_idx[valid_uid]
                r_val = master_first_vtk.r[idx]
                th_val = master_first_vtk.theta[idx]

                r_grid[i, j] = r_val
                theta_grid[i, j] = th_val
                X_grid[i, j] = r_val * np.sin(th_val)
                Z_grid[i, j] = r_val * np.cos(th_val)
            else:
                r_grid[i, j], theta_grid[i, j], X_grid[i, j], Z_grid[i, j] = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )

    table_data = np.zeros((len(betas), len(r_range), len(theta_range)))
    x_metrics_data = np.zeros((len(betas), len(r_range), len(theta_range)))

    # 1: Extract unique ids
    for b_idx, beta in enumerate(betas):
        print(f"Processing beta model loop reference: {beta}...")
        vtks_path = get_vtks_path(beta)

        # Load hydro snapshots exactly ONCE per beta setup to remove nested disk I/O loops
        hydro_paths = sorted(glob.glob(f"{vtks_path}/data.*.vtk"))
        active_hydro_vtk = readVTK(
            hydro_paths[1] if len(hydro_paths) > 1 else hydro_paths[0]
        )

        paths = sorted(glob.glob(f"{vtks_path}/part.*.vtk"))
        last_vtk = readVTK(paths[-1])
        second_vtk = readVTK(paths[1])

        active_uids = set(last_vtk.data["uid"]).intersection(master_uid_to_idx.keys())
        thetas_history, times = get_theta_trajectory(paths, list(active_uids))

        # Handle cases where active_uids returned empty configurations
        if thetas_history.size > 0:
            final_thetas = thetas_history[-1]
        else:
            final_thetas = np.array([])

        for i, r in enumerate(r_range):
            for j, theta in enumerate(theta_range):
                r_ini = r_grid[i, j]
                theta_ini = theta_grid[i, j]

                if np.isnan(r_ini):
                    table_data[b_idx, i, j] = np.nan
                    x_metrics_data[b_idx, i, j] = np.nan
                    continue

                # Store structural parameters inside grid records
                # x_metrics_data[b_idx, i, j] = massloss(active_hydro_vtk)
                x_metrics_data[b_idx, i, j] = get_x_axis_metric(
                    r_ini, theta_ini, active_hydro_vtk, betamid=beta
                )

                p, b2, vp2, rho = get_local_hydro_data(r, theta, active_hydro_vtk)

                cell_uids = uids_grid[r, j, :].flatten()
                size, crit_uid, all_stokes = calculate_cell_critical_size(
                    second_vtk, cell_uids, final_thetas, Hideal, epsilon
                )
                table_data[b_idx, i, j] = size  ####### HEEEEEEEEEEEEEEEEEEERE

                # Debugging Trajectory Profiler
                if (
                    i % 2 == 0
                    and j % 2 == 0
                    and crit_uid is not None
                    and final_thetas.size > 0
                ):
                    alt_ini = np.tan(np.pi / 2 - theta_ini) / epsilon

                    if alt_ini < 5.0 or r_ini < 15.0:
                        continue

                    fig_db, ax_db = plt.subplots(figsize=(7, 4))
                    corona_angle = np.arctan(Hideal * epsilon)
                    ax_db.axhline(
                        corona_angle,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Corona Boundary ({Hideal}H)",
                    )

                    for uid in cell_uids:
                        if (
                            uid in all_stokes
                            and uid < len(final_thetas)
                            and not np.isnan(final_thetas[uid])
                        ):
                            traj = np.pi / 2 - thetas_history[:, uid]
                            if uid == crit_uid:
                                ax_db.plot(
                                    times,
                                    traj,
                                    color="blue",
                                    linewidth=2.5,
                                    zorder=5,
                                    label=f"CRITICAL ($St_0$={all_stokes[uid]:.2e})",
                                )
                            else:
                                ax_db.plot(
                                    times,
                                    traj,
                                    color="gray",
                                    alpha=0.3,
                                    linewidth=0.8,
                                )

                    ax_db.set_title(
                        rf"$\beta_\mathrm{{mid}}$: {beta} | Position: $({r_ini:.1f}\,\mathrm{{au}}, {alt_ini:.1f}H)$ | $St_\mathrm{{crit}}$={size:.2e}"
                    )
                    ax_db.set_xlabel("Time")
                    ax_db.set_ylabel(r"Altitude Angle $\left(\pi/2 - \theta\right)$")
                    ax_db.legend(loc="upper left", fontsize=8)
                    ax_db.grid(True, linestyle=":", alpha=0.5)

                    debug_path = f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/debug_angles_beta{beta}_cell_{i}_{j}.png"
                    fig_db.savefig(debug_path, bbox_inches="tight", dpi=150)
                    plt.close(fig_db)

    # 2: Fit and plot
    print("Generating global trend lines and continuous power laws...")
    fig_lines, ax_lines = plt.subplots(figsize=(9, 8))
    grid_slope = np.full(X_grid.shape, np.nan)

    for i in range(len(r_range)):
        for j in range(len(theta_range)):
            r_ini = r_grid[i, j]
            theta_ini = theta_grid[i, j]

            if np.isnan(r_ini):
                continue

            alt_ini = np.tan(np.pi / 2 - theta_ini) / epsilon
            if alt_ini < 5.0 or r_ini < 15.0:
                continue

            list_sizes = []
            x_metrics = []
            is_valid = True

            for b_idx, beta in enumerate(betas):
                sz = table_data[b_idx, i, j]
                xm = x_metrics_data[b_idx, i, j]

                if np.isnan(sz) or sz <= 0 or np.isnan(xm) or xm <= 0:
                    is_valid = False
                    break
                list_sizes.append(sz)
                x_metrics.append(xm)

            if not is_valid or len(x_metrics) < 2:
                continue

            # Logarithmic optimization curve fit
            a, b = np.polyfit(np.log(x_metrics), np.log(list_sizes), deg=1)
            grid_slope[i, j] = a

            if i % 2 == 0 and j % 2 == 0:
                expb = np.exp(b)
                sorted_idx = np.argsort(x_metrics)
                xm_sorted = np.array(x_metrics)[sorted_idx]
                fit_line = (xm_sorted**a) * expb

                (line,) = ax_lines.plot(
                    xm_sorted, fit_line, linestyle="-", linewidth=1.5
                )
                ax_lines.scatter(
                    x_metrics, list_sizes, color=line.get_color(), s=35, zorder=3
                )

                line.set_label(
                    rf"Init: $({r_ini:.1f},{alt_ini:.1f}H) \, \sigma={a:.2f}$"
                )

    ax_lines.set_xlabel(X_AXIS_LABEL, fontsize=12)
    ax_lines.set_ylabel(r"$s_\mathrm{crit}$ [m]", fontsize=12)
    ax_lines.set_xscale("log")
    ax_lines.set_yscale("log")
    ax_lines.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_lines.legend(loc="upper right", fontsize=7, ncol=2, frameon=True)

    lines_path = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/sizes_beta.png"
    fig_lines.savefig(lines_path, dpi=300, bbox_inches="tight")
    plt.close(fig_lines)

    # 3: Map
    print("Generating final grid slope matrix map with text labels...")
    fig_mesh, ax_mesh = plt.subplots(figsize=(10, 6.5))

    X_line = np.linspace(0, 1.15 * np.nanmax(X_grid))
    Y_line = Hideal * X_line * epsilon
    # ax_mesh.plot(X_line, Y_line, color="white", alpha=0.4, linestyle="--")
    # ax_mesh.annotate(
    #     f"${Hideal}H$ (Corona)",
    #     xy=(X_line[-1], Y_line[-1]),
    #     textcoords="offset points",
    #     xytext=(0, 5),
    #     ha="right",
    # )
    plot_and_annotate_lines(ax_mesh, X_line, 5, 0.05)

    mesh = ax_mesh.pcolormesh(
        X_grid, Z_grid, grid_slope, shading="auto", cmap=CMAP_SLOPE, rasterized=True
    )
    cbar = colorbar(mesh)
    cbar.set_label(
        rf"Scaling Index $\sigma$ ($s_\mathrm{{crit}} \propto {X_AXIS_LEGEND_STRING}^\sigma$)"
    )

    for i in range(grid_slope.shape[0]):
        for j in range(grid_slope.shape[1]):
            val = grid_slope[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 5.0 else "white"

                ax_mesh.text(
                    X_grid[i, j],
                    Z_grid[i, j],
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    ax_mesh.set_aspect("equal")
    # ax_mesh.set_xlim(12.5, 1.05 * np.nanmax(X_grid))
    ax_mesh.set_xlim(12.5, 34.5)
    ax_mesh.set_ylim(2, 1.15 * np.nanmax(Z_grid))
    ax_mesh.set_xlabel(r"$x$ [au]")
    ax_mesh.set_ylabel(r"$z$ [au]")
    ax_mesh.set_title(r"Spatial variation of the scaling exponent $\sigma$")

    mesh_path = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/sizes_beta_mesh.png"
    fig_mesh.savefig(mesh_path, dpi=300, bbox_inches="tight")
    plt.close(fig_mesh)

    print(
        f"Analysis saved successfully:\n Lines Plot -> {lines_path}\n Exponent Mesh -> {mesh_path}"
    )


if __name__ == "__main__":
    run_analysis()
