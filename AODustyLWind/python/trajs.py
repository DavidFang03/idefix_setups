import matplotlib.pyplot as plt
import numpy as np
from idefix2python import readVTK
import glob
from pathlib import Path
import inifix
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d

plt.style.use("dark_background")

parts_cmap = plt.get_cmap("cool")

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

def colors(vtk):
    sizes = size(vtk)
    norm = LogNorm(vmin=np.min(sizes), vmax=np.max(sizes))

    return parts_cmap(norm(sizes))

def size(v):
    beta = v.data["DRAGCOEFF"]
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos

iniPath = Path(
    "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/inputs_v2/dw100_v2_thin_b1e4_2000p.ini"
)
with iniPath.open("rb") as fh:
    inidata = inifix.load(fh, sections="require")

frequency = 1
datas_vtk = sorted(
    glob.glob(
        "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v2/dw100_v2_thin_b1e4_2000p/vtks/data*vtk"
    )
)[::frequency]
parts_vtk = sorted(
    glob.glob(
        "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v2/dw100_v2_thin_b1e4_2000p/vtks/part*vtk"
    )
)[::frequency]

num_r = int(inidata["Particles"]["num_r"][0])
num_theta = int(inidata["Particles"]["num_theta"][0])
num_size = int(inidata["Particles"]["num_size"][0])

uids_grid = np.arange(num_r * num_theta * num_size).reshape(num_r, num_theta, num_size)
same_r = uids_grid[5, :, :].flatten()
same_angle = uids_grid[:, 3, :].flatten()
same_size = uids_grid[:, :, 10:20].flatten()
same_pos = [uid for uid in same_r if uid in same_angle]
uids = list(same_pos)


def extract_particle_trajectory(parts_vtk_paths, target_uid):
    """Extracts time-series data: Cartesian coordinates, phi, t
    for one single UID.

    Parameters:
    -----------
    parts_vtk_paths : list of str or Path
        Sorted file paths to the particle VTK files.
    target_uid : int
        The unique identifier (UID) of the particle to track.

    Returns:
    --------
    px, py, pz, pphi, t : ndarray
    """
    px, py, pz, t = [], [], [], []

    for file_path in parts_vtk_paths:
        pv = readVTK(file_path)
        uids_array = np.array(pv.data["uid"])
        t.append(pv.t[0])
        if target_uid in uids_array:
            idx = np.where(uids_array == target_uid)[0][0]
            # Pull keys directly from particle dictionary
            r = pv.r[idx]
            theta = pv.theta[idx]
            phi = pv.data["phi"][idx]

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            px.append(x)
            py.append(y)
            pz.append(z)
        else:
            # Fallback to last known position if particle is temporarily missing
            px.append(px[-1] if px else 0.0)
            py.append(py[-1] if py else 0.0)
            pz.append(pz[-1] if pz else 0.0)

    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)
    pphi = np.arctan2(py, px)
    t = np.array(t)

    return px, py, pz, pphi, t


def extract_gas_slice(data_vtk_path):
    """Extracts the underlying 2D cylindrical grid mesh (R, Z) and the
    2D density (RHO) field slice from a single fluid VTK snapshot.

    Parameters:
    -----------
    data_vtk_path : str or Path
        File path to a specific gas data VTK file.

    Returns:
    --------
    R, Z, rho_slice : ndarray
    """
    dv = readVTK(data_vtk_path)

    # 1D coordinate arrays from the grid geometry
    r_grid = dv.r
    theta_grid = dv.theta

    # Generate the 2D grid structure
    R, Theta = np.meshgrid(r_grid, theta_grid)

    X, Z = R * np.sin(Theta), R * np.cos(Theta)

    # Slices the 3D data array [Z, PHI, R] at the first azimuthal index
    rho_slice = dv.data["RHO"][:, :, 0].T

    return X, Z, rho_slice


dv_initial = readVTK(datas_vtk[0])
pv_initial = readVTK(parts_vtk[0])
R, Theta = np.meshgrid(dv_initial.r, dv_initial.theta)

uid = uids[0]
c = colors(pv_initial)[uid]
s = size(pv_initial)[uid]
# Simple, direct definitions mapping to your physical dimensions
X = R * np.sin(Theta)
Z = R * np.cos(Theta)
# px, py, pz, pphi, t = extract_particle_trajectory(parts_vtk, uid)
fig, axs = plt.subplots()
# axs.plot(px, pz)
axs.pcolormesh(X, Z, extract_gas_slice(datas_vtk[0])[2])
# fig.savefig("test.png")



def render_particle_and_gas_slice(
    px,
    py,
    pz,
    pphi,
    X,
    Z,
    rho_slices,
    save_path="trajectory_fixed.mp4",
    disable_background=False,
    disable_trajectory=False,
):
    """Animates a particle's trajectory through a 3D projected gas density slice.

    Features can be turned on/off using the boolean flags.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Feature Toggle: Trajectory Setup
    trail_line, head_point = None, None
    if not disable_trajectory:
        (trail_line,) = ax.plot(
            [], [], [], color=c, linewidth=2, label=f"$s={float_to_latex(s)} \\,\\mathrm{{m}}$", zorder=5
        )
        (head_point,) = ax.plot(
            [], [], [], color=c, marker="o", markersize=6, zorder=6
        )

    # Container to clear old surfaces frame-by-frame
    surface_container = [None]

    def init():
        artists = []
        if not disable_trajectory:
            trail_line.set_data([], [])
            trail_line.set_3d_properties([])
            head_point.set_data([], [])
            head_point.set_3d_properties([])
            artists.extend([trail_line, head_point])
        return artists

    def update(frame):
        artists = []

        # Feature Toggle: Gas Slice (Background) Animation
        if not disable_background:
            if surface_container[0] is not None:
                surface_container[0].remove()

            # Rotate the 2D grid plane by the particle's current azimuthal angle (phi)
            phi = pphi[frame]
            X_rot = X * np.cos(phi)
            Y_rot = X * np.sin(phi)

            # Normalize density for the color map
            rho = rho_slices[frame]
            # print(np.shape(rho))
            # print(np.shape(X_rot))
            # print(np.shape(Y_rot))
            # print(np.shape(Z))

            vmin = max(rho.min(), 1e-10)
            vmax = max(rho.max(), vmin + 1e-10)

            sm = plt.cm.ScalarMappable(
                cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax)
            )

            colors = sm.to_rgba(rho)
            sm = plt.cm.ScalarMappable(
                cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax)
            )

            surface_container[0] = ax.plot_surface(
                X_rot,
                Y_rot,
                Z,
                facecolors=colors,
                shade=False,
                alpha=0.7,
                edgecolor="none",
                antialiased=False,
                # rstride=4,  # Downsamples row execution for rendering speed and smoothness
                # cstride=4,  # Downsamples column execution to kill the Moiré patterns
            )
        # Feature Toggle: Trajectory Line Update
        if not disable_trajectory:
            trail_line.set_data(px[: frame + 1], py[: frame + 1])
            trail_line.set_3d_properties(pz[: frame + 1])

            head_point.set_data([px[frame]], [py[frame]])
            head_point.set_3d_properties([pz[frame]])
            artists.extend([trail_line, head_point])
        
        ax.set_title(f"${float_to_latex(t[frame])}$ yr")


        return artists

    # Set scaling limits based on maximum coordinate extent
    # max_bound = max(np.max(np.abs(px)), np.max(np.abs(py)), np.max(np.abs(pz)))
    max_bound = 100
    ax.set_xlim([-max_bound, max_bound])
    ax.set_ylim([-max_bound, max_bound])
    ax.set_zlim([-max_bound, max_bound])
    ax.set_xlabel(r"$x$ [au]")
    ax.set_ylabel(r"$y$ [au]")
    ax.set_zlabel(r"$z$ [au]")
    ax.set_aspect("equal")
    if not disable_trajectory:
        ax.legend(loc="upper right")

    ani = FuncAnimation(
        fig, update, frames=len(px), init_func=init, blit=False, interval=50
    )

    print("Writing rendering engine outputs to disk...")
    ani.save(save_path, writer="ffmpeg", fps=60/frequency, dpi=150)
    plt.close(fig)
    print(f"Done! Saved file to {save_path}")


if __name__ == "__main__":
    target_particle_uid = uids[0]

    # --- PIPELINE CONFIGURATION FLAGS ---
    DISABLE_BG = False
    DISABLE_TRAJ = False
    # ------------------------------------

    # 1. Trajectory Data
    px, py, pz, pphi, t = extract_particle_trajectory(parts_vtk, target_particle_uid)

    # 2. Grid & Gas Data
    dv_initial = readVTK(datas_vtk[0])
    R, Theta = np.meshgrid(dv_initial.r, dv_initial.theta)

    # Simple, direct definitions mapping to your physical dimensions
    X = R * np.sin(Theta)
    Z = R * np.cos(Theta)

    rho_slices = []
    for file_path in datas_vtk:
        _, _, frame_rho = extract_gas_slice(file_path)
        rho_slices.append(frame_rho)

    # 3. Animation Generation
    render_particle_and_gas_slice(
        px,
        py,
        pz,
        pphi,
        X=X,
        Z=Z,
        rho_slices=rho_slices,
        disable_background=DISABLE_BG,
        disable_trajectory=DISABLE_TRAJ,
    )
