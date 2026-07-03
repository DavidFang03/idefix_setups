import glob
from pathlib import Path
import inifix
from idefix2python import readVTK
import matplotlib
import numpy as np

# Force Matplotlib into an absolute headless math backend BEFORE importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def extract_final_frame_data(parts_vtk_paths, datas_vtk_paths, target_uids):
    """Extracts positions and density matrices only for the last snapshot index."""
    last_data_path = datas_vtk_paths[-1]
    last_part_path = parts_vtk_paths[-1]

    dv = readVTK(last_data_path)
    pv_data = readVTK(last_part_path)

    r_grid = dv.r
    theta_grid = dv.theta
    rho_slice = dv.data["RHO"][:, :, 0].T

    uids_array = np.array(pv_data.data["uid"])
    trajectories = []

    # Map through all requested particle IDs
    for uid in target_uids:
        if uid in uids_array:
            idx = np.where(uids_array == uid)[0][0]
            r = pv_data.r[idx]
            theta = pv_data.theta[idx]
            phi = pv_data.data["phi"][idx]

            trajectories.append({"uid": uid, "r": r, "theta": theta, "phi": phi})

    return r_grid, theta_grid, rho_slice, trajectories


def save_vector_pdf(
    r_grid, theta_grid, rho_slice, trajectories, pdf_path="final_frame_dual.pdf"
):
    """Generates a high-fidelity vector PDF containing multiple oriented density planes."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # Global normalization for consistent colormapping across both surfaces
    vmin = max(rho_slice.min(), 1e-10)
    vmax = rho_slice.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Downsample factor to maintain instantaneous execution speed
    stride = 4
    r_sub = r_grid[::stride]
    theta_sub = theta_grid[::stride]
    rho_sub = rho_slice[::stride, ::stride]

    R, Theta = np.meshgrid(r_sub, theta_sub)
    X_base = R * np.sin(Theta)
    Z_base = R * np.cos(Theta)

    colors = plt.cm.inferno(norm(rho_sub))

    # Distinct colors for tracking Particle 1 vs Particle 2
    particle_colors = ["#FF3333", "#33FF33", "#3333FF"]

    print(f"Processing {len(trajectories)} particle planes...")

    # --- MULTI-PLANE RENDER ENGINE ---
    for idx, traj in enumerate(trajectories):
        phi = traj["phi"]

        # Rotate the base grid framework by this specific particle's phi angle
        X_rot = X_base * np.cos(phi)
        Y_rot = X_base * np.sin(phi)

        # Render this particle's individual oriented fluid density plane
        ax.plot_surface(
            X_rot,
            Y_rot,
            Z_base,
            facecolors=colors,
            shade=False,
            antialiased=False,
            edgecolor="none",
            alpha=0.6,  # Slices are translucent so you can see them intersect
        )

        # Compute Cartesian coordinate for the particle head position
        px = traj["r"] * np.sin(traj["theta"]) * np.cos(phi)
        py = traj["r"] * np.sin(traj["theta"]) * np.sin(phi)
        pz = traj["r"] * np.cos(traj["theta"])

        p_color = particle_colors[idx % len(particle_colors)]
        ax.scatter(
            [px],
            [py],
            [pz],
            color=p_color,
            s=80,
            edgecolors="white",
            zorder=10,
            label=f"Particle UID {traj['uid']}",
        )

    # Spatial bounds configuration
    max_bound = 100
    ax.set_xlim([-max_bound, max_bound])
    ax.set_ylim([-max_bound, max_bound])
    ax.set_zlim([-max_bound, max_bound])

    # Strip default grey borders for a clean presentation look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    ax.legend(loc="upper right", labelcolor="white")

    print(f"Saving crisp vector layout to {pdf_path}...")
    plt.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)


# -------------------------------------------------------------------
# Execution Control Blueprint
# -------------------------------------------------------------------

if __name__ == "__main__":
    iniPath = Path(
        "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/inputs_v2/dw100_v3_thin_tv_b1e4_2000p.ini"
    )
    with iniPath.open("rb") as fh:
        inidata = inifix.load(fh, sections="require")

    frequency = 3
    base_dir = Path(
        "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v2/dw100_v3_thin_tv_b1e4_2000p/vtks"
    )
    datas_vtk = sorted(glob.glob(str(base_dir / "data*vtk")))[::frequency]
    parts_vtk = sorted(glob.glob(str(base_dir / "part*vtk")))[::frequency]

    num_r = int(inidata["Particles"]["num_r"][0])
    num_theta = int(inidata["Particles"]["num_theta"][0])
    num_size = int(inidata["Particles"]["num_size"][0])

    uids_grid = np.arange(num_r * num_theta * num_size).reshape(
        num_r, num_theta, num_size
    )
    same_r = uids_grid[5, :, :].flatten()
    same_angle = uids_grid[:, 3, :].flatten()
    same_pos = [uid for uid in same_r if uid in same_angle]
    uids = list(same_pos)

    # --- LOOK HERE: SIMULTANEOUS TWO PARTICLE TRACKING ---
    # We grab the first two unique identifiers matching your target spatial position criteria
    target_uids = [uids[0], uids[1]]

    # Parse and extract values for the final step slice
    r_grid, theta_grid, rho_slice, trajectories = extract_final_frame_data(
        parts_vtk, datas_vtk, target_uids
    )

    # Generate the dual-plane vector plot
    save_vector_pdf(
        r_grid,
        theta_grid,
        rho_slice,
        trajectories,
        pdf_path="final_frame_dual_planes.pdf",
    )
    print("Headless dual-plane generation successfully finished!")
