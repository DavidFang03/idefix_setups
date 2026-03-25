import os
import sys
from multiprocessing import Pool

sys.path.append(os.getenv("IDEFIX_DIR"))
# from pytools.vtk_io import readVTK
from vtk_io import readVTK
from myidefixtools import dat_to_dict
import numpy as np
import matplotlib.pyplot as plt
import glob
import ffmpeg

# PathToSimulation = (
# "/mnt/lustre/tursafs1/home/dp316/dp316/dc-fang1/idefix/me/MRIShearingBoxNetFlux"
# )

# PathToSimulation = "/mnt/lustre/tursafs1/home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n8_res2x"


def fit(X, Y, deg, start=0, end=1):
    index_start = int(start * (len(X) - 1))
    index_end = int(end * (len(X) - 1))
    params, cov = np.polyfit(
        X[index_start:index_end], Y[index_start:index_end], deg=deg, cov=True
    )
    return params, np.diag(cov)


def get_bounds(vtk_list, field):
    """
    Get the bounds (min, max) of one given field during the run

    vtk_list    List of dump file paths
    field       Field
    """
    bound_up = -np.inf
    bound_low = np.inf
    for vtk_dump in vtk_list:
        V = readVTK(vtk_dump, geometry="cartesian")
        data = V.data[field]
        if np.max(data) > bound_up:
            bound_up = np.max(data)
        if np.min(data) < bound_low:
            bound_low = np.min(data)
    return (bound_low, bound_up)


def movie(folder_path, movie_path, fps=10):
    pattern_png = f"{folder_path}/*.png"
    ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
        movie_path,
        vcodec="libx264",
        crf=18,
        preset="medium",
        r=fps,
        pix_fmt="yuv420p",
        movflags="faststart",
    ).overwrite_output().run()
    print(f"[OK] {movie_path}")


class RUN:
    def __init__(self, PathToSimulation, end=1):
        """
        If there are n dumps, and end=0.5, only 0.5n of the dumps will be read.
        """
        self.PathToSimulation = PathToSimulation
        self.name = os.path.basename(self.PathToSimulation)
        self.slice1_folder = f"{self.PathToSimulation}/frames/slice1"

        self.analysis = dat_to_dict(f"{PathToSimulation}/timevol.dat", end=end)

        self.slice_list = glob.glob(f"{self.PathToSimulation}/vtks/slice1*.vtk")
        slices_nb = len(self.slice_list)
        self.slice_list[: int(end * slices_nb)]

        self.vtk_list = glob.glob(f"{self.PathToSimulation}/vtks/data*.vtk")
        vtks_nb = len(self.vtk_list)
        self.vtk_list[: int(end * vtks_nb)]

        self.bounded = False

        folders = ["frames/slice1", "videos"]
        self.videos_folder = f"{PathToSimulation}/videos"
        for subfolder in folders:
            path = f"{PathToSimulation}/{subfolder}"
            try:
                os.makedirs(path)
            except OSError as error:
                content = glob.glob(f"{path}/*")
                # user_agree = input(
                #     f"Will overwrite the {subfolder} folder ({len(content)} files) [o/r/n] (overwrite, remove, no)"
                # )
                # if user_agree == "r":
                #     for f in content:
                #         os.remove(f)
                # elif user_agree == "n":
                #     exit()

    def print_available_quantities(self):
        print("------ Available quantities ------")
        print("Analysis:", self.analysis.keys())
        first_vtkslice = readVTK(self.slice_list[0], geometry="cartesian")
        print("VTK (slice):", first_vtkslice.data.keys())
        first_vtk = readVTK(self.vtk_list[0], geometry="cartesian")
        print("VTK:", first_vtk.data.keys())

    def plot_analysis(self, start_fit=0, end_fit=1):
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle(self.PathToSimulation)
        fig.subplots_adjust(left=0.1, right=1 - 0.1, hspace=0.5, wspace=0.2)
        a = self.analysis
        t = a["t"]
        axs[0, 0].plot(a["t"], a["kinx"], label="kinx")
        axs[0, 0].plot(a["t"], a["kiny"], label="kiny")
        axs[0, 0].plot(a["t"], a["kinz"], label="kinz")
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_xlabel(r"$t$")
        axs[0, 0].set_ylabel(r"$E$")

        kin = a["kinx"] + a["kiny"] + a["kinz"]
        params, uncertainties = fit(t, np.log(kin), start=start_fit, end=end_fit, deg=1)
        akin, bkin = params
        print(f"{self.name:>20}: {akin:.2e} pm {uncertainties[0]:.1e}")
        axs[1, 0].plot(t, kin, label="kin")
        axs[1, 0].plot(
            t, np.exp(akin * t + bkin), label=f"fit $\\exp({akin:.1e}t + {bkin:.1e})$"
        )
        axs[1, 0].set_yscale("log")
        axs[1, 0].set_xlabel(r"$t$")
        axs[1, 0].set_ylabel(r"$E$")

        params, uncertainties = fit(
            t, np.log(a["bx"]), start=start_fit, end=end_fit, deg=1
        )
        abx, bbx = params
        # print(f"{self.name:>20}: {abx:.1e} pm {uncertainties[0]:.1e}")
        axs[0, 1].plot(a["t"], a["by"], label="bx", alpha=0.6)
        axs[0, 1].plot(a["t"], a["by"], label="by", alpha=0.4)
        axs[0, 1].plot(a["t"], a["bz"], label="bz")
        axs[0, 1].plot(
            a["t"],
            np.exp(abx * t + bbx),
            label=f"fit $\\exp({abx:.1e}t + {bbx:.1e})$",
            ls="--",
        )
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_xlabel(r"$t$")
        axs[0, 1].set_ylabel(r"$<B_i^2>$")

        axs[1, 1].plot(
            a["t"], a["bxby"] / (4 * np.pi), label=r"$\frac{B_xB_y}{4 \pi}$", ls="--"
        )
        axs[1, 1].plot(a["t"], a["rhovrvphi"], label=r"$\rho v_r v_\phi$", ls="--")
        axs[1, 1].plot(
            a["t"],
            np.sum([a["bxby"] / (4 * np.pi), a["rhovrvphi"]], axis=0),
            label=r"$\alpha$",
        )
        # axs[1, 1].plot(
        #     a["t"],
        #     np.exp(abx * t + bbx),
        #     label=f"fit $\\exp({abx:.1e}t + {bbx:.1e})$",
        #     ls="--",
        # )
        # axs[1, 1].set_yscale("log")
        axs[1, 1].set_xlabel(r"$t$")
        axs[1, 1].set_ylabel(r"$alpha$ components")

        for axsl in axs:
            for ax in axsl:
                ax.legend()
                ax.grid()
        image_path = f"{self.PathToSimulation}/analysis.png"
        fig.savefig(image_path)
        print(f"[OK] {image_path}")

    def slice_to_png(self, slice1_path, bounds_dict):
        fig, axs = plt.subplots(3, 3, figsize=(16, 16))
        fig.subplots_adjust(left=0.1, right=1 - 0.1, hspace=0.5, wspace=0.2)
        print(slice1_path)
        V = readVTK(slice1_path, geometry="cartesian")
        print(np.shape(V.data["BX3"][0, :, 0]))
        midplane_index = int(len(V.data["BX3"][0, :, 0]) / 2)
        print("midplane_index", midplane_index)  # no midplane but anyway

        X = V.x
        Y = V.y
        Z = V.z
        time = V.t[0]
        fig.suptitle(f"{self.PathToSimulation} $t={time:.1e}$")

        for i, dir in enumerate(["VX1", "VX2", "VX3", "BX1", "BX2", "BX3", "RHO"]):
            row = i % 3
            column = i // 3
            data = np.transpose(V.data[dir][:, midplane_index, :])
            map = axs[row, column].pcolormesh(
                X,
                Z,
                data,
                vmin=bounds_dict[dir][0],
                vmax=bounds_dict[dir][1],
                shading="nearest",
            )
            fig.colorbar(map, ax=axs[row, column])
            axs[row, column].set_xlabel(r"$x$")
            axs[row, column].set_ylabel(r"$z$")
            axs[row, column].set_title(rf"${dir}$")

        for row in axs:
            for ax in row:
                ax.set_aspect("equal", adjustable="box")
        slice1_name = os.path.basename(slice1_path)
        slice1_png_path = f"{self.slice1_folder}/{slice1_name[:-4]}.png"
        fig.savefig(slice1_png_path)
        plt.close(fig)

    def plot_slice(self, jump_to_movie=False):

        if not jump_to_movie:
            bounds_dict = {}
            for qt in ["RHO", "BX1", "BX2", "BX3", "VX1", "VX2", "VX3"]:
                bounds_dict[qt] = (
                    get_bounds(self.slice_list, qt) if self.bounded else [None, None]
                )

            for slice1_path in self.slice_list:
                self.slice_to_png(slice1_path, bounds_dict)
        movie(
            self.slice1_folder,
            f"{self.videos_folder}/slice1{'_bounded' if self.bounded else ''}.mp4",
        )


def do_task(task):
    PathToSimulation = (
        f"/home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/{task}"
    )

    run = RUN(PathToSimulation, end=0.3)
    # run.print_available_quantities()
    run.plot_analysis(start_fit=0.1, end_fit=0.4)
    run.plot_slice(jump_to_movie=False)


if __name__ == "__main__":
    sequential = False
    # run = RUN(PathToSimulation, end=0.3)
    # run.print_available_quantities()
    # run.plot_analysis(start_fit=0.1, end_fit=0.4)
    # run.plot_slice(jump_to_movie=False)

    tasks = [
        # "MRI_mode_n2",
        "MRI_mode_n2_res2x",
        # "MRI_mode_n4",
        "MRI_mode_n4_res2x",
        # "MRI_mode_n6",
        "MRI_mode_n6_res2x",
        # "MRI_mode_n8",
        "MRI_mode_n8_res2x",
        # "MRI_mode_n10",
        "MRI_mode_n10_res2x",
        # "MRI_mode_n12",
        "MRI_mode_n12_res2x",
    ]
    if sequential:
        for task in tasks:
            do_task(task)
    else:
        with Pool(len(tasks)) as pool:
            pool.map(do_task, tasks)
# plt.show()
# Streamlines
# The formula needed to transform vectors in spherical coordinates into cartesian ones is
# BX1*(coordsX*iHat+coordsY*jHat+coordsZ*kHat)/sqrt(coordsX^2+coordsY^2+coordsZ^2) + BX2*(coordsX*coordsZ*iHat+coordsY*coordsZ*jHat-(coordsX^2+coordsY^2)*kHat)/sqrt((coordsX^2+coordsY^2+coordsZ^2)*(coordsX^2+coordsY^2)) + BX3*(coordsX*jHat-coordsY*iHat)/sqrt(coordsX^2+coordsY^2)
