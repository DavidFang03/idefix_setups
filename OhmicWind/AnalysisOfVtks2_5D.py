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
from scipy.interpolate import griddata

plt.style.use("dark_background")

# PathToSimulation = (
# "/mnt/lustre/tursafs1/home/dp316/dp316/dc-fang1/idefix/me/OhmicWind"
# )

# PathToSimulation = "/mnt/lustre/tursafs1/home/dp316/dp316/dc-fang1/outputs/OhmicWind/MRI_mode_n8_res2x"


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


class Data_info:
    def __init__(self, test_file_pattern):
        self.test_files = glob.glob(test_file_pattern)
        if len(self.test_files) > 0:
            self.status = True
            self.test_file = self.test_files[0]
        else:
            self.status = False

    def set_data_test(self, handle_test_file):
        """
        How should the test_file be handled to get its data?

        Parameters:
        @handle_test_file: a function f(path)-->data [dict]
        """
        if self.status:
            self.data_test = handle_test_file(self.test_file)


def RequirePath(path, dir_or_file=False):
    if not dir_or_file:
        if not os.path.exists(path):
            raise Exception
    elif dir_or_file == "dir":
        if not os.path.isdir(path):
            raise Exception
    elif dir_or_file == "file":
        if not os.path.isfile(path):
            raise Exception
    else:
        raise Exception("wrong value for dir_or_file")


def streamplot(X, Y, Ux, Uy, resolution=200, method="linear"):
    """
    X,Y must come from meshgrid, Ux and Uy are the data points.
    Returns the 1d arrays x_coords and y-coords, with the interpolated values Ux_uni, Uy_uni (2D arrays) that can go directly in streamplot
    """
    x_min, x_max = np.nanmin(X), np.nanmax(X)
    y_min, y_max = np.nanmin(Y), np.nanmax(Y)

    resolution = 200
    x_coords = x_min + np.arange(resolution) * ((x_max - x_min) / (resolution - 1))
    y_coords = y_min + np.arange(resolution) * ((y_max - y_min) / (resolution - 1))

    X_uni, Z_uni = np.meshgrid(x_coords, y_coords)

    Ux_uni = griddata(
        (X.flat, Y.flat), Ux.flat, (X_uni, Z_uni), fill_value=0, method=method
    )
    Uy_uni = griddata(
        (X.flat, Y.flat), Uy.flat, (X_uni, Z_uni), fill_value=0, method=method
    )
    print("streamline")
    print(np.shape(Ux_uni))
    print(np.shape(Uy_uni))
    print(np.shape(X_uni))
    print(np.shape(Z_uni))
    return (x_coords, y_coords, Ux_uni, Uy_uni)


class RUN:
    def __init__(self, PathToSimulation, PathToRender, end=1):
        """
        If there are n dumps, and end=0.5, only 0.5n of the dumps will be read.
        """
        RequirePath(PathToSimulation, dir_or_file="dir")
        self.PathToSimulation = PathToSimulation
        self.name = os.path.basename(self.PathToSimulation)

        self.analysis_path = f"{PathToSimulation}/timevol.dat"
        self.vtk_folder = f"{self.PathToSimulation}/frames/global"
        self.vtk_pattern = f"{self.PathToSimulation}/vtks/data*.vtk"
        # self.slice1_folder = f"{self.PathToSimulation}/frames/slice1"
        # self.slice1_pattern = f"{self.PathToSimulation}/vtks/slice1*.vtk"
        self.slice1_folder = self.vtk_folder
        self.slice1_pattern = self.vtk_pattern

        self.slice1_list = glob.glob(self.slice1_pattern)
        self.slice1_list[: int(end * len(self.slice1_list))]

        self.vtk_list = glob.glob(self.vtk_pattern)
        self.vtk_list[: int(end * len(self.vtk_list))]

        self.data_info = {}
        self.data_types = ["analysis", "slice1", "vtk"]
        self.data_info["analysis"] = Data_info(self.analysis_path)
        self.data_info["slice1"] = Data_info(self.slice1_pattern)
        self.data_info["vtk"] = Data_info(self.vtk_pattern)
        self.data_info["analysis"].set_data_test(
            lambda path: dat_to_dict(path, end=end)
        )
        self.data_info["slice1"].set_data_test(
            lambda path: readVTK(path, geometry="spherical").data
        )
        self.data_info["vtk"].set_data_test(
            lambda path: readVTK(path, geometry="spherical").data
        )

        self.bounded = False

        folders = ["frames/slice1", "frames/global", "videos"]
        self.videos_folder = f"{PathToRender}/videos"
        for subfolder in folders:
            path = f"{PathToRender}/{subfolder}"
            try:
                os.makedirs(path)
            except OSError as _:
                pass
                content = glob.glob(f"{path}/*")
                user_agree = input(
                    f"Will overwrite the {subfolder} folder ({len(content)} files) [o/r/n] (overwrite, remove, no)"
                )
                if user_agree == "r":
                    for f in content:
                        os.remove(f)
                elif user_agree == "n":
                    exit()

    def print_available_quantities(self):
        print(f"------ Available quantities in {self.PathToSimulation} ------")
        for data_type in self.data_types:
            if self.data_info[data_type].status:
                print(f"{data_type}:", self.data_info[data_type].data_test.keys())
                for qt in self.data_info[data_type].data_test.keys():
                    print(
                        f"{qt:>10} {np.shape(self.data_info[data_type].data_test[qt].data)}"
                    )

    def plot_analysis(self):
        self.analysis = self.data_info["analysis"].data_test
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle(self.PathToSimulation)
        fig.subplots_adjust(left=0.1, right=1 - 0.1, hspace=0.5, wspace=0.2)
        a = self.analysis
        t = a["t"]
        print(t, a["divB"])
        axs[0, 0].plot(t, a["divB"])
        axs[0, 0].set_xlabel(r"$t$")
        axs[0, 0].set_ylabel(r"$\mathrm{div} B$")

        for axsl in axs:
            for ax in axsl:
                ax.legend()
                ax.grid()
        image_path = f"{self.PathToSimulation}/analysis.png"
        fig.savefig(image_path)
        print(f"[OK] {image_path}")

    def slice_to_png(self, slice1_path, bounds_dict):
        fig, axs = plt.subplots(3, 3, figsize=(16, 16))
        fig.subplots_adjust(left=0.05, right=1 - 0.05, hspace=0.2, wspace=0.2)
        print(slice1_path)
        V = readVTK(slice1_path, geometry="spherical")

        R = V.r
        Theta = V.theta
        Phi = V.phi
        print(Phi)
        print(np.shape(R), np.shape(Theta))

        R, Theta = np.meshgrid(V.r, V.theta)
        print(np.shape(R), np.shape(Theta))
        X = R * np.sin(Theta)
        Z = R * np.cos(Theta)
        # Z = np.flip(Z)  # increasing z for streamplot

        for qt in V.data:
            V.data[qt] = np.transpose(V.data[qt][:, :, 0])

        time = V.t[0]
        fig.suptitle(f"{self.PathToSimulation} $t={time:.1e}$")

        Br = V.data["BX1"]
        Btheta = V.data["BX2"]
        Bphi = V.data["BX3"]

        V.data["Bx"] = np.sin(Theta) * Br + np.cos(Theta) * Btheta
        V.data["Bz"] = np.cos(Theta) * Br - np.sin(Theta) * Btheta
        V.data["By"] = Bphi

        (x_coords, z_coords, Bx_uni, Bz_uni) = streamplot(
            X, Z, V.data["Bx"], V.data["Bz"]
        )

        for i, dir in enumerate(["Bx", "By", "Bz"]):
            row = i % 3
            column = i // 3
            data = V.data[dir]
            map = axs[row, column].pcolormesh(
                X,
                Z,
                data,
                # vmin=bounds_dict[dir][0],
                # vmax=bounds_dict[dir][1],
                # shading="flat",
                # rasterized=True,
            )

            axs[row, column].streamplot(
                x_coords,
                z_coords,
                Bx_uni,
                Bz_uni,
                density=[0.5, 1],
            )

            fig.colorbar(map, ax=axs[row, column])
            axs[row, column].set_xlabel(r"$x$")
            axs[row, column].set_ylabel(r"$z$")
            axs[row, column].set_title(rf"${dir}$")

        # ## * Streamlines

        for row in axs:
            for ax in row:
                ax.set_aspect("equal", adjustable="box")
        slice1_name = os.path.basename(slice1_path)
        slice1_png_path = f"{self.slice1_folder}/{slice1_name[:-4]}.png"
        fig.savefig(slice1_png_path, dpi=200)

    def plot_slice(self, jump_to_movie=False):

        if not jump_to_movie:
            bounds_dict = {}
            for qt in ["RHO", "BX1", "BX2", "BX3", "VX1", "VX2", "VX3"]:
                bounds_dict[qt] = (
                    get_bounds(self.slice1_list, qt) if self.bounded else [None, None]
                )

            for slice1_path in [self.slice1_list[0]]:
                self.slice_to_png(slice1_path, bounds_dict)
        # movie(
        #     self.slice1_folder,
        #     f"{self.videos_folder}/slice1{'_bounded' if self.bounded else ''}.mp4",
        # )

    # def plot_vtk(self, jump_to_movie=False):

    # if not jump_to_movie:
    #     bounds_dict = {}
    #     for qt in self.data_info["vtk"].data_test.keys():
    #         bounds_dict[qt] = (
    #             get_bounds(self.vtk_list, qt) if self.bounded else [None, None]
    #         )

    #     for vtk_path in self.vtk_list:
    #         self.slice_to_png(vtk_path, bounds_dict)
    # movie(
    #     self.vtk_folder,
    #     f"{self.videos_folder}/vtk{'_bounded' if self.bounded else ''}.mp4",
    # )


def do_task(task):
    PathToSimulation = f"/home/dp316/dp316/dc-fang1/outputs/OhmicWind/{task}"
    PathToRender = f"/home/dp316/dp316/dc-fang1/outputs/OhmicWind/{task}"

    run = RUN(PathToSimulation, PathToRender)
    run.print_available_quantities()
    # run.plot_analysis()
    run.plot_slice(jump_to_movie=False)


if __name__ == "__main__":
    sequential = True

    tasks = ["OW_test"]
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
