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
from scipy.interpolate import griddata, RegularGridInterpolator
import matplotlib.colors as colors

plt.style.use("dark_background")
test_first_run = False
if test_first_run:
    run_movie = False
else:
    run_movie = True

lw_streamline = 0.5
density_streamline = [1, 2]
arrowstyle_streamline = "->"


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
    """
    Different types of output: global (vtk), slice (vtk), timevol (dat)
    """

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


class Quantity2D:
    def __init__(self, key, name):
        self.key = key
        self.name = name

    def set_data(self, data):
        self.data = data

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_streamline_toshow(self, xUni, yUni, UxUni, UyUni, streamline_name):
        """
        One can pcolormesh a given quantity but plot streamlines of a different quantity.
        """
        self.xUni, self.yUni = xUni, yUni
        self.UxUni, self.UyUni = UxUni, UyUni
        self.streamline_name = streamline_name


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


# def get_streamplot_data(
#     Xpoints, Ypoints, Ux_data, Uy_data, resolution=200, method="linear"
# ):
#     """
#     X,Y must come from meshgrid, Ux and Uy are the data points.
#     Returns the 1d arrays x_coords and y-coords, with the interpolated values Ux_uni, Uy_uni (2D arrays) that can go directly in streamplot
#     """
#     X = Xpoints
#     Y = Ypoints
#     Ux = Ux_data
#     Uy = Uy_data
#     X = np.flip(Xpoints, axis=1)
#     Y = np.flip(Ypoints, axis=0)
#     print(X[0, :])
#     print(Y[:, 0])
#     Ux = np.flip(Ux_data, axis=1)
#     Uy = np.flip(Uy_data, axis=0)
#     x_min, x_max = np.min(X), np.max(X)
#     y_min, y_max = np.min(Y), np.max(Y)

#     resolution = 200
#     x_coords = x_min + np.arange(resolution) * ((x_max - x_min) / (resolution - 1))
#     y_coords = y_min + np.arange(resolution) * ((y_max - y_min) / (resolution - 1))

#     X_uni, Y_uni = np.meshgrid(x_coords, y_coords)

#     # Ux_uni = RegularGridInterpolator(
#     #     (X[0, :], Y[:, 0]), Ux.T, fill_value=0, method=method, bounds_error=False
#     # )

#     # Uy_uni = RegularGridInterpolator(
#     #     (X[0, :], Y[:, 0]), Uy.T, fill_value=0, method=method, bounds_error=False
#     # )

#     Ux_uni = griddata(
#         (X.flat, Y.flat), Ux.flat, (X_uni, Y_uni), fill_value=0, method=method
#     )
#     Uy_uni = griddata(
#         (X.flat, Y.flat), Uy.flat, (X_uni, Y_uni), fill_value=0, method=method
#     )
#     # print("streamline")
#     # print(np.shape(Ux_uni))
#     # print(np.shape(Uy_uni))
#     # print(np.shape(X_uni))
#     # print(np.shape(Y_uni))
#     # pts = np.stack((X_uni, Y_uni), axis=-1)
#     # return (
#     #     x_coords,
#     #     y_coords,
#     #     Ux_uni(pts),
#     #     Uy_uni(pts),
#     # )
#     return (x_coords, y_coords, Ux_uni, Uy_uni)


def get_streamplot_data(
    RLine, ThetaLine, Ux_data, Uy_data, resolution=200, method="linear"
):
    """
    X, Y doivent provenir d'un meshgrid. Ux et Uy sont les points de données.
    Renvoie les tableaux 1D x_coords et y_coords, avec les valeurs interpolées
    Ux_uni, Uy_uni (tableaux 2D) prêts pour plt.streamplot.
    """

    # 4. Créer la nouvelle grille régulière

    # 5. Créer les interpolateurs
    # On passe la transposée (.T) pour que les axes correspondent à (x, y)
    Ux_interp = RegularGridInterpolator(
        (RLine, ThetaLine),
        Ux_data.T,
        fill_value=np.nan,
        method=method,
        bounds_error=False,
    )
    Uy_interp = RegularGridInterpolator(
        (RLine, ThetaLine),
        Uy_data.T,
        fill_value=np.nan,
        method=method,
        bounds_error=False,
    )
    r_min, r_max = np.min(RLine), np.max(RLine)
    # y_min, y_max = np.min(ThetaLine), np.max(ThetaLine)

    x_coords = r_min + np.arange(resolution) * ((r_max - r_min) / (resolution - 1))
    z_min, z_max = -r_max, r_max
    z_coords = z_min + np.arange(resolution) * ((z_max - z_min) / (resolution - 1))

    X_uni, Z_uni = np.meshgrid(x_coords, z_coords)
    R_fromuni = np.sqrt(X_uni**2 + Z_uni**2)
    Theta_fromuni = np.pi / 2 - np.atan(Z_uni / X_uni)

    # 6. Évaluer les interpolateurs en empilant les points proprement
    pts = np.stack((R_fromuni, Theta_fromuni), axis=-1)

    return (
        x_coords,
        z_coords,
        Ux_interp(pts),
        Uy_interp(pts),
    )


class RUN:
    def __init__(self, PathToProject, RunName, end=1):
        """
        If there are n dumps, and end=0.5, only 0.5n of the dumps will be read.
        """
        self.bounded = False
        self.parallel = True
        if test_first_run:
            self.parallel = False
        self.doStreamLines = True

        RequirePath(PathToProject, dir_or_file="dir")
        self.DataPath = f"{PathToProject}/outputs/{RunName}"
        self.iniPath = f"{PathToProject}/inputs/{RunName}.ini"
        self.name = RunName
        self.formatInputs_text = self.formatInputs()

        # Data location
        self.analysis_path = f"{self.DataPath}/timevol.dat"
        self.vtk_pattern = f"{self.DataPath}/vtks/data*.vtk"
        self.slice1_pattern = self.vtk_pattern

        # Renders location
        self.globalFrames_folder = f"{PathToProject}/frames/{RunName}/global"
        self.slice1Frames_folder = self.globalFrames_folder
        self.singleFrame_folder = f"{PathToProject}/frames/{RunName}"
        self.videos_folder = f"{PathToProject}/videos/{RunName}"

        self.analysisFrame_path = f"{self.singleFrame_folder}/analysis.png"
        self.meshFrame_path = f"{self.singleFrame_folder}/mesh.png"
        self.slice1Movie_path = (
            f"{self.videos_folder}/slice1{'_bounded' if self.bounded else ''}.mp4"
        )
        # self.slice1Frames_folder = f"{self.PathToSimulation}/frames/slice1"
        # self.slice1_pattern = f"{self.PathToSimulation}/vtks/slice1*.vtk"

        self.slice1_list = sorted(glob.glob(self.slice1_pattern))
        self.slice1_list = self.slice1_list[: int(end * len(self.slice1_list))]

        self.vtk_list = sorted(glob.glob(self.vtk_pattern))
        self.vtk_list = self.vtk_list[: int(end * len(self.vtk_list))]

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

        self.quantities2D = {
            "Bx": Quantity2D("Bx", r"$B_x$"),
            "By": Quantity2D("By", r"$B_y$"),
            "Bz": Quantity2D("Bz", r"$B_z$"),
            "vx": Quantity2D("vx", r"$v_x$"),
            "vy": Quantity2D("vy", r"$v_y$"),
            "vz": Quantity2D("vz", r"$v_z$"),
            "RHO": Quantity2D("RHO", r"$\rho"),
            "cs": Quantity2D("cs", r"$c_s$"),
            "logMach": Quantity2D("logMach", r"$\log \text{Mach}$"),
        }

        # we must create
        # ({project}/frames/{RunName})
        # {project}/frames/{RunName}/global
        # {project}/frames/{RunName}/slice1
        # {project}/videos/{RunName}
        folders = [
            self.globalFrames_folder,
            self.slice1Frames_folder,
            self.videos_folder,
        ]
        for path in folders:
            try:
                print(f"mkdir {path}")
                os.makedirs(path)
            except OSError as _:
                pass
                subfolder = os.path.basename(path)
                content = glob.glob(f"{path}/*")
                user_agree = input(
                    f"Will overwrite the {subfolder} folder ({len(content)} files) [o/r/n] (overwrite, remove, no)"
                )
                if user_agree == "r":
                    for f in content:
                        os.remove(f)
                elif user_agree == "n":
                    exit()

        if self.data_info["slice1"].status:
            vtk = readVTK(self.data_info["slice1"].test_file, geometry="spherical")
            self.RLine = vtk.r
            self.ThetaLine = vtk.theta
            self.R, self.Theta = np.meshgrid(self.RLine, self.ThetaLine)
            self.X = self.R * np.sin(self.Theta)
            self.Z = self.R * np.cos(self.Theta)
            fig, ax = plt.subplots(2)
            ax[0].plot(self.X[0, :])
            ax[1].plot(self.Z[:, 0])
            fig.savefig(self.meshFrame_path)
            plt.close(fig)

    def print_available_quantities(self):
        print(f"------ Available quantities in {self.DataPath} ------")
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
        fig.suptitle(self.DataPath)
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
        image_path = self.analysisFrame_path
        fig.savefig(image_path)
        print(f"[OK] {image_path}")

    def slice_to_png(self, slice1_path):
        bounds_dict = self.bounds_dict
        fig, axs = plt.subplots(3, 3, figsize=(16, 16))
        fig.subplots_adjust(
            left=0.15,
            right=1 - 0.05,
            bottom=0.07,
            top=1 - 0.07,
            hspace=0.2,
            wspace=0.1,
        )
        print(slice1_path)
        V = readVTK(slice1_path, geometry="spherical")

        R = self.R
        Theta = self.Theta
        X = self.X
        Z = self.Z
        # Z = np.flip(Z)  # increasing z for streamplot

        for qt in V.data:
            V.data[qt] = np.transpose(V.data[qt][:, :, 0])

        time = V.t[0]
        fig.suptitle(f"{self.DataPath} \n $t={time:.1e}$")

        self.annotateInputs(axs)

        Br = V.data["BX1"]
        Btheta = V.data["BX2"]
        Bphi = V.data["BX3"]
        vr = V.data["VX1"]
        vtheta = V.data["VX2"]
        vphi = V.data["VX3"]

        V.data["Bx"] = np.sin(Theta) * Br + np.cos(Theta) * Btheta
        V.data["Bz"] = np.cos(Theta) * Br - np.sin(Theta) * Btheta
        V.data["By"] = Bphi
        V.data["vx"] = np.sin(Theta) * vr + np.cos(Theta) * vtheta
        V.data["vz"] = np.cos(Theta) * vr - np.sin(Theta) * vtheta
        V.data["vy"] = vphi
        V.data["cs"] = np.sqrt(V.data["PRS"] / V.data["RHO"])
        V.data["Mach"] = (
            np.sqrt(V.data["vx"] ** 2 + V.data["vy"] ** 2 + V.data["vz"] ** 2)
            / V.data["cs"]
        )
        V.data["logMach"] = np.log10(V.data["Mach"])

        for qty in self.quantities2D.keys():
            self.quantities2D[qty].set_data(V.data[qty])

        if self.doStreamLines:
            (x_coords, z_coords, Bx_uni, Bz_uni) = get_streamplot_data(
                self.RLine, self.ThetaLine, V.data["Bx"], V.data["Bz"]
            )
            (v_xpoints, v_zpoints, vx_uni, vz_uni) = get_streamplot_data(
                self.RLine, self.ThetaLine, V.data["vx"], V.data["vz"]
            )
            for qty in ["Bx", "By", "Bz"]:
                self.quantities2D[qty].set_streamline_toshow(
                    x_coords,
                    z_coords,
                    Bx_uni,
                    Bz_uni,
                    streamline_name=r"$B_p$",
                )
            for qty in ["vx", "vy", "vz", "RHO", "cs", "logMach"]:
                self.quantities2D[qty].set_streamline_toshow(
                    x_coords,
                    z_coords,
                    vx_uni,
                    vz_uni,
                    streamline_name=r"$v_p$",
                )
            # print(v_xpoints, v_zpoints, vx_uni, vz_uni)

        for i, qty in enumerate(self.quantities2D.keys()):
            row = i % 3
            column = i // 3
            data = V.data[qty]
            ax = axs[row, column]
            qtyInfo = self.quantities2D[qty]
            map = ax.pcolormesh(
                X,
                Z,
                data,
                # vmin=bounds_dict[qty][0],
                # vmax=bounds_dict[qty][1],
                # shading="flat",
                # rasterized=True,
            )

            if self.doStreamLines:
                ax.streamplot(
                    qtyInfo.xUni,
                    qtyInfo.yUni,
                    qtyInfo.UxUni,
                    qtyInfo.UyUni,
                    density=density_streamline,
                    linewidth=lw_streamline,
                    arrowstyle=arrowstyle_streamline,
                )

            fig.colorbar(map, ax=ax, label=qtyInfo.name)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$z$")
            ax.set_title(
                rf"{qtyInfo.name if not self.doStreamLines else qtyInfo.streamline_name}"
            )

        # for i, dir in enumerate(["vx", "vy", "vz", "RHO", "cs", "logMach"]):
        #     row = i % 3
        #     column = i // 3 + 1
        #     data = V.data[dir]
        #     map = axs[row, column].pcolormesh(
        #         X,
        #         Z,
        #         data,
        #         # vmin=bounds_dict[dir][0],
        #         # vmax=bounds_dict[dir][1],
        #         # shading="flat",
        #         # rasterized=True,
        #     )
        #     if self.doStreamLines:
        #         axs[row, column].streamplot(
        #             v_xpoints,
        #             v_zpoints,
        #             vx_uni,
        #             vz_uni,
        #             density=density_streamline,
        #             linewidth=lw_streamline,
        #             arrowstyle=arrowstyle_streamline,
        #         )

        #     fig.colorbar(map, ax=axs[row, column])
        #     axs[row, column].set_xlabel(r"$x$")
        #     axs[row, column].set_ylabel(r"$z$")
        #     axs[row, column].set_title(rf"${dir}$")

        # for row in axs:
        #     for ax in row:
        #         ax.set_aspect("equal", adjustable="box")
        slice1_name = os.path.basename(slice1_path)
        slice1_png_path = f"{self.slice1Frames_folder}/{slice1_name[:-4]}.png"
        fig.savefig(slice1_png_path, dpi=200)
        plt.close(fig)

    def formatInputs(self):
        with open(self.iniPath) as ini:
            lines = ini.readlines()
            istart = lines.index("[Setup]\n") + 1
            iend = lines.index("[Output]\n")
            text = ""
            for i in range(istart, iend):
                line = lines[i]
                line_split = line.split()
                if len(line_split) > 1 and line_split[0] != "#":
                    text += f"{line_split[0]:>25} {line_split[1]:<10}"
                    text += "\n"
        return text

    def annotateInputs(self, axs):
        axs.flat[1].annotate(
            self.formatInputs_text,
            xy=(0.0, 0.5),
            xycoords="figure fraction",
            verticalalignment="center",
            horizontalalignment="left",
            family="monospace",
            fontsize=10,
        )

    def plot_slice(self, jump_to_movie=False):

        if not jump_to_movie:
            bounds_dict = {}
            for qt in ["RHO", "BX1", "BX2", "BX3", "VX1", "VX2", "VX3"]:
                bounds_dict[qt] = (
                    get_bounds(self.slice1_list, qt) if self.bounded else [None, None]
                )
            self.bounds_dict = bounds_dict
            if test_first_run:
                for slice1_path in [self.slice1_list[0]]:
                    self.slice_to_png(slice1_path)
            else:
                if self.parallel:
                    with Pool(16) as pool:
                        pool.map(self.slice_to_png, self.slice1_list)
                else:
                    for slice1_path in self.slice1_list:
                        self.slice_to_png(slice1_path)

        if run_movie:
            movie(
                self.slice1Frames_folder,
                self.slice1Movie_path,
            )

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
    #     self.globalFrames_folder,
    #     f"{self.videos_folder}/vtk{'_bounded' if self.bounded else ''}.mp4",
    # )


def do_task(task):
    PathToProject = "/home/dp316/dp316/dc-fang1/IdefixRuns/OhmicWindv2"
    # RunName = f"/home/dp316/dp316/dc-fang1/IdefixRuns/OhmicWindv2/"

    run = RUN(PathToProject, task, end=1)
    run.print_available_quantities()
    run.plot_slice(jump_to_movie=False)
    # run.plot_analysis()


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
