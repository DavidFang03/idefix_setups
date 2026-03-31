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
from scipy.interpolate import RegularGridInterpolator
import json
from matplotlib.colors import LogNorm

with open("config.json") as f:
    configs = json.load(f)


def process_configs(configs):
    for runName in configs:
        if "copy" in configs[runName]:
            configs[runName] = configs[configs[runName]["copy"]]
    return configs


configs = process_configs(configs)

count = 0


plt.style.use("dark_background")
test_first_run = False
sequential = True
bounded = True
if test_first_run:
    run_movie = False
    sequential = True  # safety guard
else:
    run_movie = True

lw_streamline = 0.2
density_streamline = [1, 2]
arrowstyle_streamline = "->"

# Do not touch
geometry = "spherical"


def fit(X, Y, deg, start=0, end=1):
    index_start = int(start * (len(X) - 1))
    index_end = int(end * (len(X) - 1))
    params, cov = np.polyfit(
        X[index_start:index_end], Y[index_start:index_end], deg=deg, cov=True
    )
    return params, np.diag(cov)


def divide_discardingNullDenominator(a, b):
    """
    Returns a/b but with None wherever b=0
    """
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def applyOperation_discardingNone(op, array):
    mask = (array != np.nan) & (array != 0)
    output = np.full(array.shape, np.nan)
    valid_data = array[mask].astype(float)
    output[mask] = op(valid_data)

    return output
    # return op(array, out=np.full(array.shape, None), where=array != None)


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


class Quantity1D:
    def __init__(self, key, symbol, **kwargs):
        scale = kwargs.get("scale", "linear")
        title = kwargs.get("title", symbol)
        self.key = key
        self.symbol = symbol
        self.title = title
        self.scale = scale

    def set_data(self, data):
        self.data = data


class Quantity2D:
    def __init__(self, key, name):
        self.key = key
        self.name = name

    def set_data(self, data):
        self.data = data

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_cmap(self, cmap):
        self.cmap = cmap

    def set_norm(self, norm):
        self.norm = norm

    def set_streamline_toshow(self, xUni, yUni, UxUni, UyUni, streamline_name):
        """
        One can pcolormesh a given quantity but plot streamlines of a different quantity.
        """
        self.xUni, self.yUni = xUni, yUni
        self.UxUni, self.UyUni = UxUni, UyUni
        self.streamline_name = streamline_name

    def set_plot_coords(self, i, j):
        self.coords = [i, j]


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
    X,Y must come from meshgrid, Ux and Uy are the data points.
    Returns the 1d arrays x_coords and y-coords, with the interpolated values Ux_uni, Uy_uni (2D arrays) that can go directly in streamplot
    """

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

    x_coords = r_min + np.arange(resolution) * ((r_max - r_min) / (resolution - 1))
    z_min, z_max = -r_max, r_max
    z_coords = z_min + np.arange(resolution) * ((z_max - z_min) / (resolution - 1))

    X_uni, Z_uni = np.meshgrid(x_coords, z_coords)
    R_fromuni = np.sqrt(X_uni**2 + Z_uni**2)
    Theta_fromuni = np.pi / 2 - np.atan(Z_uni / X_uni)

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
        self.bounded = bounded
        self.parallel = True
        if test_first_run:
            self.parallel = False
        self.doStreamLines = True

        RequirePath(PathToProject, dir_or_file="dir")
        self.DataPath = f"{PathToProject}/outputs/{RunName}"
        self.iniPath = f"{PathToProject}/inputs/{RunName}.ini"
        self.name = RunName
        self.formatInputs_text = self.formatInputs()
        self.config = configs[self.name]

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
            "RHO": Quantity2D("RHO", r"$\rho$"),
            "cs": Quantity2D("cs", r"$c_s$"),
            "Mach_p": Quantity2D("Mach_p", r"$\text{Mach}_\text{p}$"),
        }

        for i, qty in enumerate(self.quantities2D.keys()):
            row = i % 3
            column = i // 3
            self.quantities2D[qty].set_plot_coords(row, column)

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
                # subfolder = os.path.basename(path)
                # content = glob.glob(f"{path}/*")
                # user_agree = input(
                #     f"Will overwrite the {subfolder} folder ({len(content)} files) [o/r/n] (overwrite, remove, no)"
                # )
                # if user_agree == "r":
                #     for f in content:
                #         os.remove(f)
                # elif user_agree == "n":
                #     exit()

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
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(self.DataPath)
        fig.subplots_adjust(left=0.1, right=1 - 0.05, top=0.8, hspace=0.3, wspace=0.2)
        a = self.analysis
        t = a["t"]
        self.quantities1D = {
            "divB": Quantity1D("divB", r"$\mathrm{div} B$", scale="linear"),
            "mass": Quantity1D("mass", r"$M$", title="Mass (normalized)", scale="log"),
        }
        for i, timeseries in enumerate(self.quantities1D.keys()):
            ax = axs[i % 3, i // 3]
            qty1D_info = self.quantities1D[timeseries]
            ax.plot(t, a[timeseries], label=qty1D_info.title)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(qty1D_info.symbol)
            ax.set_yscale(qty1D_info.scale)
            ax.set_title(qty1D_info.title)
            ax.legend()
            ax.grid()

        self.annotateInputs(axs)

        image_path = self.analysisFrame_path
        fig.savefig(image_path)
        print(f"[OK] {image_path}")

    def processVTK(self, V):
        """
        #!!! to move in setup.cpp
        """
        Theta = self.Theta
        for qt in V.data:
            V.data[qt] = np.transpose(V.data[qt][:, :, 0])

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
        V.data["Mach_p"] = divide_discardingNullDenominator(
            np.sqrt(V.data["vx"] ** 2 + V.data["vz"] ** 2),
            V.data["cs"],
        )
        # V.data["rho"] = np.log10(V.data["RHO"])

        # V.data["Mach_p"] = applyOperation_discardingNone(np.log10, V.data["Mach_p"])
        # V.data["Mach_p"] = np.log10(V.data["Mach_p"])

    def get_bounds_indiv(self, args):
        """
        args (list[2]) must have two components:
            first:   vtk_path (str)
            second:   fields_indexes (dict) where fields_indexes[field] = index
        """
        vtk_path = args[0]
        fields_indexes = args[1]
        V = readVTK(vtk_path, geometry=geometry)
        self.processVTK(V)
        bounds = np.empty((len(fields_indexes), 2))
        for field in fields_indexes.keys():
            data = V.data[field]
            index = fields_indexes[field]
            bounds[index, 0] = np.min(data[data != np.nan])
            bounds[index, 1] = np.max(data[data != np.nan])
        # bounds = np.min(data), np.max(data)
        # print(bounds[fields_indexes["Mach_p"]])
        return bounds

    def get_bounds(self, vtk_list, fields):
        """
        Get the bounds (min, max) of all given fields. I recommend not passing the entire vtk_list but rather vtk_list[1:] to discard the first output(s ?).

        vtk_list    List of dump file paths
        field       Field

        returns
        dict where dict[field] = (min, max)
        """
        mapfields_indexes = {}
        for i, field in enumerate(fields):
            mapfields_indexes[field] = i

        with Pool(16) as pool:
            all_bounds = pool.map(
                self.get_bounds_indiv,
                np.column_stack(
                    (
                        vtk_list,
                        [mapfields_indexes for _ in vtk_list],
                    )
                ),
            )
        print(np.shape(all_bounds))
        all_bounds = np.array(all_bounds)
        bounds = {}
        for field in fields:
            i = mapfields_indexes[field]
            bounds[field] = np.min(all_bounds[:, i, 0]), np.max(all_bounds[:, i, 1])
        return bounds

    def slice_to_png(self, slice1_path):
        rows, columns = 3, 3
        cbars = np.full((rows, columns), None)
        fig, axs = plt.subplots(rows, columns, figsize=(12, 16))
        fig.subplots_adjust(
            left=0.05,
            right=1 - 0.05,
            bottom=0.07,
            top=0.83,
            hspace=0.25,
            wspace=0.02,
        )
        V = readVTK(slice1_path, geometry=geometry)

        X = self.X
        Z = self.Z

        self.processVTK(V)

        time = V.t[0]
        fig.suptitle(f"{self.name}\n{slice1_path}\n$t={time:.1e}$")

        self.annotateInputs(axs)

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
            for qty in ["vx", "vy", "vz", "RHO", "cs", "Mach_p"]:
                self.quantities2D[qty].set_streamline_toshow(
                    x_coords,
                    z_coords,
                    vx_uni,
                    vz_uni,
                    streamline_name=r"$v_p$",
                )

        for i, qty in enumerate(self.quantities2D.keys()):
            qtyInfo = self.quantities2D[qty]
            data = V.data[qty]
            ax = axs[*qtyInfo.coords]
            if qtyInfo.norm is None:
                map = ax.pcolormesh(
                    X,
                    Z,
                    data,
                    vmin=qtyInfo.bounds[0],
                    vmax=qtyInfo.bounds[1],
                    cmap=qtyInfo.cmap,
                    # shading="flat",
                    # rasterized=True,
                )
            elif qtyInfo.norm == "log":
                if None in qtyInfo.bounds:
                    vmin, vmax = data.min(), data.max()
                else:
                    vmin, vmax = qtyInfo.bounds
                # print(f"VMIN {qty}", vmin, vmax)
                map = ax.pcolormesh(
                    X,
                    Z,
                    data,
                    cmap=qtyInfo.cmap,
                    norm=(LogNorm(vmin=vmin, vmax=vmax)),
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

            cbars[*qtyInfo.coords] = fig.colorbar(map, ax=ax, label=qtyInfo.name)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x$")
            if qtyInfo.coords[1] == 0:
                ax.set_ylabel(r"$z$")
            ax.set_title(
                rf"{qtyInfo.name if not self.doStreamLines else qtyInfo.streamline_name}"
            )

        Mach_pInfo = self.quantities2D["Mach_p"]
        level0 = axs[*Mach_pInfo.coords].contour(
            X,
            Z,
            V.data["Mach_p"],
            [1],
            alpha=0.5,
            colors=["green"],
            linewidths=[1],
        )
        cbars[*qtyInfo.coords].add_lines(level0)

        slice1_name = os.path.basename(slice1_path)
        slice1_png_path = f"{self.slice1Frames_folder}/{slice1_name[:-4]}.png"
        fig.savefig(slice1_png_path, dpi=200)
        plt.close(fig)
        print(f"[OK] {slice1_png_path}")

    def formatInputs(self):
        with open(self.iniPath) as ini:
            lines = ini.readlines()
            indexes_to_format = []

            # Grid part
            istart = lines.index("[Grid]\n") + 1
            iend = lines.index("[TimeIntegrator]\n")
            indexes_to_format += [*range(istart, iend)]
            # Setup part
            istart = lines.index("[Setup]\n") + 1
            iend = lines.index("[Output]\n")
            indexes_to_format += [*range(istart, iend)]
            text = ""
            for i in indexes_to_format:
                line = lines[i]
                line_split = line.split()
                if len(line_split) > 1 and line_split[0] != "#":
                    text += f"{line_split[0]:>25} {' '.join(line_split[1:]):<10}"
                    text += "\n"
        return text

    def annotateInputs(self, axs):
        axs.flat[1].annotate(
            self.formatInputs_text,
            xy=(0.1, 0.9),
            xycoords="figure fraction",
            verticalalignment="center",
            horizontalalignment="left",
            family="monospace",
            fontsize=10,
        )

    def plot_slice(self, jump_to_movie=False):

        if not jump_to_movie:
            all_quantities = self.quantities2D.keys()
            config = self.config
            print("Computing bounds, please wait...")
            quantities_tobound = [
                key
                for key in all_quantities
                if key not in config or "range" not in config[key]
            ]
            print(config)
            print(quantities_tobound)
            all_bounds = {}
            if len(quantities_tobound) > 0:
                all_bounds = self.get_bounds(self.slice1_list[5:], quantities_tobound)
                [print(f"{key}: {all_bounds[key]}") for key in all_bounds]
            for qt in all_quantities:
                if not self.bounded:
                    self.quantities2D[qt].set_bounds([None, None])
                elif qt in all_bounds:
                    self.quantities2D[qt].set_bounds(all_bounds[qt])
                elif qt in config and "range" in config[qt]:
                    self.quantities2D[qt].set_bounds(config[qt]["range"])
                else:
                    self.quantities2D[qt].set_bounds([None, None])

            for qt in all_quantities:
                self.quantities2D[qt].set_cmap("berlin")
                self.quantities2D[qt].set_norm(None)
                if qt in config:
                    if "cmap" in config[qt]:
                        self.quantities2D[qt].set_cmap(config[qt]["cmap"])
                    if "norm" in config[qt]:
                        self.quantities2D[qt].set_norm(config[qt]["norm"])

            print("Bounds computed")
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


def do_task(task):
    PathToProject = "/home/dp316/dp316/dc-fang1/IdefixRuns/OhmicWindv2"
    # RunName = f"/home/dp316/dp316/dc-fang1/IdefixRuns/OhmicWindv2/"

    run = RUN(PathToProject, task, end=1)
    run.print_available_quantities()
    run.plot_slice(jump_to_movie=False)
    run.plot_analysis()


if __name__ == "__main__":
    tasks = ["OW_Rm1", "OW_test"]
    if sequential:
        for task in tasks:
            count = 0
            do_task(task)
    else:
        with Pool(len(tasks)) as pool:
            pool.map(do_task, tasks)
# plt.show()
# Streamlines
# The formula needed to transform vectors in spherical coordinates into cartesian ones is
# BX1*(coordsX*iHat+coordsY*jHat+coordsZ*kHat)/sqrt(coordsX^2+coordsY^2+coordsZ^2) + BX2*(coordsX*coordsZ*iHat+coordsY*coordsZ*jHat-(coordsX^2+coordsY^2)*kHat)/sqrt((coordsX^2+coordsY^2+coordsZ^2)*(coordsX^2+coordsY^2)) + BX3*(coordsX*jHat-coordsY*iHat)/sqrt(coordsX^2+coordsY^2)
