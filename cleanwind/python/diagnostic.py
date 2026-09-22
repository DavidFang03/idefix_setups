from idefix2python import RunContext, Pipeline, MapMovie2D, Fig
import numpy as np
import matplotlib
from common import WindyDisk
import os
from dotenv import load_dotenv

load_dotenv()

matplotlib.use("pdf")

RUNS_FOLDER_PATH = str(os.getenv("RUNS_FOLDER_PATH"))
print(RUNS_FOLDER_PATH)
projectPath = f"{RUNS_FOLDER_PATH}/cleanwind"
configPath = f"{RUNS_FOLDER_PATH}/AODustyLWind/config.json"
# task = "cw_20_b1e4"
# task = "clean_wind_100_v2_b1e4"
task = "lr_wind_v8_MHDOFF_b1e4"
runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    # pdf_mode=True,
)

eps = 0.05
betamid = float(runContext.inidata["Setup"]["beta"])
wd = WindyDisk(runContext.inidata, runContext.gridInfo)


def float_to_latex(num: float) -> str:
    """
    Converts a float (including scientific notation like 4e3)
    into a LaTeX formatted string.
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


def plasmabeta(v):
    P = v.data["PRS"]
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return 8 * np.pi * P / B2


def title(ax, v):
    fig = ax.get_figure()
    fig.suptitle(rf"$t={float_to_latex(v.t[0] / (2 * np.pi))}$ yr")


inferno = {"cmap": "inferno"}
quantities = [
    MapMovie2D(
        "RHO",
        r"\rho",
        title="Gas density",
        plot_coords=[0, 0],
        streamlines=["VX1", "VX2"],
        customize=title,
        style_kwargs={"cmap": "viridis"},
    ),
    # MapMovie2D(
    #     "beta",
    #     r"$\beta$",
    #     title=r"Plasma $\beta$ (poloidal)",
    #     plot_coords=[0, 1],
    #     streamlines=["BX1", "BX2"],
    #     compute=plasmabeta,
    #     bounds=[1, betamid],
    #     norm="log",
    # ),
    MapMovie2D(
        "T",
        r"$T$",
        title=r"Temperature",
        plot_coords=[0, 1],
        streamlines=["VX1", "VX2"],
        compute=wd.temperature,
        norm="log",
    ),
    MapMovie2D(
        "vz",
        r"$v_z$",
        plot_coords=[0, 2],
        streamlines=["VX1", "VX2"],
        compute=wd.vz,
        bounds=[-1e-5, 1e-5],
        style_kwargs={"cmap": "coolwarm"},
    ),
]

for qty in quantities:
    qty.ymin = -2.5
    qty.ymax = 2.5
    qty.xmax = 2.5
fig1 = Fig(quantities)


# Initialize context

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

if __name__ == "__main__":
    pipeline.run()
