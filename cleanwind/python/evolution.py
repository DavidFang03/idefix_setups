from idefix2python import RunContext, Pipeline, MapMovie2D, Fig
import numpy as np
import matplotlib

matplotlib.use("pdf")

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
# projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind"
configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/config.json"
# task = "cw_20_b1e4"
task = "clean_wind_100_v2_b1e4"
runContext = RunContext(
    task,
    projectPath,
    configPath=configPath,
    # pdf_mode=True,
)

eps = 0.05
betamid = float(runContext.inidata["Setup"]["beta"])


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
        streamline_kwargs={
            "color": (0.6, 0.6, 0.6, 0.9),
            "density": 1,
            "linewidth": 1,
        },
    ),
    MapMovie2D(
        "beta",
        r"$\beta$",
        title=r"Plasma $\beta$ (poloidal)",
        plot_coords=[0, 1],
        streamlines=["BX1", "BX2"],
        compute=plasmabeta,
        bounds=[1, betamid],
        norm="log",
        streamline_kwargs={
            "color": (0.6, 0.6, 0.6, 0.9),
            "density": 1,
            "linewidth": 1,
        },
    ),
]
fig1 = Fig(quantities)
# fig1 = Fig(
#     [
#         MapMovie2D("InvDt", plot_coords=[0, 0], streamlines=["VX1", "VX2"]),
#     ]
# )
# fig1.axes[0, 0].xmin = 0


custom_fields2D = []


# Initialize context

# Inject ONLY the 2D fields into the pipeline
pipeline = Pipeline(runContext, [fig1])

pipeline.run()
