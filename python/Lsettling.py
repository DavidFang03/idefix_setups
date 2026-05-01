from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap, PartQuantity
import utilities

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/VerticalSettling"
task = "Settling_Tau"


# tau = 0.1


class analytical_trajectory:
    def __init__(self, tau):
        self.tau = tau
        self.plot_kwargs = {
            "ls": "--",
            "color": "white",
            "lw": 0.5,
            "alpha": 0.75,
            "label": "Predicted",
        }

    def __call__(self, t):
        z0 = 0.1
        r0 = 2

        fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, tau0=self.tau, r0=r0, z0=z0)
        return fluid.zSettling(t)
        # return utilities.solve_2nd_order_ode(fluid.azSettling, z0, 0, t)


# def analytical_trajectory(t):

custom_spaceTimeHeatmaps = []
taus = [1, 0.1, 0.01]
for uid in [0, 1, 2]:
    tau = taus[uid]
    # z_part = PartQuantity(
    #     "PART_X2",
    #     r"$r^\mathrm{part}$",
    #     plot_coords=[0, 0],
    #     uids=[uid],
    # )
    custom_spaceTimeHeatmaps.append(
        SpaceTimeHeatmap(
            f"Dust{uid}_RHO",
            r"$\rho^\mathrm{dust}$",
            plot_coords=[uid, 0],
            title=rf"$\tau = {tau}$",
            cmap="inferno",
            norm="linear",
            ref_function=analytical_trajectory(tau),
            uids=[uid],
        ),
    )

# z_part1 = PartQuantity(
#     "PART_X2",
#     r"$r^\mathrm{part}$",
#     plot_coords=[0, 1],
# )

# z_part2 = PartQuantity(
#     "PART_X3",
#     r"$r^\mathrm{part}$",
#     plot_coords=[0, 2],
# )


#     SpaceTimeHeatmap(
#     "RHO",
#     r"$\rho^\mathrm{dust}$",
#     plot_coords=[0, 0],
#     title=r"$\tau = 1$",
#     cmap="inferno",
#     norm="log",
#     ref_function=analytical_trajectory,
#     trace_over=[z_part],
# )
# SpaceTimeHeatmap(
#     "Dust0_RHO",
#     r"$\rho^\mathrm{dust}$",
#     plot_coords=[0, 1],
#     title=r"$\tau = 0.2$",
#     cmap="inferno",
#     ref_function=get_analytical_trajectory(0.2),
#     trace_over=[z_part],
# ),
# SpaceTimeHeatmap(
#     "Dust0_RHO",
#     r"$\rho^\mathrm{dust}$",
#     plot_coords=[0, 2],
#     title=r"$\tau = 0.04$",
#     cmap="inferno",
#     ref_function=get_analytical_trajectory(0.04),
#     trace_over=[z_part],
# ),

SpaceTimeHeatmap.suptitle = "Particle trajectory over gas density"

# custom_partQuantities = [z_part]


runContext = RunContext(
    task,
    projectPath,
    partFolder="/home/dp316/dp316/dc-fang1/IdefixRuns/VerticalSettling/setup_l",
)
pipeline = Pipeline(runContext, spaceTimeHeatmaps=custom_spaceTimeHeatmaps)

pipeline.run()
