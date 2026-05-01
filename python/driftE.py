from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap, PartQuantity
import utilities

projectPath = "/home/dfang/Code/idefix_setups/ThomasDrift"
task = "Drift_Size"


# tau = 0.1


class analytical_trajectory:
    def __init__(self, beta):
        self.beta = beta
        self.plot_kwargs = {
            "ls": "--",
            "color": "white",
            "lw": 0.5,
            "alpha": 0.75,
            "label": "Predicted",
        }

    def __call__(self, t):
        z0 = 0
        r0 = 2
        fluid = utilities.Fluid(
            cs0=0.05,
            csSlope=-0.5,
            sigma0=0.125,
            sigmaSlope=-0.5,
            beta=self.beta,
            r0=r0,
            z0=z0,
            drag="epstein",
        )
        return utilities.integrate(fluid.vrDrift, r0, t)


# def analytical_trajectory(t):
if __name__ == "__main__":
    custom_spaceTimeHeatmaps = []
    betas = [1, 0.1, 0.01]
    for uid in [0, 1, 2]:
        beta = betas[uid]
        # z_part = PartQuantity(
        #     "PART_X1",
        #     r"$r^\mathrm{part}$",
        #     plot_coords=[0, 0],
        #     uids=[uid],
        # )
        custom_spaceTimeHeatmaps.append(
            SpaceTimeHeatmap(
                f"Dust{uid}_RHO",
                r"$\rho^\mathrm{dust}$",
                plot_coords=[uid, 0],
                title=rf"$\beta \equiv s \rho^\mathrm{{part}} = {beta}$",
                cmap="inferno",
                norm="linear",
                ref_function=analytical_trajectory(beta),
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

    SpaceTimeHeatmap.suptitle = "Particle trajectory over gas density\n(EPSTEIN)"

    # custom_partQuantities = [z_part]

    runContext = RunContext(
        task,
        projectPath,
        partFolder="/home/dfang/Code/idefix_setups/ThomasDrift/setup_l",
        active_directions=[0],
    )
    pipeline = Pipeline(runContext, spaceTimeHeatmaps=custom_spaceTimeHeatmaps)

    pipeline.run()
