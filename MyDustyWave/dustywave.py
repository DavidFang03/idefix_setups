from idefix2python import RunContext, Pipeline, LineMovie1D

projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/MyDustyWave"
task = "wave_n1"

custom_LineMovie1Ds = [
    LineMovie1D("RHO", r"$\rho$", title="Density", plot_coords=[0, 0]),
    LineMovie1D("VX1", r"$v_x$", plot_coords=[1, 0]),
    LineMovie1D("VX2", r"$v_y$", plot_coords=[2, 0]),
    LineMovie1D("VX3", r"$v_z$", plot_coords=[3, 0]),
]

runContext = RunContext(task, projectPath)
pipeline = Pipeline(runContext, movies1D=custom_LineMovie1Ds)

pipeline.run()
