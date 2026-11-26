import numpy as np


def size(v):
    beta = v["dragcoeff"].data
    rho0 = 6.0e-10
    rhos = 1.0
    au = 1.5e11
    return beta * (rho0 * au) / rhos


def escape_velocity(ds):
    d = np.sqrt(ds["x1"].data ** 2 + ds["x2"].data ** 2 + ds["x3"].data ** 2)
    return np.sqrt(2 / d)


def OmegaK(R):
    return R ** (-1.5)


def St(ds):
    st = ds["tstop"].data * OmegaK(ds["x1"].data * np.sin(ds["x2"].data))
    return st
