import numpy as np


def fit(X, Y, deg, start=0, end=1):
    index_start = int(start * (len(X) - 1))
    index_end = int(end * (len(X) - 1))
    params, cov = np.polyfit(
        X[index_start:index_end], Y[index_start:index_end], deg=deg, cov=True
    )
    return params, np.diag(cov)


def integrate(df, f0, times):
    values = [f0]
    dt = np.diff(times)
    for i, t in enumerate(times[:-1]):
        previousValue = values[-1]
        value = previousValue + df(previousValue) * dt[i]
        values += [value]
    return values


def vK(r):
    return r ** (-0.5)


class Fluid:
    def __init__(self, cs0, csSlope, sigma0, sigmaSlope, tau0, r0=2, z0=0.1):
        self.cs0 = cs0
        self.csSlope = csSlope
        self.sigma0 = sigma0
        self.sigmaSlope = sigmaSlope
        self.rhoSlope = sigmaSlope - 1
        # # HSlope = csSlope + 0.5
        # # self.rhoSlope = sigmaSlope + csSlope - 0.5
        # self.rhoSlope = sigmaSlope
        # self.cs0 = 0.05
        # self.csSlope = -0.5
        # self.rhoSlope = -1.5
        self.tau0 = tau0

        self.r0 = r0
        self.z0 = z0

    def eta(self, r):
        cs0 = self.cs0
        csSlope = self.csSlope
        rhoSlope = self.rhoSlope
        cs2 = cs0**2 * np.pow(r, 2 * csSlope)
        return cs2 / (vK(r) ** 2) * (2 * csSlope + rhoSlope)

    def Stokes(self, r):
        tau0 = self.tau0

        # return tau0
        OmegaK = r ** (-1.5)
        return tau0 * OmegaK
        # return tau0 * self.sigma0 * OmegaK / r ** (self.csSlope)

    def vrDrift(self, r):
        st = self.Stokes(r)
        return self.eta(r) * vK(r) / (st + 1 / st)

    def vzSettling_approx(self, z):
        r = 2.0
        st = self.Stokes(r)
        OmegaK = r ** (-1.5)
        return -OmegaK * st * z

    def azSettling(self, z, vz, t):
        r = 2.0
        tstop = self.tau0

        OmegaK = r ** (-1.5)
        return -vz / tstop - z * OmegaK**2

    def zSettling(self, t):
        Om = self.r0 ** (-1.5)
        st = self.tau0 * Om
        sigmap = -1 / (2 * self.tau0) + Om * np.sqrt(1 / (4 * st**2) - 1)
        sigmam = -1 / (2 * self.tau0) - Om * np.sqrt(1 / (4 * st**2) - 1)
        A = self.z0 * sigmam / (sigmap - sigmam)
        B = -A * sigmap / sigmam
        return -A * np.exp(sigmap * t) - B * np.exp(sigmam * t)


def solve_2nd_order_ode(f, u0, du0, times):
    """
    u" = f(u',u,t)
    """
    u = [u0]
    du = [du0]
    dt = np.diff(times)
    for i, t in enumerate(times[:-1]):
        u_prev = u[-1]
        du_prev = du[-1]

        du += [du_prev + f(u_prev, du_prev, t) * dt[i]]
        u += [u_prev + du_prev * dt[i]]
    return u
