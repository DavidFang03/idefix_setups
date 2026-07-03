import numpy as np
from scipy.integrate import solve_ivp


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
    def __init__(self, cs0, csSlope, sigma0, sigmaSlope, beta, r0, z0, drag):
        self.cs0 = cs0
        self.csSlope = csSlope
        self.sigma0 = sigma0
        self.sigmaSlope = sigmaSlope
        self.rhoSlope = sigmaSlope - 1
        self.beta = beta
        self.drag = drag

        self.r0 = r0
        self.z0 = z0

        self.sol_r = solve_ivp(
            self.vrDrift,
            [0, 10],
            [self.r0],
            dense_output=True,
            # method="LSODA",
            method="DOP853",
        ).sol

    def cs(self, r):
        return self.cs0 * r ** (self.csSlope)

    def sigma(self, r):
        return self.sigma0 * r ** (self.sigmaSlope)

    def eta(self, r):
        csSlope = self.csSlope
        rhoSlope = self.rhoSlope
        return (self.cs(r) / vK(r)) ** 2 * (2 * csSlope + rhoSlope)

    def Stokes(self, r, z):
        if self.drag.lower() in ["epstein", "size"]:
            cs = self.cs0 * r ** (self.csSlope)
            h = cs * r**1.5
            rho = self.sigma0 * r ** (self.rhoSlope) * np.exp(-(z**2) / (2 * h**2))
            tau = self.beta / (cs * rho)

        elif self.drag.lower() in ["tau"]:
            tau = self.beta

        else:
            raise NotImplementedError(f"Not implemented drag: {self.drag}")

        # return tau0
        OmegaK = r ** (-1.5)
        return tau * OmegaK
        # return tau0 * self.sigma0 * OmegaK / r ** (self.csSlope)

    def vrDrift(self, t, r):
        st = self.Stokes(r, z=0)
        return self.eta(r) * vK(r) / (st + 1 / st)

    def arDrift(self, r, vr, t):
        st = self.Stokes(r, z=0)
        vtheta = vK(r) - 0.5 * st * vr + 0.5 * self.eta(r) * vK(r)
        tstop = st * r**1.5
        centrig = vtheta**2 / r
        grav = -1 / r**2
        drag = -1 / tstop * (vr - 0)  # vgas_r = 0 because inviscid
        return centrig + grav + drag

    def drift_system(self, t, y):
        """
        y[0] = position (r)
        y[1] = velocity (vr)
        """
        r, vr = y
        dr_dt = vr
        dvr_dt = self.arDrift(r, vr, t)

        return [dr_dt, dvr_dt]

    def vzSettling_approx(self, z):
        r = 2.0
        st = self.Stokes(r, z)
        OmegaK = r ** (-1.5)
        return -OmegaK * st * z

    def azSettling(self, z, vz, t):
        r = self.sol_r(t)[0]
        st = self.Stokes(r, z)
        OmegaK = r ** (-1.5)
        tstop = st / OmegaK

        return -vz / tstop - z * OmegaK**2

    def settling_system(self, t, y):
        """
        y[0] = position (z)
        y[1] = velocity (vz)
        """
        z, vz = y
        dz_dt = vz
        dvz_dt = self.azSettling(z, vz, t)

        return [dz_dt, dvz_dt]

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
