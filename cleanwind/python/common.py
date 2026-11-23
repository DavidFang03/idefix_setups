import numpy as np
from scipy.interpolate import RegularGridInterpolator

inner_excluded = 20  # for the vertical integration inside the disk, the inner part has to few cells so I exclude a little part.


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


def plasmabeta(ds):
    P = ds["PRS"].data
    B2 = ds["BX1"].data ** 2 + ds["BX2"].data ** 2 + ds["BX3"].data ** 2
    return 8 * np.pi * P / B2


class WindyDisk:
    def __init__(self, inidata, grid, r=50):
        """
        Accretion rate at radius r
        """
        self.inidata = inidata
        self.r = r
        self.grid = grid

        Hideal = self.inidata["Setup"]["Hideal"]
        epsilon = self.inidata["Setup"]["epsilon"]

        self.Hideal = Hideal
        self.epsilon = epsilon

        self.thetatop = np.pi / 2 - np.atan(Hideal * epsilon)
        self.thetabottom = np.pi / 2 + np.atan(Hideal * epsilon)

    # def Mloss(self, v):
    #     """
    #     (Zhu & Stone 2018 Eq. 29)
    #     """
    #     Hideal = self.inidata["Setup"]["Hideal"]
    #     epsilon = self.inidata["Setup"]["epsilon"]
    #     thetabottom, thetatop = np.tan(Hideal)

    #     thetatop = np.pi / 2 - np.atan(Hideal * epsilon)
    #     thetabottom = np.pi / 2 + np.atan(Hideal * epsilon)
    #     jm = np.searchsorted(self.grid.X2_med, thetatop)
    #     jp = np.searchsorted(self.grid.X2_med, thetabottom)
    #     i = np.searchsorted(self.grid.X1_med, self.r)

    #     mloss = 4*np.pi*self.r**2 *  np.sum(rho[jm:jp, :] * vr[jm:jp, :] * dR)

    #     macc = 0

    #     return macc

    def temperature(self, v):
        return v["PRS"].data / v["RHO"].data

    def Macc(self, v):
        """
        (Roberts & Latter 2026 Eq. 19)
        """

        thetatop, thetabottom = self.thetatop, self.thetabottom
        jm = np.searchsorted(self.grid.X2_med, thetatop)
        jp = np.searchsorted(self.grid.X2_med, thetabottom)
        i = np.searchsorted(self.grid.X1_med, self.r)

        rho = v["RHO"].data[i, jm:jp]
        vr = v["VX1"].data[i, jm:jp]
        dtheta = np.diff(self.grid.X2)[jm:jp]
        theta = self.grid.X2_med[jm:jp]

        macc = -2 * np.pi * self.r**2 * np.sum(np.sin(theta) ** 2 * rho * vr * dtheta)

        return macc

    def flux_function(self, v):
        """
        (Roberts & Latter 2026 Eq. 23) to obtain Fig A1.
        = Flux threading the midplane. Therefore, r=R.
        """
        r0 = self.grid.X1[0]
        jmid = np.searchsorted(self.grid.X2_med, np.pi / 2)

        br = np.squeeze(v["BX1"].data)
        btheta = np.squeeze(v["BX2"].data)
        dr = np.diff(self.grid.X1)
        dtheta = np.diff(self.grid.X2)
        theta = self.grid.X2_med

        lhs = r0**2 * np.sum(np.sin(theta[:jmid]) * br[0, :jmid] * dtheta[:jmid])
        btheta_eq = 0.5 * (btheta[:, jmid - 1] + btheta[:, jmid])
        rhs = []
        for i, r in enumerate(self.grid.X1_med):
            rhs.append(
                -(
                    np.sum(self.grid.X1_med[0:i] * btheta_eq[0:i] * dr[0:i])
                    + r * btheta_eq[i] * (r - self.grid.X1[i])
                )
            )

        psi = lhs + np.asarray(rhs)
        return psi

    def Sigma(self, v):
        """
        Vertically integrated density in the disk
        """
        # To get a cylindrical grid I interpolate
        Sigma_list = [np.nan] * inner_excluded
        interp = RegularGridInterpolator(
            (self.grid.X1_med, self.grid.X2_med),
            v["RHO"].data,
            fill_value=np.nan,
            bounds_error=False,
        )

        # Evaluate for a cylindrical grid R, z
        for R in self.grid.X1_med[inner_excluded:]:
            z0 = self.epsilon * self.Hideal * R
            dz = 2 * z0 / 128
            z = np.arange(-z0, z0, dz)  # fuck it
            theta_list = np.pi / 2 - np.arctan2(z, R)

            r_list = np.sqrt(R**2 + z**2)

            # back to a spherical grid
            pts = np.column_stack((r_list, theta_list))
            Sigma_list.append(np.sum(interp(pts) * dz))
        return Sigma_list

    def vz(self, v):
        return (
            np.cos(self.grid.X2_med[:, None]) * v["VX1"].data
            - np.sin(self.grid.X2_med[:, None]) * v["VX2"].data
        )

    def vz_c(self, v):
        """
        along cylindrical radius
        """
        interp_vr = RegularGridInterpolator(
            (self.grid.X1_med, self.grid.X2_med),
            np.squeeze(v["VX1"].data),
            fill_value=np.nan,
            bounds_error=False,
        )
        interp_vtheta = RegularGridInterpolator(
            (self.grid.X1_med, self.grid.X2_med),
            np.squeeze(v["VX2"].data),
            bounds_error=False,
            fill_value=np.nan,  # some part at the outer edge will be excluded.
        )

        R = self.grid.X1_med
        z0 = self.epsilon * self.Hideal * R
        theta0 = np.pi / 2 - np.arctan2(z0, R)
        r0 = np.sqrt(R**2 + z0**2)
        pts = np.column_stack((r0, theta0))

        vz = np.cos(theta0) * interp_vr(pts) - np.sin(theta0) * interp_vtheta(pts)
        vz[:inner_excluded] = np.nan  # excluding first point
        return vz

    def rho_c(self, v):
        interp_rho = RegularGridInterpolator(
            (self.grid.X1_med, self.grid.X2_med),
            np.squeeze(v["RHO"].data),
            bounds_error=False,
            fill_value=np.nan,  # some part at the outer edge will be excluded.
        )
        R = self.grid.X1_med
        z0 = self.epsilon * self.Hideal * R
        theta0 = np.pi / 2 - np.arctan2(z0, R)
        r0 = np.sqrt(R**2 + z0**2)
        pts = np.column_stack((r0, theta0))
        rho = interp_rho(pts)
        rho[:inner_excluded] = np.nan

        return rho

    def massloss_up(self, v):
        """
        Lesur 2021 Eq. 12
        zeta_pm = (rho vz) / (Sigma OmegaK)
        function of R (cylindrical). Thus constant for self-similar model.
        """

        OmegaK = self.grid.X1_med ** (-1.5)
        zetap = self.rho_c(v) * self.vz_c(v) / (np.asarray(self.Sigma(v)) * OmegaK)
        return np.astype(zetap, np.float32)  # wtf??

    def time(self, v):
        return v.t[0] * np.ones(len(self.grid.X1_med))

    def totalmass(self, v):
        dr = np.diff(self.grid.X1)
        dtheta = np.diff(self.grid.X2)
        return np.sum(v["RHO"].data * self.grid.X1_med * dtheta[:, None] * dr)

    def midplane_beta(self, v):
        jmid = np.searchsorted(self.grid.X2_med, np.pi / 2)
        return plasmabeta(v)[jmid, :]

    def bz(self, v):
        return (
            np.cos(self.grid.X2_med[:, None]) * v["BX1"].data
            - np.sin(self.grid.X2_med[:, None]) * v["BX2"].data
        )

    def bz_midplane(self, v):
        jmid = np.searchsorted(self.grid.X2_med, np.pi / 2)
        return self.bz(v)[jmid, :]

    def Ephi_midplane(self, v):
        jmid = np.searchsorted(self.grid.X2_med, np.pi / 2)
        return v["EPHI".data][jmid, :]

    def vB(self, v):
        """
        Not time averaged
        """
        jmid = np.searchsorted(self.grid.X2_med, np.pi / 2)
        Bz = self.bz(v)
        return v["EPHI".data][jmid, :] / Bz[jmid, :]
