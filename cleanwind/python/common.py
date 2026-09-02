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


def plasmabeta(v):
    P = v.data["PRS"]
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return 8 * np.pi * P / B2


class WindyDisk:
    def __init__(self, inidata, gridInfo, r=50):
        """
        Accretion rate at radius r
        """
        self.inidata = inidata
        self.r = r
        self.gridInfo = gridInfo

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
    #     jm = np.searchsorted(v.theta, thetatop)
    #     jp = np.searchsorted(v.theta, thetabottom)
    #     i = np.searchsorted(v.r, self.r)

    #     mloss = 4*np.pi*self.r**2 *  np.sum(rho[jm:jp, :] * vr[jm:jp, :] * dR)

    #     macc = 0

    #     return macc

    def Macc(self, v):
        """
        (Roberts & Latter 2026 Eq. 19)
        """

        thetatop, thetabottom = self.thetatop, self.thetabottom
        jm = np.searchsorted(v.theta, thetatop)
        jp = np.searchsorted(v.theta, thetabottom)
        i = np.searchsorted(v.r, self.r)

        rho = v.data["RHO"][jm:jp, i]
        vr = v.data["VX1"][jm:jp, i]
        dtheta = np.diff(v.thetal)[jm:jp]
        theta = v.theta[jm:jp]

        macc = -2 * np.pi * self.r**2 * np.sum(np.sin(theta) ** 2 * rho * vr * dtheta)

        return macc

    def flux_function(self, v):
        """
        (Roberts & Latter 2026 Eq. 23) to obtain Fig A1.
        = Flux threading the midplane. Therefore, r=R.
        """

        r0 = v.r[0]

        jmid = np.searchsorted(v.theta, np.pi / 2)

        br = v.data["BX1"]
        btheta = v.data["BX2"]
        dr = np.diff(v.rl)
        dtheta = np.diff(v.thetal)
        theta = v.theta

        lhs = r0**2 * np.sum(
            np.sin(theta)[:jmid, None] * br[:jmid, 0] * dtheta[:jmid, None]
        )
        rhs = []
        for i, r in enumerate(v.r):
            rhs.append(-np.sum(v.r[0:i] * btheta[jmid, 0:i] * dr[0:i]))

        psi = lhs + np.asarray(rhs)
        return psi

    def Sigma(self, v):
        """
        Vertically integrated density in the disk
        """
        # To get a cylindrical grid I interpolate
        Sigma_list = [np.nan] * inner_excluded
        interp = RegularGridInterpolator(
            (v.r, v.theta),
            v.data["RHO"].T,
            fill_value=np.nan,
            bounds_error=False,
        )

        # Evaluate for a cylindrical grid R, z
        for R in v.r[inner_excluded:]:
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
            np.cos(v.theta[:, None]) * v.data["VX1"]
            - np.sin(v.theta[:, None]) * v.data["VX2"]
        )

    def vz_c(self, v):
        """
        along cylindrical radius
        """
        interp_vr = RegularGridInterpolator(
            (v.r, v.theta),
            v.data["VX1"].T,
            fill_value=np.nan,
            bounds_error=False,
        )
        interp_vtheta = RegularGridInterpolator(
            (v.r, v.theta),
            v.data["VX2"].T,
            bounds_error=False,
            fill_value=np.nan,  # some part at the outer edge will be excluded.
        )

        R = v.r
        z0 = self.epsilon * self.Hideal * R
        theta0 = np.pi / 2 - np.arctan2(z0, R)
        r0 = np.sqrt(R**2 + z0**2)
        pts = np.column_stack((r0, theta0))

        vz = np.cos(theta0) * interp_vr(pts) - np.sin(theta0) * interp_vtheta(pts)
        vz[:inner_excluded] = np.nan  # excluding first point
        return vz

    def rho_c(self, v):
        interp_rho = RegularGridInterpolator(
            (v.r, v.theta),
            v.data["RHO"].T,
            bounds_error=False,
            fill_value=np.nan,  # some part at the outer edge will be excluded.
        )
        R = v.r
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

        OmegaK = v.r ** (-1.5)
        zetap = self.rho_c(v) * self.vz_c(v) / (np.asarray(self.Sigma(v)) * OmegaK)
        return zetap

    def time(self, v):
        return v.t[0] * np.ones(len(v.r))

    def totalmass(self, v):
        dr = np.diff(v.rl)
        dtheta = np.diff(v.thetal)
        return np.sum(v.data["RHO"] * v.r * dtheta[:, None] * dr)

    def midplane_beta(self, v):
        jmid = np.searchsorted(v.theta, np.pi / 2)
        return plasmabeta(v)[jmid, :]

    def bz(self, v):
        return (
            np.cos(v.theta[:, None]) * v.data["BX1"]
            - np.sin(v.theta[:, None]) * v.data["BX2"]
        )

    def bz_midplane(self, v):
        jmid = np.searchsorted(v.theta, np.pi / 2)
        return self.bz(v)[jmid, :]

    def Ephi_midplane(self, v):
        jmid = np.searchsorted(v.theta, np.pi / 2)
        return v.data["Ephi"][jmid, :]

    def vB(self, v):
        """
        Not time averaged
        """
        jmid = np.searchsorted(v.theta, np.pi / 2)
        Bz = self.bz(v)
        return v.data["Ephi"][jmid, :] / Bz[jmid, :]
