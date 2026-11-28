import numpy as np
from findiff import Diff
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

inner_excluded = 20  # for the vertical integration inside the disk, the inner part has to few cells so I exclude a little part.


def interpolate(r, theta, u):
    interp = RegularGridInterpolator(
        (r, theta),
        u,
        fill_value=np.nan,
        bounds_error=False,
    )
    return interp


def cartesian_components(r, theta, ur, utheta):
    Theta = np.meshgrid(r, theta, indexing="ij")[1]
    ux = np.sin(Theta) * np.squeeze(ur) + np.cos(Theta) * np.squeeze(utheta)
    uz = np.cos(Theta) * np.squeeze(ur) - np.sin(Theta) * np.squeeze(utheta)
    return ux, uz


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


def curl_phi(r, theta, u_r, u_theta, acc=2):
    """
    Spherical curl:
    w_z = (1/R) * [ d(R*u_theta)/dR - d(u_r)/dTheta ]
    w_phi = (1/r) * [ d(r*u_theta)/dr - d(u_r)/dTheta ]."""
    is_3d = u_r.ndim == 3 and u_r.shape[-1] == 1

    u_r_2d = u_r.squeeze(axis=-1) if is_3d else u_r
    u_theta_2d = u_theta.squeeze(axis=-1) if is_3d else u_theta

    assert r.ndim == 1 and theta.ndim == 1

    rgrid, thetagrid = np.meshgrid(r, theta, indexing="ij")

    d_dr = Diff(0, r, periodic=False, acc=acc)
    d_dtheta = Diff(1, theta, periodic=False, acc=acc)

    term_r = d_dr(rgrid * u_theta_2d)
    term_theta = d_dtheta(u_r_2d)

    w_phi = (term_r - term_theta) / rgrid
    return w_phi[:, :, np.newaxis] if is_3d else w_phi


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

    def OmegaK(self, R):
        return R ** (-1.5)

    def vK(self, R):
        return R * self.OmegaK(R)

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

    # def flux_function(self, v):
    #     """
    #     (Roberts & Latter 2026 Eq. 23) to obtain Fig A1.
    #     = Flux threading the midplane. Therefore, r=R.
    #     """
    #     r0 = self.grid.X1[0]
    #     jmid = np.searchsorted(self.grid.X2_med, np.pi / 2)

    #     br = np.squeeze(v["BX1"].data)
    #     btheta = np.squeeze(v["BX2"].data)
    #     dr = np.diff(self.grid.X1)
    #     dtheta = np.diff(self.grid.X2)
    #     theta = self.grid.X2_med

    #     lhs = r0**2 * np.sum(np.sin(theta[:jmid]) * br[0, :jmid] * dtheta[:jmid])
    #     btheta_eq = 0.5 * (btheta[:, jmid - 1] + btheta[:, jmid])
    #     rhs = []
    #     for i, r in enumerate(self.grid.X1_med):
    #         rhs.append(
    #             -(
    #                 np.sum(self.grid.X1_med[0:i] * btheta_eq[0:i] * dr[0:i])
    #                 + r * btheta_eq[i] * (r - self.grid.X1[i])
    #             )
    #         )

    #     psi = lhs + np.asarray(rhs)
    #     return psi
    def flux_function(self, v, z=0, verify=True, rtol=1e-2):
        """
        Compute the poloidal magnetic flux function Psi(r, z) at constant altitude z
        evaluated directly on self.grid.X1_med (Roberts & Latter 2026 Eq. 23).
        """
        r_faces = self.grid.X1
        theta_faces = self.grid.X2
        r_med = self.grid.X1_med
        theta_med = self.grid.X2_med

        br = np.squeeze(v["BX1"].data)
        btheta = np.squeeze(v["BX2"].data)

        assert br.shape == (len(r_med), len(theta_med)), (
            f"BX1 shape {br.shape} != {(len(r_med), len(theta_med))}"
        )
        assert btheta.shape == (len(r_med), len(theta_med)), (
            f"BX2 shape {btheta.shape} != {(len(r_med), len(theta_med))}"
        )

        interp_br = RegularGridInterpolator(
            (r_med, theta_med), br, bounds_error=False, fill_value=None
        )
        interp_bth = RegularGridInterpolator(
            (r_med, theta_med), btheta, bounds_error=False, fill_value=None
        )

        r0 = r_faces[0]

        # 1. Inner boundary flux along r = r0 from theta_min to theta_0 = arccos(z / r0)
        if abs(z) < r0:
            theta_0 = np.arccos(z / r0)
            assert theta_faces[0] <= theta_0 <= theta_faces[-1], (
                f"theta_0={theta_0:.3f} outside grid [{theta_faces[0]:.3f}, {theta_faces[-1]:.3f}]."
            )
            th_cap = np.linspace(theta_faces[0], theta_0, 500)
            br_cap = interp_br(np.column_stack([np.full_like(th_cap, r0), th_cap]))
            psi_inner = np.trapezoid(r0**2 * np.sin(th_cap) * br_cap, th_cap)
            r_start = r0
        else:
            psi_inner = 0.0
            r_start = abs(z)

        # 2. Integrate horizontally along z = const at X1_med radii
        psi = np.full_like(r_med, np.nan, dtype=float)
        valid = r_med >= r_start
        r_valid = r_med[valid]
        th_valid = np.arccos(np.clip(z / r_valid, -1.0, 1.0))
        in_theta = (th_valid >= theta_faces[0]) & (th_valid <= theta_faces[-1])

        r_valid = r_valid[in_theta]
        assert len(r_valid) > 0, f"No grid cells intersect altitude z={z}."

        # Prepend r_start so the integration starts cleanly at the inner boundary
        prepend = r_valid[0] > r_start
        r_int = np.insert(r_valid, 0, r_start) if prepend else r_valid
        R_int = np.sqrt(np.maximum(0.0, r_int**2 - z**2))
        th_int = np.arccos(np.clip(z / r_int, -1.0, 1.0))

        pts = np.column_stack([r_int, th_int])
        bz_int = interp_br(pts) * np.cos(th_int) - interp_bth(pts) * np.sin(th_int)

        psi_int = psi_inner + cumulative_trapezoid(R_int * bz_int, R_int, initial=0.0)

        out_mask = np.zeros_like(r_med, dtype=bool)
        out_mask[np.where(valid)[0][in_theta]] = True
        psi[out_mask] = psi_int[1:] if prepend else psi_int

        # 3. Solenoidal (div B = 0) check at the outermost valid cell
        if verify:
            idx_last = np.where(out_mask)[0][-1]
            r_out = r_med[idx_last]
            th_out = np.arccos(z / r_out)

            th_check = np.linspace(theta_faces[0], th_out, 500)
            br_check = interp_br(
                np.column_stack([np.full_like(th_check, r_out), th_check])
            )
            psi_arc = np.trapezoid(r_out**2 * np.sin(th_check) * br_check, th_check)

            scale = max(np.nanmax(np.abs(psi)), np.abs(psi_arc), 1e-12)
            rel_err = abs(psi[idx_last] - psi_arc) / scale
            assert rel_err < rtol, (
                f"Flux check failed at z={z}: horizontal={psi[idx_last]:.6e}, "
                f"spherical_arc={psi_arc:.6e} (rel_err={rel_err:.2%})."
            )

        return np.astype(psi, np.float32)

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
        vb = v["EPHI"].data[:, jmid] / Bz[:, jmid]
        vK = self.vK(self.grid.X1_med)
        result = vb / vK[:, None]
        return result


class FluxTransport:
    def __init__(self, grid, ds):
        r, theta = grid.X1_med, grid.X2_med
        self.r, self.theta = r, theta
        self.vx, self.vz = cartesian_components(
            r, theta, ds["VX1"].data, ds["VX2"].data
        )
        self.Bx, self.Bz = cartesian_components(
            r, theta, ds["BX1"].data, ds["BX2"].data
        )
        self.ds = ds

        self.Jphi = np.squeeze(curl_phi(r, theta, ds["BX1"].data, ds["BX2"].data))

        self._advection_term = self.vx - (self.vz * self.Bx) / self.Bz

        self._ohmic_term = np.squeeze(self.ds["ETA"].data) * self.Jphi / self.Bz

        self._ambipolar_term = np.squeeze(self.ds["AM"].data) * self.Jphi * self.Bz

        self._from_idefix = np.squeeze(self.ds["EPHI"].data) / self.Bz

    def at(self, x: float, Z: np.ndarray):
        assert x > 0
        self.points = np.column_stack(
            (np.sqrt(x**2 + Z**2), np.pi / 2 - np.atan2(Z, x))
        )
        return self

    @property
    def advection_term(self):
        return interpolate(self.r, self.theta, self._advection_term)(self.points)

    @property
    def ohmic_term(self):
        return interpolate(self.r, self.theta, self._ohmic_term)(self.points)

    @property
    def ambipolar_term(self):
        return interpolate(self.r, self.theta, self._ambipolar_term)(self.points)

    @property
    def total(self):
        return interpolate(
            self.r,
            self.theta,
            self._advection_term + self._ohmic_term + self._ambipolar_term,
        )(self.points)

    @property
    def from_idefix(self):
        return interpolate(self.r, self.theta, self._from_idefix)(self.points)
