#pragma once
#include "../../shared/wind/ionisation.hpp"
#include "../../shared/wind/params.hpp"

using namespace Params;

namespace Wind {
KOKKOS_INLINE_FUNCTION real computeDensityFloor(real x1, real x2, real x3, real d_floor_0, real Rin, real c0) {
  // return d_floor_0;

  real R = get_R(x1, x2, x3);
  real z = get_z(x1, x2, x3);

  real D_return;
  if (R > Rin) {
    D_return = d_floor_0 / (R * sqrt(R)) * 1.0 / (z * z + 1.2 * (c0 * R) * (c0 * R));
  } else {
    D_return = d_floor_0 / (Rin * sqrt(Rin)) * 1.0 / (z * z + 1.2 * (c0 * Rin) * (c0 * Rin));
  }
  if (D_return < 1.0e-9) {
    D_return = 1e-9;
  }
  return D_return;
}

KOKKOS_INLINE_FUNCTION real computeVaMax(real t_change, real Va_ini_max, real Va_fin_max, real t) {
  real Va_lim;

  if (t < t_change) {
    Va_lim = t * (Va_fin_max - Va_ini_max) / t_change + Va_ini_max;
  } else {
    Va_lim = Va_fin_max;
  }

  return Va_lim;
}

void InternalBoundary(Hydro *hydro, const real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];

#ifndef DISABLE_MHD
  real Va_ini_max = 50.0;
  real Va_fin_max = 8.0;
  real vAmax = Wind::computeVaMax(4.0, Va_ini_max, Va_fin_max, t);
#endif

  real densityFloor0 = densityFloorGlob;
  real Rin = RinGlob;
  real epsilon = epsilonGlob;
  real epsilonTop = epsilonTopGlob;
  real Hideal = HidealGlob;
  real trSmoothingTemp = trSmoothingTempGlob;
  real gamma_m1 = gammaGlob - 1.0;

  idefix_for(
      "InternalBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real rho_target = Vc(RHO, k, j, i);

#ifndef DISABLE_MHD
        real b2 = EXPAND(Vc(BX1, k, j, i) * Vc(BX1, k, j, i), +Vc(BX2, k, j, i) * Vc(BX2, k, j, i), +Vc(BX3, k, j, i) * Vc(BX3, k, j, i));
        real rho_va = b2 / (vAmax * vAmax);
        if (rho_va > rho_target) {
          rho_target = rho_va;
        }
#endif

        real densityFloor = Wind::computeDensityFloor(x1(i), x2(j), x3(k), densityFloor0, Rin, epsilon);
        if (densityFloor > rho_target) {
          rho_target = densityFloor;
        }

        if (rho_target > Vc(RHO, k, j, i)) {
          Vc(RHO, k, j, i) = rho_target;
          Uc(RHO, k, j, i) = rho_target; // necessary?

#ifndef ISOTHERMAL
          real temp = Wind::temperature(x1(i), x2(j), x3(k), epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);
          Vc(PRS, k, j, i) = temp * rho_target;

          real v2 = Vc(VX1, k, j, i) * Vc(VX1, k, j, i) + Vc(VX2, k, j, i) * Vc(VX2, k, j, i) + Vc(VX3, k, j, i) * Vc(VX3, k, j, i);

          real e_thermal = Vc(PRS, k, j, i) / gamma_m1;
          real e_kinetic = 0.5 * rho_target * v2;

#ifndef DISABLE_MHD
          real e_mag = 0.5 * b2;
          Uc(ENG, k, j, i) = e_thermal + e_kinetic + e_mag;
#else
          Uc(ENG, k, j, i) = e_thermal + e_kinetic;
#endif
#endif
        }
      });
}

} // namespace Wind