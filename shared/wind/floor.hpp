#pragma once
#include "../../shared/wind/params.hpp"

using namespace Params;

namespace Wind {
KOKKOS_INLINE_FUNCTION real computeDensityFloor(real R, real z, real d_floor_0, real Rin, real c0) {
  return d_floor_0;

  //     real D_return;
  // if (R > Rin) {
  //   D_return = d_floor_0 / (R * sqrt(R)) * 1.0 / (z * z + 1.2 * (c0 * R) * (c0 * R));
  // } else {
  //   D_return = d_floor_0 / (Rin * sqrt(Rin)) * 1.0 / (z * z + 1.2 * (c0 * Rin) * (c0 * Rin));
  // }
  // if (D_return < 1.0e-10) {
  //   D_return = 1e-10;
  // }
  // return D_return;
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

// void InternalBoundary(Hydro *hydro, const real t) {
//   auto *data = hydro->data;
//   IdefixArray4D<real> Vc = hydro->Vc;
//   IdefixArray4D<real> Vs = hydro->Vs;
//   IdefixArray1D<real> x1 = data->x[IDIR];
//   IdefixArray1D<real> x2 = data->x[JDIR];

//   real vAmax = computeVaMax(4.0, 50.0, 8.0, t);
//   real densityFloor0 = densityFloorGlob;
//   real Rin = 1.0;
//   real epsilon = epsilonGlob;

//   IdefixArray3D<real> array = myGlobals->array1;

//   idefix_for(
//       "InternalBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
//         real R = x1(i) * sin(x2(j));
//         real z = x1(i) * cos(x2(j));
//         // real zh = FABS(z / R) / epsilon;

//         real b2 = EXPAND(Vc(BX1, k, j, i) * Vc(BX1, k, j, i), +Vc(BX2, k, j, i) * Vc(BX2, k, j, i), +Vc(BX3, k, j, i) * Vc(BX3, k, j, i));
//         real va2 = b2 / Vc(RHO, k, j, i);
//         real myMax = vAmax;
//         // if(x1(i)<1.1) myMax=myMax/50.0;
//         if (va2 > myMax * myMax) {
//           real T = Vc(PRS, k, j, i) / Vc(RHO, k, j, i);
//           Vc(RHO, k, j, i) = b2 / (myMax * myMax);
//           Vc(PRS, k, j, i) = T * Vc(RHO, k, j, i);
//         }
//         real densityFloor = computeDensityFloor(R, z, densityFloor0, Rin, epsilon);
//         if (Vc(RHO, k, j, i) < densityFloor) {
//           array1(k, j, i) = array1(k, j, i) + densityFloor - Vc(RHO, k, j, i);

//           real T = Vc(PRS, k, j, i) / Vc(RHO, k, j, i);
//           Vc(RHO, k, j, i) = densityFloor;
//         }
//       });
// }

} // namespace Wind