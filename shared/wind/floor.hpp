#pragma once
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

} // namespace Wind