#pragma once

#include "../../shared/wind/floor.hpp"

using namespace Params;

namespace Wind {
KOKKOS_INLINE_FUNCTION real density_initial(real x1, real x2, real x3, real epsilon, real Rin, real densityFloor0) {
  real r = get_r(x1, x2, x3);
  real R = get_R(x1, x2, x3);
  real Rmin = FMAX(R, Rin);

  real cs2 = epsilon * epsilon / Rmin;
  real rho = pow(Rmin, -1.5) * exp((1.0 / r - 1.0 / Rmin) / cs2);

  real densityFloor = Wind::computeDensityFloor(x1, x2, x3, densityFloor0, Rin, epsilon);
  if (rho < densityFloor)
    rho = densityFloor;

  return rho;

  // real z = get_z(x1, x2, x3);
  // real H = epsilon * Rmin;
  // return pow(Rmin, -1.5) * exp(-(z * z) / (2 * H * H));
}

KOKKOS_INLINE_FUNCTION real vphi_initial(real x1, real x2, real x3, real epsilon, real Rin) {
  real r = get_r(x1, x2, x3);
  real R = get_R(x1, x2, x3);
  real z = get_z(x1, x2, x3);
  real Rmin = FMAX(R, Rin);

  return R * pow(Rmin, -1.5) * sqrt(FMAX(R / r - 2.5 * (epsilon * epsilon), 0.0));

  if (R >= Rin) {
    return 1.0 / sqrt(R) * sqrt(FMAX(1.0 - 2.5 * (epsilon * epsilon / Rmin), 0.0));
  } else {
    return 1.0 / sqrt(Rin) * sqrt(FMAX(Rin / r - 2.5 * epsilon * epsilon, 0.0));
    // return (R / Rin) * vphi_edge; // Continuous solid body rotation
  }
}

// KOKKOS_INLINE_FUNCTION real vphi_initial(real x1, real x2, real x3, real epsilon, real Rin) {
//   real r = get_r(x1, x2, x3);
//   real R = get_R(x1, x2, x3);

//   if (R < 1e-12)
//     return 0.0;

//   real vk0_sq = 1.0 / R; // GM / R assuming GM = 1
//   real vphi_sq = 0.0;
//   real eps2 = epsilon * epsilon;

//   if (R >= Rin) {
//     // Outer region (q = -1.0, p = -1.5)
//     vphi_sq = vk0_sq * (R / r - 2.5 * eps2);
//   } else {
//     // Inner region (cs^2 = eps2 / Rin)
//     // Continuous at R = Rin and in exact radial hydrostatic balance
//     real R_over_r = R / r;
//     vphi_sq = vk0_sq - 1.5 * (eps2 / Rin) + (eps2 / (R * R)) * (1.0 - R_over_r * R_over_r * R_over_r);
//   }

//   return sqrt(FMAX(vphi_sq, 0.0));
// }

void MySourceTerm(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];
  real epsilonTop = epsilonTopGlob;
  real epsilon = epsilonGlob;
  real dt = dtin;
  real Hideal = HidealGlob;
  real trSmoothingTemp = trSmoothingTempGlob;
  real densityFloor0 = densityFloorGlob;
  real Rin = RinGlob;

  real gamma_m1 = gammaGlob - 1.0;
  real tau0 = tauGlob;

  idefix_for(
      "MySourceTerm", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = get_z(x1(i), x2(j), x3(k));
        real R = get_R(x1(i), x2(j), x3(k));
        real R0 = FMAX(R, Rin);

        real Teff = Wind::temperature(x1(i), x2(j), x3(k), epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);

        real Ptarget = Teff * Vc(RHO, k, j, i);
        // real tau = tau0 * (FMIN(pow(R, 1.5), 1.0));
        real tau = tau0 * pow(R0, 1.5);

        real exp_factor = 1.0 - exp(-dt / tau);
        real dE_thermal = -(Vc(PRS, k, j, i) - Ptarget) / gamma_m1 * exp_factor; // better?
        Uc(ENG, k, j, i) += dE_thermal;

    // Uc(ENG, k, j, i) += -dt * (Vc(PRS, k, j, i) - Ptarget) / (tau * gamma_m1);

    // inner shell relaxation

#ifdef RELAX
        real tauVel = tau;
        real Tdisk = Teff;
        if (R < 1.2) {
          // real rhoTarget = 1.0 / (R0 * sqrt(R0)) * exp(1.0 / Tdisk * (1.0 / sqrt(R0 * R0 + z * z) - 1.0 / R0));
          real rhoTarget = Wind::density_initial(x1(i), x2(j), x3(k), epsilon, Rin, densityFloor0);

          //
          // real vx3Target = 1.0 / sqrt(R0) * sqrt(FMAX(R0 / sqrt(R0 * R0 + z * z) - 2.5 * Tdisk, 0.0));
          real vphiTarget = Wind::vphi_initial(x1(i), x2(j), x3(k), epsilon, Rin);
          //
          real drho = (Vc(RHO, k, j, i) - rhoTarget) / tauVel;
          real dmx1 = Vc(RHO, k, j, i) * Vc(VX1, k, j, i) / tauVel + Vc(VX1, k, j, i) * drho;
          real dmx2 = Vc(RHO, k, j, i) * Vc(VX2, k, j, i) / tauVel + Vc(VX2, k, j, i) * drho;
          real dmx3 = Vc(RHO, k, j, i) * (Vc(VX3, k, j, i) - vphiTarget) / tauVel + Vc(VX3, k, j, i) * drho;
          real deng = Vc(VX1, k, j, i) * dmx1 + Vc(VX2, k, j, i) * dmx2 + Vc(VX3, k, j, i) * dmx3;
          //
          Uc(RHO, k, j, i) += -drho * dt;
          Uc(MX1, k, j, i) += -dmx1 * dt;
          Uc(MX2, k, j, i) += -dmx2 * dt;
          Uc(MX3, k, j, i) += -dmx3 * dt;
          Uc(ENG, k, j, i) += -deng * dt;
        }
#endif
      });
}

} // namespace Wind