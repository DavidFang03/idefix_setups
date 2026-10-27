#pragma once

#include "../../shared/wind/params.hpp"

using namespace Params;
namespace Wind {

KOKKOS_INLINE_FUNCTION real window(real R, real z, real Rin, real epsilon, real Hideal, real trSmoothing) {
  real R0 = FMAX(R, Rin);
  // real zh = z / (R0 * epsilon);
  real zh = z / (R0 * epsilon);
  // return tanh((fabs(z) - Hideal * R * epsilon) / trSmoothing);
  return tanh((fabs(zh) - Hideal) / (trSmoothing));
}

#ifndef DISABLE_MHD
void Ambipolar(DataBlock &data, real t, IdefixArray3D<real> &xAin) {
  IdefixArray3D<real> xA = xAin;
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray1D<real> x3 = data.x[KDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  real Hideal = HidealGlob;
  real epsilon = epsilonGlob;
  real AmMid = AmMidGlob;
  real etamax = 10 * epsilon * epsilon; // Corresponds to Rm=0.1
  real waveKillWidth = 0.1;
  real trSmoothing = trSmoothingGlob;
  real Rin = RinGlob;

  idefix_for(
      "Ambipolar", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = get_R(x1(i), x2(j), x3(k));
        real z = get_z(x1(i), x2(j), x3(k));
        R = FMAX(R, Rin);
        real Omega = pow(R, -1.5);

        real Am = 2 * AmMid / (1.0 - window(R, z, Rin, epsilon, Hideal, trSmoothing));

        real B2 = Vc(BX1, k, j, i) * Vc(BX1, k, j, i) + Vc(BX2, k, j, i) * Vc(BX2, k, j, i) + Vc(BX3, k, j, i) * Vc(BX3, k, j, i);
        real eta = B2 / (Omega * Am * Vc(RHO, k, j, i));

        if (eta > etamax)
          xA(k, j, i) = etamax / B2;
        else
          xA(k, j, i) = 1.0 / (Omega * Am * Vc(RHO, k, j, i));

        // Kill it at the radial boundaryloop
        // if (x1(i) / Rin < Rin * (1 + waveKillWidth)) {
        //   real w = (x1(i) - Rin) / (Rin * waveKillWidth);

        //   xA(k, j, i) = xA(k, j, i) * w;
        // }
      });
}

void Resistivity(DataBlock &data, real t, IdefixArray3D<real> &etain) {
  IdefixArray3D<real> eta = etain;
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray1D<real> x3 = data.x[KDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  real epsilon = epsilonGlob;
  real trSmoothing = trSmoothingGlob;
  real Hideal = HidealGlob;
  real Rin = RinGlob;

  real Rm0copy = Rm0;
  real etaBuffer0 = etab0;

  idefix_for(
      "Resistivity", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real r = get_r(x1(i), x2(j), x3(k));
        real R = get_R(x1(i), x2(j), x3(k));
        real z = get_z(x1(i), x2(j), x3(k));

        real Rinf = ONE_F;

        real R0 = FMAX(R, Rin);

        // Buffer region at inner radius. Linear damping.
        real EtaBuffer = etaBuffer0 * epsilon * epsilon * 0.05 * FMAX((1.25 * Rinf - r), 0.0); // # [Rinf, Rinf+0.25R0]

        // Precription of Roberts,Latter,Lesur (2026): Rm = 2 (rho0 r0)/(rho R) * (1-window)^(-1)
        // Rm -> eta by using Rm = H^2*Omega/eta => eta= epsilon**2 * R^(2-1.5) / Rm = epsilon**2 * R**1.5 *rho * (1-window)/ (2 * Rm0)
        // Even though Rm goes to infinity, no need for a Rm_max because writing directly eta here.

        real TransDC = 0.5 * (1.0 - window(R, z, Rin, epsilon, Hideal, trSmoothing));

        real etaO = epsilon * epsilon * Rin * sqrt(Rin) * Vc(RHO, k, j, i) / Rm0copy * TransDC + EtaBuffer;

        eta(k, j, i) = etaO;
      });
}
#endif

KOKKOS_INLINE_FUNCTION real temperature(real x1, real x2, real x3, real epsilon, real epsilonTop, real Rin, real Hideal, real trSmoothingTemp) {
  real R = get_R(x1, x2, x3);
  real z = get_z(x1, x2, x3);
  real R0 = FMAX(R, Rin);

// return epsilon * epsilon / r;
#ifndef CORONA
  return epsilon * epsilon / R0;
#else
  // real Tdisk = epsilon * epsilon / r;
  // real Tcorona = epsilonTop * epsilonTop / r;
  real Tdisk = epsilon * epsilon / R0;
  real Tcorona = epsilonTop * epsilonTop / R0;
  return 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * window(R, z, Rin, epsilon, Hideal, trSmoothingTemp);
// real Zh = FABS(z / R0) / epsilon;
// return 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * tanh((Zh - Hideal) / trSmoothingTemp);
#endif
}

} // namespace Wind