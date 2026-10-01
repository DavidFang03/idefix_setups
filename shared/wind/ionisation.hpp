#pragma once

#include "../../shared/wind/params.hpp"

using namespace Params;
namespace Wind {

KOKKOS_INLINE_FUNCTION real window(real R, real z, real Rin, real epsilon, real Hideal, real trSmoothing) {
  // real R0 = max(R, Rin);
  // real zh = z / (R0 * epsilon);
  real zh = z / (R * epsilon);
  return tanh((fabs(zh) - Hideal) / (trSmoothing));
}

#ifndef DISABLE_MHD
void Ambipolar(DataBlock &data, real t, IdefixArray3D<real> &xAin) {
  IdefixArray3D<real> xA = xAin;
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  real Hideal = HidealGlob;
  real epsilon = epsilonGlob;
  real AmMid = AmMidGlob;
  real etamax = 10 * epsilon * epsilon; // Corresponds to Rm=0.1
  real waveKillWidth = 0.1;
  real trSmoothing = trSmoothingGlob;

  idefix_for(
      "Ambipolar", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = x1(i) * cos(x2(j));
        real Rin = ONE_F;
        real R = FMAX(FABS(x1(i) * sin(x2(j))), Rin);
        real Omega = pow(R, -1.5);

        real Am = 2 * AmMid / (1 - window(R, z, Rin, epsilon, Hideal, trSmoothing));

        real B2 = Vc(BX1, k, j, i) * Vc(BX1, k, j, i) + Vc(BX2, k, j, i) * Vc(BX2, k, j, i) + Vc(BX3, k, j, i) * Vc(BX3, k, j, i);
        real eta = B2 / (Omega * Am * Vc(RHO, k, j, i));

        if (eta > etamax)
          xA(k, j, i) = etamax / B2;
        else
          xA(k, j, i) = 1.0 / (Omega * Am * Vc(RHO, k, j, i));

        // Kill it at the radial boundaryloop
        if (x1(i) / Rin < Rin * (1 + waveKillWidth)) {
          real w = (x1(i) - Rin) / (Rin * waveKillWidth);

          xA(k, j, i) = xA(k, j, i) * w;
        }
      });
}

void Resistivity(DataBlock &data, real t, IdefixArray3D<real> &etain) {
  IdefixArray3D<real> eta = etain;
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  real trSmoothing = trSmoothingGlob;
  real Hideal = HidealGlob;
  real epsilon = epsilonGlob;

  real Rin = data.mygrid->xbeg[IDIR]; // =1
  real Rm0copy = Rm0;
  real etaBuffer0 = etab0;

  idefix_for(
      "Resistivity", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = x1(i) * cos(x2(j));
        real R = x1(i) * sin(x2(j));
        real r = x1(i);

        // Buffer region at inner radius. Linear damping.
        real EtaBuffer = etaBuffer0 * epsilon * epsilon * 0.05 * FMAX((1.25 * Rin - r), 0.0); // # [Rin, Rin+0.25R0]

        // Precription of Roberts,Latter,Lesur (2026): Rm = 2 (rho0 r0)/(rho R) * (1-window)^(-1)
        // Rm -> eta by using Rm = H^2*Omega/eta => eta= epsilon**2 * R^(2-1.5) / Rm = epsilon**2 * R**1.5 *rho * (1-window)/ (2 * Rm0)
        // Even though Rm goes to infinity, no need for a Rm_max because writing directly eta here.

        real TransDC = 0.5 * (1 - window(R, z, Rin, epsilon, Hideal, trSmoothing));

        eta(k, j, i) = epsilon * epsilon * pow(R, 1.5) * Vc(RHO, k, j, i) / Rm0copy * TransDC + EtaBuffer;
      });
}
#endif

KOKKOS_INLINE_FUNCTION real temperature(real r, real theta, real epsilon, real epsilonTop, real Rin, real Hideal, real trSmoothingTemp) {
  real z = r * cos(theta);
  real R = r * sin(theta);
  real R0 = FMAX(R, Rin);

  // return epsilon * epsilon / r;
  // return epsilon * epsilon / R0;

  // real Tdisk = epsilon * epsilon / r;
  // real Tcorona = epsilonTop * epsilonTop / r;
  real Tdisk = epsilon * epsilon / R0;
  real Tcorona = epsilonTop * epsilonTop / R0;
  return 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * window(R, z, Rin, epsilon, Hideal, trSmoothingTemp);
  // real Zh = FABS(z / R0) / epsilon;
  // return 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * tanh((Zh - Hideal) / trSmoothingTemp);
}

void MySourceTerm(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  real epsilonTop = epsilonTopGlob;
  real epsilon = epsilonGlob;
  real dt = dtin;
  real Hideal = HidealGlob;
  real trSmoothingTemp = trSmoothingTempGlob;

  real gamma_m1 = gammaGlob - 1.0;
  real tau0 = tauGlob;

  idefix_for(
      "MySourceTerm", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real r = x1(i);
        real th = x2(j);
        real z = r * cos(th);
        real R = r * sin(th);
        real Rin = 1.0;
        real R0 = FMAX(R, Rin);

        real Teff = temperature(r, th, epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);

        real Ptarget = Teff * Vc(RHO, k, j, i);
        // real tau = tau0 * (FMIN(pow(R, 1.5), 1.0)); // ???
        real tau = tau0 * pow(R0, 1.5);

        Uc(ENG, k, j, i) += -dt * (Vc(PRS, k, j, i) - Ptarget) / (tau * gamma_m1);
      });
}
} // namespace Wind