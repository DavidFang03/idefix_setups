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

  idefix_for(
      "Ambipolar", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real Rin = ONE_F;
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

  real Rin = 1.0; // =1
  real Rm0copy = Rm0;
  real etaBuffer0 = etab0;

  idefix_for(
      "Resistivity", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real r = get_r(x1(i), x2(j), x3(k));
        real R = get_R(x1(i), x2(j), x3(k));
        real z = get_z(x1(i), x2(j), x3(k));

        real R0 = FMAX(R, Rin);

        // Buffer region at inner radius. Linear damping.
        real EtaBuffer = etaBuffer0 * epsilon * epsilon * 0.05 * FMAX((1.25 * Rin - r), 0.0); // # [Rin, Rin+0.25R0]

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
  return epsilon * epsilon / R0;

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
  IdefixArray1D<real> x3 = data->x[KDIR];
  real epsilonTop = epsilonTopGlob;
  real epsilon = epsilonGlob;
  real dt = dtin;
  real Hideal = HidealGlob;
  real trSmoothingTemp = trSmoothingTempGlob;
  real densityFloor0 = densityFloorGlob;

  real gamma_m1 = gammaGlob - 1.0;
  real tau0 = tauGlob;

  idefix_for(
      "MySourceTerm", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = get_z(x1(i), x2(j), x3(k));
        real R = get_R(x1(i), x2(j), x3(k));
        real Rin = 1.0;
        real R0 = FMAX(R, Rin);

        real Teff = temperature(x1(i), x2(j), x3(k), epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);

        real Ptarget = Teff * Vc(RHO, k, j, i);
        // real tau = tau0 * (FMIN(pow(R, 1.5), 1.0));
        real tau = tau0 * pow(R0, 1.5);

        real exp_factor = 1.0 - exp(-dt / tau);
        real dE_thermal = -(Vc(PRS, k, j, i) - Ptarget) / gamma_m1 * exp_factor; // better?
        Uc(ENG, k, j, i) += dE_thermal;

        // Uc(ENG, k, j, i) += -dt * (Vc(PRS, k, j, i) - Ptarget) / (tau * gamma_m1);

        // inner shell relaxation

        // real tauVel = tau;
        // real Tdisk = Teff;
        // if (R < 1.2) {
        //   real rhoTarget = 1.0 / (R0 * sqrt(R0)) * exp(1.0 / Tdisk * (1.0 / sqrt(R0 * R0 + z * z) - 1.0 / R0));
        //   real densityFloor = computeDensityFloor(R, z, densityFloor0, Rin, epsilon);
        //   if (rhoTarget < densityFloor)
        //     rhoTarget = densityFloor;

        //   real vx3Target = 1.0 / sqrt(R0) * sqrt(FMAX(R0 / sqrt(R0 * R0 + z * z) - 2.5 * Tdisk, 0.0));

        //   real drho = (Vc(RHO, k, j, i) - rhoTarget) / tauVel;
        //   real dmx1 = Vc(RHO, k, j, i) * Vc(VX1, k, j, i) / tauVel + Vc(VX1, k, j, i) * drho;
        //   real dmx2 = Vc(RHO, k, j, i) * Vc(VX2, k, j, i) / tauVel + Vc(VX2, k, j, i) * drho;
        //   real dmx3 = Vc(RHO, k, j, i) * (Vc(VX3, k, j, i) - vx3Target) / tauVel + Vc(VX3, k, j, i) * drho;
        //   real deng = Vc(VX1, k, j, i) * dmx1 + Vc(VX2, k, j, i) * dmx2 + Vc(VX3, k, j, i) * dmx3;

        //   Uc(RHO, k, j, i) += -drho * dt;
        //   Uc(MX1, k, j, i) += -dmx1 * dt;
        //   Uc(MX2, k, j, i) += -dmx2 * dt;
        //   Uc(MX3, k, j, i) += -dmx3 * dt;
        //   Uc(ENG, k, j, i) += -deng * dt;
        // }
      });
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
  real Rin = 1.0;
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