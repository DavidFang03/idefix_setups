#pragma once

#include "../../shared/wind/params.hpp"

using namespace Params;

namespace Wind {
void Ambipolar(DataBlock &data, real t, IdefixArray3D<real> &xAin) {
  IdefixArray3D<real> xA = xAin;
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  real Hideal = HidealGlob;
  real epsilon = epsilonGlob;
  real AmMid = AmMidGlob;
  real etamax = 10 * epsilon * epsilon; // Corresponds to Rm=0.1
  real Rin = 1.0;
  real waveKillWidth = 0.1;
  real trSmoothing = trSmoothingGlob;

  idefix_for(
      "Ambipolar", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = x1(i) * cos(x2(j));
        real R = FMAX(FABS(x1(i) * sin(x2(j))), ONE_F);
        real Omega = pow(R, -1.5);

        real zh = z / (R * epsilon); // z in units of disc scale height h=R*epsilon
        real Am;

        Am = AmMid / (0.5 * (1 - tanh((fabs(zh) - Hideal) / trSmoothing)));

        real B2 = Vc(BX1, k, j, i) * Vc(BX1, k, j, i) + Vc(BX2, k, j, i) * Vc(BX2, k, j, i) + Vc(BX3, k, j, i) * Vc(BX3, k, j, i);
        real eta = B2 / (Omega * Am * Vc(RHO, k, j, i));

        if (eta > etamax)
          xA(k, j, i) = etamax / B2; //! SAME THING????
        else
          xA(k, j, i) = 1.0 / (Omega * Am * Vc(RHO, k, j, i)); //! SAME THING????

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

  real R0 = data.mygrid->xbeg[IDIR]; // =1
  // The constant pre factor for R_m in the dead zone
  real Rm0copy = Rm0;
  real etaBuffer0 = etab0;

  idefix_for(
      "Resistivity", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = x1(i) * cos(x2(j));
        real R = x1(i) * sin(x2(j)); // cylindric R
        real Ri = FMAX(R0, R);
        real r = x1(i);
        real zh = z / (R * epsilon); //=1 ??? =z/H
        real Omega = pow(R, -1.5);
        // Inner region damping. Buffer region
        real EtaBuffer = etaBuffer0 * epsilon * epsilon * 0.05 * FMAX((1.25 * R0 - r), 0.0); // # [R0, R0+0.25R0]

        // Transition across disk and corona (want eta to be zero outside the
        // disk dead zone)
        real TransDC = 0.5 * (1 - tanh((fabs(zh) - Hideal) / (trSmoothing)));
        // Transition across the DZI (want eta to be zero outside the disk dead
        // zone)
        // real TransDZI = 0.5 * (1 + tanh((R - 10.0) / (0.1 * trSmoothing)));
        // //! cause eta to diverge->crash
        // // The expression for the magnetic Reynolds number (R_m) in the dead
        // // zone of the disk
        // real RmDZ = RmDZ0 * 1 / (Vc(RHO, k, j, i) * Ri);
        // // The expression for the Ohmic resistivity in the dead zone of the
        // disk real etaDZ = pow(epsilon * Ri, 2) * Omega /
        //              RmDZ; // exact expresion to get eta from Rm.
        // The final expression for the Ohmic resistivity (includes the buffer
        // zone contribution). Makes R_M = 50 at the inner edge of the DZI
        // (feels appropriate - maybe disk slightly heavy? - discuss) eta(k,j,i)
        // = (pow(Ri,1.5))*Vc(RHO,k,j,i)/(10.0*10.0*pow(10,0.5))*TransDC +
        // EtaBuffer;
        // eta(k,j,i) = EtaBuffer +
        // (pow(Ri,1.5))*Vc(RHO,k,j,i)/2500*TransDZI*TransDC; eta(k,j,i) =
        // EtaBuffer
        // + etaDZ;
        // eta(k, j, i) = etaDZ * TransDC * TransDZI + EtaBuffer;
        // eta(k, j, i) = EtaBuffer;
        // real eta0 = pow(epsilon * Ri, 2) * Omega / Rm0copy;
        // eta0 = 0;
        // Precription of Roberts,Latter,Lesur (2026): Rm propto 1/(rho R)
        eta(k, j, i) = epsilon * epsilon * pow(Ri, 1.5) * Vc(RHO, k, j, i) / Rm0copy * TransDC + EtaBuffer;
        // eta(k, j, i) = eta0 * (pow(Ri, 1.5)) * Vc(RHO, k, j, i) /
        //                    (10.0 * 10.0 * pow(10, 0.5)) * TransDC +
        //                EtaBuffer;
      });
}

KOKKOS_INLINE_FUNCTION real temperature(real R, real z, real epsilon, real epsilonTop, real Rin, real Hideal, real trSmoothingTemp) {
  real R0 = FMAX(R, Rin);
  real Zh = FABS(z / R0) / epsilon;
  real Tdisk = epsilon * epsilon / R0;
  real Tcorona = epsilonTop * epsilonTop / R0;
  return 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * tanh((Zh - Hideal) / trSmoothingTemp);
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
  real Rin = 1.0;
  real trSmoothingTemp = trSmoothingTempGlob;

  real gamma_m1 = gammaGlob - 1.0;
  real tau0 = tauGlob;

  idefix_for(
      "MySourceTerm", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real r = x1(i);
        real th = x2(j);
        real z = r * cos(th);
        real R = r * sin(th);

        real Teff = temperature(R, z, epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);

        // Cooling /heatig function
        real Ptarget = Teff * Vc(RHO, k, j, i);
        real tau = tau0 * (FMIN(pow(R, 1.5), 1.0));

        Uc(ENG, k, j, i) += -dt * (Vc(PRS, k, j, i) - Ptarget) / (tau * gamma_m1);
      });
}
} // namespace Wind