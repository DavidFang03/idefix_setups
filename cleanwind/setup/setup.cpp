// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file test/MHD/AmbipolarWind/setup.cpp
//
// Last modified : 06/2026
//
// Copyright(C) by :
// - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2020 - 2026)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include "setup.hpp"
#include "../../shared/wind/bc.hpp"
#include "../../shared/wind/floor.hpp"
#include "../../shared/wind/ionisation.hpp"
#include "../../shared/wind/params.hpp"
#include "dumpImage.hpp"

#include "idefix.hpp"

#ifdef RELOAD
std::string reload_path;
#endif

using namespace Params;

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray1D<real> x3 = data.x[KDIR];
  real epsilon = epsilonGlob;
  real epsilonTop = epsilonTopGlob;
  real Hideal = HidealGlob;
  real trSmoothingTemp = trSmoothingTempGlob;
  real Rin = 1.0;
  idefix_for(
      "MySoundSpeed", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = get_R(x1(i), x2(j), x3(k));
        real R0 = FMAX(R, Rin);
        real temp = Wind::temperature(x1(i), x2(j), x3(k), epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);
        cs(k, j, i) = sqrt(temp);
      });
}

void ComputeUserVars(DataBlock &data, UserDefVariablesContainer &variables) {

  // Use Invdt as scratch array
  IdefixArray3D<real> scrh("Scratch", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

  // Ask for a computation of xA ambipolar in this scratch array

#ifndef DISABLE_MHD
  Wind::Ambipolar(data, data.t, scrh);
#endif
  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  // Make references to the user-defined arrays (variables is a container of IdefixHostArray3D)
  // Note that the labels should match the variable names in the input file
  IdefixHostArray3D<real> Am = variables["Am"];
  IdefixHostArray3D<real> InvDt = variables["InvDt"];
  IdefixHostArray3D<real> Ephi = variables["Ephi"];

  IdefixHostArray1D<real> x1 = d.x[IDIR];
  IdefixHostArray1D<real> x2 = d.x[JDIR];
  IdefixHostArray1D<real> x3 = d.x[KDIR];
  IdefixHostArray4D<real> Vc = d.Vc;
  IdefixArray3D<real>::host_mirror_type scrhHost = Kokkos::create_mirror_view(scrh);
  Kokkos::deep_copy(scrhHost, scrh);

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        real z = get_z(x1(i), x2(j), x3(k));
        real R = get_R(x1(i), x2(j), x3(k));
        real R0 = FMAX(R, ONE_F);
        real Omega = pow(R0, -1.5);
#ifndef DISABLE_MHD
        Am(k, j, i) = 1.0 / (Omega * scrhHost(k, j, i) * Vc(RHO, k, j, i));
        Ephi(k, j, i) = d.Ex3(k, j, i);
#endif
        InvDt(k, j, i) = d.InvDt(k, j, i);
      }
    }
  }
}

// Default constructor

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&Wind::UserdefBoundary);
  data.hydro->EnrollInternalBoundary(&Wind::InternalBoundary);

#ifndef ISOTHERMAL
  data.hydro->EnrollUserSourceTerm(&Wind::MySourceTerm);
#else
  data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
#endif

#ifndef DISABLE_MHD
  data.hydro->EnrollAmbipolarDiffusivity(&Wind::Ambipolar);
  data.hydro->EnrollOhmicDiffusivity(&Wind::Resistivity);
  data.hydro->EnrollEmfBoundary(&Wind::EmfBoundary);
#endif

  output.EnrollUserDefVariables(&ComputeUserVars);
  gammaGlob = data.hydro->eos->GetGamma();
  tauGlob = input.Get<real>("Setup", "tau0", 0);
  epsilonGlob = input.Get<real>("Setup", "epsilon", 0);
  epsilonTopGlob = input.Get<real>("Setup", "epsilonTop", 0);

  betaGlob = input.Get<real>("Setup", "beta", 0);
  HidealGlob = input.Get<real>("Setup", "Hideal", 0);
  AmMidGlob = input.Get<real>("Setup", "Am", 0);
  densityFloorGlob = input.Get<real>("Setup", "densityFloor", 0);
  trSmoothingGlob = input.Get<real>("Setup", "transitionSmoothing", 0);
  trSmoothingTempGlob = input.Get<real>("Setup", "transitionSmoothingTemp", 0);
  Rm0 = input.Get<real>("Setup", "Rm0", 0);
  etab0 = input.Get<real>("Setup", "etab0", 0);

#ifdef RELOAD
  reload_path = input.Get<std::string>("Setup", "reload_path", 0);
#endif
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
  // Create a host copy
  DataBlockHost d(data);

  // Make vector potential
  IdefixHostArray4D<real> A = IdefixHostArray4D<real>("Setup_VectorPotential", 3, data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

  real Rin = 1.0;
  real m = -5.0 / 4.0;
  real B0 = epsilonGlob * sqrt(2.0 / betaGlob);

#ifdef RELOAD
  DumpImage image(reload_path, &data);

  // Note that the restart dump array only contains the full (global) active domain
  // (i.e. it excludes the boundaries, but it is not decomposed accross MPI procs)
  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        int iglob = i - 2 * d.beg[IDIR] + d.gbeg[IDIR];
        int jglob = j - 2 * d.beg[JDIR] + d.gbeg[JDIR];
        int kglob = k - 2 * d.beg[KDIR] + d.gbeg[KDIR];

        d.Vc(RHO, k, j, i) = image.arrays["Vc-RHO"](kglob, jglob, iglob);

        d.Vc(PRS, k, j, i) = image.arrays["Vc-PRS"](kglob, jglob, iglob);
        d.Vc(VX1, k, j, i) = image.arrays["Vc-VX1"](kglob, jglob, iglob);
        d.Vc(VX2, k, j, i) = image.arrays["Vc-VX2"](kglob, jglob, iglob);
        d.Vc(VX3, k, j, i) = image.arrays["Vc-VX3"](kglob, jglob, iglob);
      }
    }
  }
#else
  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real x1 = d.x[IDIR](i);
        real x2 = d.x[JDIR](j);
        real x3 = d.x[KDIR](k);

        real r = get_r(x1, x2, x3);
        real R = get_R(x1, x2, x3);
        real z = get_z(x1, x2, x3);
        real Rmin = FMAX(R, Rin);

        real temp = Wind::temperature(x1, x2, x3, epsilonGlob, epsilonTopGlob, Rin, HidealGlob, trSmoothingTempGlob);

        real H = epsilonGlob * Rmin;
        d.Vc(RHO, k, j, i) = pow(Rmin, -1.5) * exp(-(z * z) / (2 * H * H));

        d.Vc(VHOR, k, j, i) = ZERO_F;
        d.Vc(VVER, k, j, i) = ZERO_F;

        // d.Vc(VX3, k, j, i) = 1.0 / sqrt(Rmin) * sqrt(FMAX(Rmin / r - 2.5 * epsilonGlob * epsilonGlob, 0.0));
        // if (R < Rin) {
        //   d.Vc(VX3, k, j, i) = R * pow(r, -1.5);
        // }

        real vphi_edge = 1.0 / sqrt(Rin) * sqrt(FMAX(Rin / r - 2.5 * epsilonGlob * epsilonGlob, 0.0));

        if (R >= Rin) {
          d.Vc(VPHI, k, j, i) = 1.0 / sqrt(R) * sqrt(FMAX(R / r - 2.5 * epsilonGlob * epsilonGlob, 0.0));
        } else {
          d.Vc(VPHI, k, j, i) = (R / Rin) * vphi_edge; // Continuous solid body rotation
        }

        real densityFloor = Wind::computeDensityFloor(x1, x2, x3, densityFloorGlob, Rin, epsilonGlob);
        if (d.Vc(RHO, k, j, i) < densityFloor) {
          d.Vc(RHO, k, j, i) = densityFloor;
        }

#ifndef ISOTHERMAL
        d.Vc(PRS, k, j, i) = temp * d.Vc(RHO, k, j, i);
#endif
      }
    }
  }
#endif

#ifndef DISABLE_MHD
  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real x1 = d.x[IDIR](i);
        real x2 = d.x[JDIR](j);

        // real r = get_r(x1, x2);
        real R = get_R(x1, x2, x3);
        real z = get_z(x1, x2, x3);

        real m = -5.0 / 4.0;
        real B0 = epsilonGlob * sqrt(2.0 / betaGlob);

        A(HOR, k, j, i) = ZERO_F;
        A(VER, k, j, i) = ZERO_F;

        if (R > Rin) {
          A(PHIDIR, k, j, i) = B0 * (pow(Rin, m + 2.0) / R * (-1.0 / (m + 2.0)) + pow(R, m + 1.0) / (m + 2.0) + Rin * Rin / (2.0 * R));
        } else {
          A(PHIDIR, k, j, i) = B0 * R / 2.0;
        }
      }
    }
  }
  d.MakeVsFromAmag(A);

#endif

  // Send it all, if needed
  d.SyncToDevice();
}
