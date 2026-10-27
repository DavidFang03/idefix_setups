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
#include "../../shared/wind/ic.hpp"
#include "../../shared/wind/ionisation.hpp"
#include "../../shared/wind/params.hpp"
#include "dumpImage.hpp"

#include "idefix.hpp"

#ifdef RELOAD
std::string reload_path;
#endif

using namespace Params;

// void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
//   IdefixArray1D<real> x1 = data.x[IDIR];
//   IdefixArray1D<real> x2 = data.x[JDIR];
//   IdefixArray1D<real> x3 = data.x[KDIR];
//   real epsilon = epsilonGlob;
//   real epsilonTop = epsilonTopGlob;
//   real Hideal = HidealGlob;
//   real trSmoothingTemp = trSmoothingTempGlob;
//   real Rin = RinGlob;
//   idefix_for(
//       "MySoundSpeed", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
//         real temp = Wind::temperature(x1(i), x2(j), x3(k), epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);
//         cs(k, j, i) = sqrt(temp);
//       });
// }

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  IdefixArray1D<real> r = data.x[IDIR];
  IdefixArray1D<real> th = data.x[JDIR];
  real epsilon = epsilonGlob;
  idefix_for(
      "MySoundSpeed", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = r(i) * sin(th(j));
        real Rmin = FMAX(R, 0.1);
        cs(k, j, i) = epsilon / sqrt(Rmin);
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
#ifdef ISOTHERMAL
  IdefixArray3D<real> scrh_cs("Scratch_cs", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  MySoundSpeed(data, data.t, scrh_cs);
  IdefixHostArray3D<real> cs = variables["cs"];
  IdefixArray3D<real>::host_mirror_type scrhHost_cs = Kokkos::create_mirror_view(scrh_cs);
  Kokkos::deep_copy(scrhHost_cs, scrh_cs);
#endif

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        real z = get_z(x1(i), x2(j), x3(k));
        real R = get_R(x1(i), x2(j), x3(k));
        real R0 = FMAX(R, RinGlob);
        real Omega = pow(R0, -1.5);
#ifndef DISABLE_MHD
        Am(k, j, i) = 1.0 / (Omega * scrhHost(k, j, i) * Vc(RHO, k, j, i));
        Ephi(k, j, i) = d.Ex3(k, j, i);
#endif
        InvDt(k, j, i) = d.InvDt(k, j, i);
#ifdef ISOTHERMAL
        cs(k, j, i) = scrhHost_cs(k, j, i);
#endif
      }
    }
  }
}

void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  real epsilon = epsilonGlob;
  if (dir == IDIR) {

    int ighost;
    if (side == left) {
      ighost = data->beg[IDIR];
    } else if (side == right) {
      ighost = data->end[IDIR] - 1;
    }

    hydro->boundary->BoundaryFor(
        "UserDefBoundary", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          real R = x1(i) * sin(x2(j));
          real z = x1(i) * cos(x2(j));
          real Vk = 1.0 / sqrt(R);
          real cs2 = (epsilon * Vk) * (epsilon * Vk);

          Vc(RHO, k, j, i) = 1.0 / (R * sqrt(R)) * exp(1.0 / (cs2) * (1 / x1(i) - 1 / R));
          Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
          Vc(VX3, k, j, i) = Vk * sqrt(R / x1(i) - 2.5 * epsilon * epsilon);
        });
  }

  if (dir == JDIR) {
    int jghost;
    if (side == left) {
      jghost = data->beg[JDIR];
    } else if (side == right) {
      jghost = data->end[JDIR] - 1;
    }

    hydro->boundary->BoundaryFor(
        "UserDefBoundary", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          real R = x1(i) * sin(x2(j));
          real z = x1(i) * cos(x2(j));
          real Vk = 1.0 / sqrt(R);
          real cs2 = (epsilon * Vk) * (epsilon * Vk);

          Vc(RHO, k, j, i) = 1.0 / (R * sqrt(R)) * exp(1.0 / (cs2) * (1 / x1(i) - 1 / R));
          Vc(VX1, k, j, i) = Vc(VX1, k, jghost, i);
          Vc(VX2, k, j, i) = ZERO_F;
          Vc(VX3, k, j, i) = Vk * sqrt(R / x1(i) - 2.5 * epsilon * epsilon);
        });
  }
}

// Default constructor

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  // data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
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

  RinGlob = input.Get<real>("Setup", "Rin", 0);

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

  real Rin = RinGlob;

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

        // real r = get_r(x1, x2, x3);
        // real R = get_R(x1, x2, x3);
        // real z = get_z(x1, x2, x3);
        // real Rmin = FMAX(R, Rin);
        // real Vk = 1.0 / sqrt(R);
        // real cs2 = (epsilonGlob * Vk) * (epsilonGlob * Vk);
        // d.Vc(RHO, k, j, i) = 1.0 / (R * sqrt(R)) * exp(1.0 / (cs2) * (1 / r - 1 / R));

        d.Vc(RHO, k, j, i) = Wind::density_initial(x1, x2, x3, epsilonGlob, Rin, densityFloorGlob);
        // d.Vc(VX3, k, j, i) = Vk * sqrt(R / r - 2.5 * epsilonGlob * epsilonGlob);

        d.Vc(VHOR, k, j, i) = ZERO_F;
        d.Vc(VVER, k, j, i) = ZERO_F;
        d.Vc(VPHI, k, j, i) = Wind::vphi_initial(x1, x2, x3, epsilonGlob, Rin);

#ifndef ISOTHERMAL
        real temp = Wind::temperature(x1, x2, x3, epsilonGlob, epsilonTopGlob, Rin, HidealGlob, trSmoothingTempGlob);
        d.Vc(PRS, k, j, i) = temp * d.Vc(RHO, k, j, i);
#endif
      }
    }
  }
#endif

#ifndef DISABLE_MHD
  // Make vector potential
  IdefixHostArray4D<real> A = IdefixHostArray4D<real>("Setup_VectorPotential", 3, data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  real m = -5.0 / 4.0;
  real B0 = epsilonGlob * sqrt(2.0 / betaGlob);

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
          A(PHIDIR, k, j, i) = B0 * (pow(Rin, m + 2.0) / R * (-1.0 / (m + 2.0)) + pow(R, m + 1.0) / (m + 2.0) + Rin * Rin / (2.0 * R)); // check Rin
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
