#include "setup.hpp"
#include "../../shared/wind/bc.hpp"
#include "../../shared/wind/floor.hpp"
#include "../../shared/wind/ionisation.hpp"
#include "../../shared/wind/params.hpp"
#include "dumpImage.hpp"
#include "idefix.hpp"

using namespace Params;

static std::string dat_path;

class MyGlobalClass {
public:
  // Class constructor
  MyGlobalClass(DataBlock &data) {
    // allocate some memory for the array the class contains
    this->array1 = IdefixArray3D<real>("MyAwesomeArray", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    // this->vpsi = IdefixHostArray1D<real>("vpsi", data.np_tot[IDIR]);
  }

  // array1, member of the class
  IdefixArray3D<real> array1;
  // IdefixHostArray1D<real> vpsi;
};

// A global class instance named "myGlobals"
MyGlobalClass *myGlobals;

// void Psi(DataBlock &data, IdefixHostArray1D<real> &psiIN) {
//   IdefixHostArray1D<real> psi = psiIN;
//   IdefixArray1D<real> x1 = data.x[IDIR];
//   IdefixArray1D<real> x2 = data.x[JDIR];
//   IdefixArray1D<real> dx1 = data.dx[IDIR];
//   IdefixArray1D<real> dx2 = data.dx[JDIR];
//   IdefixArray4D<real> Vc = data.hydro->Vc;

//   int jmid = data.np_tot[JDIR] / 2;
//   int jend = data.np_tot[JDIR];

//   real lhs = 0.;
//   idefix_reduce("Sum", 0, 0, 0, jend, 0, 0, KOKKOS_LAMBDA(int k, int j, int i, real &localSum) { localSum += pow(x1(i), 2) * sin(x2(j)) * Vc(BX1, k, j, i) * dx2(j); }, Kokkos::Sum<real>(lhs));

//   for (int ii = 0; ii < data.np_tot[IDIR]; ii++) {
//     real rhs = 0.;
//     idefix_reduce("Sum", 0, 0, jmid, jmid, 0, ii, KOKKOS_LAMBDA(int k, int j, int i, real &localSum) { localSum += x1(i) * Vc(BX2, k, j, i) * dx1(i); }, Kokkos::Sum<real>(rhs));
//     psiIN(ii) = lhs - rhs;
//   }
// }

void ComputeUserVars(DataBlock &data, UserDefVariablesContainer &variables) {

#ifndef DISABLE_MHD
  // Use Invdt as scratch array
  IdefixArray3D<real> scrh("Scratch", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  // IdefixArray3D<real> scrh_eta("Scratch_eta", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  // // Ask for a computation of xA ambipolar in this scratch array
  Wind::Resistivity(data, data.t, scrh);
  // Wind::Ambipolar(data, data.t, scrh);
#endif

  IdefixArray3D<real> array1 = myGlobals->array1;

  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  // Make references to the user-defined arrays (variables is a container of
  // IdefixHostArray3D) Note that the labels should match the variable names in
  // the input file
#ifndef DISABLE_MHD
  // IdefixHostArray3D<real> eta = variables["eta"];
  IdefixHostArray3D<real> Am = variables["Am"];

  // IdefixHostArray3D<real> EPhi = variables["Ephi"];
  // IdefixArray3D<real>::HostMirror scrhHost = Kokkos::create_mirror_view(scrh);
  // Kokkos::deep_copy(scrhHost, scrh);
  // IdefixArray3D<real>::HostMirror scrhHost_eta = Kokkos::create_mirror_view(scrh_eta);
  // Kokkos::deep_copy(scrhHost_eta, scrh_eta);
  IdefixArray3D<real>::host_mirror_type scrhHost = Kokkos::create_mirror_view(scrh);
#endif

  IdefixHostArray3D<real> InvDt = variables["InvDt"];
  // IdefixHostArray3D<real> addedMass = variables["addedMass"];
  // IdefixHostArray1D<real> vpsi = variables["vpsi"];

  // Vpsi(data, vpsi);

  IdefixHostArray1D<real> x1 = d.x[IDIR];
  IdefixHostArray1D<real> x2 = d.x[JDIR];
  IdefixHostArray4D<real> Vc = d.Vc;

  // IdefixArray3D<real>::HostMirror scrhHost_addedMass = Kokkos::create_mirror_view(array1);

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        real z = x1(i) * cos(x2(j));
        real R = FMAX(FABS(x1(i) * sin(x2(j))), ONE_F);
        real Omega = pow(R, -1.5);
#ifndef DISABLE_MHD
        // eta(k, j, i) = scrhHost_eta(k, j, i);
        // Am(k, j, i) = 1.0 / (Omega * scrhHost(k, j, i) * Vc(RHO, k, j, i));
        // EPhi(k, j, i) = d.Ex3(k, j, i);
        Am(k, j, i) = scrhHost(k, j, i);
#endif
        InvDt(k, j, i) = d.InvDt(k, j, i);
        // addedMass(k, j, i) = scrhHost_addedMass(k, j, i);
        // vpsi(k, j, i) = vpsi(k, j, i);
      }
    }
  }
}

void InternalBoundary(Hydro *hydro, const real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];

#ifndef DISABLE_MHD
  IdefixArray4D<real> Vs = hydro->Vs;

  real Va_ini_max = 50.0;
  real Va_fin_max = 10.0;
  real vAmax = Wind::computeVaMax(4.0, Va_ini_max, Va_fin_max, t);
#endif

  real densityFloor0 = densityFloorGlob;
  real Rin = 1.0;
  real epsilon = epsilonGlob;
  real epsilonTop = epsilonTopGlob;
  real Hideal = HidealGlob;
  real trSmoothingTemp = trSmoothingTempGlob;

  IdefixArray3D<real> array1 = myGlobals->array1;

  idefix_for(
      "InternalBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = x1(i) * sin(x2(j));
        real z = x1(i) * cos(x2(j));
    // real zh = FABS(z / R) / epsilon;

#ifndef DISABLE_MHD
        real b2 = EXPAND(Vc(BX1, k, j, i) * Vc(BX1, k, j, i), +Vc(BX2, k, j, i) * Vc(BX2, k, j, i), +Vc(BX3, k, j, i) * Vc(BX3, k, j, i));
        real va2 = b2 / Vc(RHO, k, j, i);
        real myMax = vAmax;
        // if(x1(i)<1.1) myMax=myMax/50.0;
        if (va2 > myMax * myMax) {
          Vc(RHO, k, j, i) = b2 / (myMax * myMax);
        }
#endif

        real densityFloor = Wind::computeDensityFloor(R, z, densityFloor0, Rin, epsilon);
        if (Vc(RHO, k, j, i) < densityFloor) {
          array1(k, j, i) = array1(k, j, i) + densityFloor - Vc(RHO, k, j, i);

          Vc(RHO, k, j, i) = densityFloor;
        }

#ifndef ISOTHERMAL
        real temp = Wind::temperature(x1(i), x2(j), epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);
        Vc(PRS, k, j, i) = temp * Vc(RHO, k, j, i);
#endif
      });
}
// Default constructor

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  IdefixArray1D<real> r = data.x[IDIR];
  IdefixArray1D<real> th = data.x[JDIR];
  real epsilon = epsilonGlob;
  real Rin = 1.0;
  idefix_for(
      "MySoundSpeed", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = r(i) * sin(th(j));
        real R0 = FMAX(R, Rin);
        cs(k, j, i) = epsilon / sqrt(R0);
      });
}

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&Wind::UserdefBoundary);
  data.hydro->EnrollInternalBoundary(&InternalBoundary);

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

  myGlobals = new MyGlobalClass(data);

  // alphaGlob = input.Get<real>("Setup", "alpha", 0);

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
#ifndef DISABLE_MHD
  IdefixHostArray4D<real> A = IdefixHostArray4D<real>("Setup_VectorPotential", 3, data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
#endif
  real Rin = 1.0;

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
        real r = d.x[IDIR](i);
        real th = d.x[JDIR](j);
        real R = r * sin(th);
        real z = r * cos(th);
        real Rmin = FMAX(R, Rin);

        real temp = Wind::temperature(r, th, epsilonGlob, epsilonTopGlob, Rin, HidealGlob, trSmoothingTempGlob);

        real H = epsilonGlob * Rmin;
        d.Vc(RHO, k, j, i) = pow(Rmin, -1.5) * exp(-(z * z) / (2 * H * H));

        d.Vc(VX1, k, j, i) = ZERO_F;
        d.Vc(VX2, k, j, i) = ZERO_F;

        d.Vc(VX3, k, j, i) = 1.0 / sqrt(Rmin) * sqrt(FMAX(Rmin / r - 2.5 * epsilonGlob * epsilonGlob, 0.0));
        if (R < Rin) {
          d.Vc(VX3, k, j, i) = R * pow(Rmin, -1.5);
        }
        // real only_disk = 0.5 * (1 - Wind::window(R, z, epsilonGlob, HidealGlob, trSmoothingGlob));
        // real exclude_disk = 1 - only_disk;
        // d.Vc(VX3, k, j, i) = d.Vc(VX3, k, j, i) * only_disk; // better ic?

        // // better ic?
        // real vz0 = 1.0;
        // real sign = z < 0.0 ? 1.0 : -1.0;
        // d.Vc(VX1, k, j, i) = cos(th) * exclude_disk * vz0 * sign;
        // d.Vc(VX2, k, j, i) = -sin(th) * exclude_disk * vz0 * sign;

        real densityFloor = Wind::computeDensityFloor(R, z, densityFloorGlob, Rin, epsilonGlob);
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
        real r = d.x[IDIR](i);
        real th = d.x[JDIR](j);
        real z = r * cos(th);
        real R = r * sin(th);

        real m = -5.0 / 4.0;
        real B0 = epsilonGlob * sqrt(2.0 / betaGlob);

        A(IDIR, k, j, i) = ZERO_F;
        A(JDIR, k, j, i) = ZERO_F;

        if (R > Rin) {
          A(KDIR, k, j, i) = B0 * (pow(Rin, m + 2.0) / R * (-1.0 / (m + 2.0)) + pow(R, m + 1.0) / (m + 2.0) + Rin * Rin / (2.0 * R));
        } else {
          A(KDIR, k, j, i) = B0 * R / 2.0;
        }
      }
    }
  }
#endif

// Make the field from the vector potential
#ifndef DISABLE_MHD
  d.MakeVsFromAmag(A);
#endif

  // Send it all, if needed
  d.SyncToDevice();
}
