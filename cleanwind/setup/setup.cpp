#include "setup.hpp"
#include "../../shared/wind/bc.hpp"
#include "../../shared/wind/floor.hpp"
#include "../../shared/wind/ionisation.hpp"
#include "../../shared/wind/params.hpp"
#include "analysis.hpp"
#include "dumpImage.hpp"
#include "idefix.hpp"

using namespace Params;

static std::string dat_path;
Analysis *analysis;

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

  // Use Invdt as scratch array
  IdefixArray3D<real> scrh("Scratch", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  IdefixArray3D<real> scrh_eta("Scratch_eta", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

  IdefixArray3D<real> array1 = myGlobals->array1;

  // Ask for a computation of xA ambipolar in this scratch array
  Wind::Resistivity(data, data.t, scrh_eta);
  Wind::Ambipolar(data, data.t, scrh);

  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  // Make references to the user-defined arrays (variables is a container of
  // IdefixHostArray3D) Note that the labels should match the variable names in
  // the input file
  IdefixHostArray3D<real> eta = variables["eta"];
  IdefixHostArray3D<real> Am = variables["Am"];
  IdefixHostArray3D<real> InvDt = variables["InvDt"];
  IdefixHostArray3D<real> EPhi = variables["Ephi"];
  IdefixHostArray3D<real> addedMass = variables["addedMass"];
  // IdefixHostArray1D<real> vpsi = variables["vpsi"];

  // Vpsi(data, vpsi);

  IdefixHostArray1D<real> x1 = d.x[IDIR];
  IdefixHostArray1D<real> x2 = d.x[JDIR];
  IdefixHostArray4D<real> Vc = d.Vc;
  IdefixArray3D<real>::HostMirror scrhHost = Kokkos::create_mirror_view(scrh);
  Kokkos::deep_copy(scrhHost, scrh);
  IdefixArray3D<real>::HostMirror scrhHost_eta = Kokkos::create_mirror_view(scrh_eta);
  Kokkos::deep_copy(scrhHost_eta, scrh_eta);
  IdefixArray3D<real>::HostMirror scrhHost_addedMass = Kokkos::create_mirror_view(array1);

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        real z = x1(i) * cos(x2(j));
        real R = FMAX(FABS(x1(i) * sin(x2(j))), ONE_F);
        real Omega = pow(R, -1.5);
        eta(k, j, i) = scrhHost_eta(k, j, i);
        Am(k, j, i) = 1.0 / (Omega * scrhHost(k, j, i) * Vc(RHO, k, j, i));
        InvDt(k, j, i) = d.InvDt(k, j, i);
        EPhi(k, j, i) = d.Ex3(k, j, i);
        addedMass(k, j, i) = scrhHost_addedMass(k, j, i);
        // vpsi(k, j, i) = vpsi(k, j, i);
      }
    }
  }
}

void AnalysisFunction(DataBlock &data) { analysis->PerformAnalysis(data); }

void InternalBoundary(Hydro *hydro, const real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Vs = hydro->Vs;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];

  real vAmax = Wind::computeVaMax(4.0, 50.0, 8.0, t);
  real densityFloor0 = densityFloorGlob;
  real Rin = 1.0;
  real epsilon = epsilonGlob;

  IdefixArray3D<real> array1 = myGlobals->array1;

  idefix_for(
      "InternalBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = x1(i) * sin(x2(j));
        real z = x1(i) * cos(x2(j));
        // real zh = FABS(z / R) / epsilon;

        real b2 = EXPAND(Vc(BX1, k, j, i) * Vc(BX1, k, j, i), +Vc(BX2, k, j, i) * Vc(BX2, k, j, i), +Vc(BX3, k, j, i) * Vc(BX3, k, j, i));
        real va2 = b2 / Vc(RHO, k, j, i);
        real myMax = vAmax;
        // if(x1(i)<1.1) myMax=myMax/50.0;
        if (va2 > myMax * myMax) {
          real T = Vc(PRS, k, j, i) / Vc(RHO, k, j, i);
          Vc(RHO, k, j, i) = b2 / (myMax * myMax);
          Vc(PRS, k, j, i) = T * Vc(RHO, k, j, i);
        }
        real densityFloor = Wind::computeDensityFloor(R, z, densityFloor0, Rin, epsilon);
        if (Vc(RHO, k, j, i) < densityFloor) {
          array1(k, j, i) = array1(k, j, i) + densityFloor - Vc(RHO, k, j, i);

          real T = Vc(PRS, k, j, i) / Vc(RHO, k, j, i);
          Vc(RHO, k, j, i) = densityFloor;
        }
      });
}
// Default constructor

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&Wind::UserdefBoundary);
  data.hydro->EnrollAmbipolarDiffusivity(&Wind::Ambipolar);
  data.hydro->EnrollOhmicDiffusivity(&Wind::Resistivity);
  data.hydro->EnrollUserSourceTerm(&Wind::MySourceTerm);
  data.hydro->EnrollInternalBoundary(&InternalBoundary);
  data.hydro->EnrollEmfBoundary(&Wind::EmfBoundary);

  output.EnrollUserDefVariables(&ComputeUserVars);

  myGlobals = new MyGlobalClass(data);

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

  dat_path = input.Get<std::string>("Output", "dat_path", 0);

  analysis = new Analysis(input, grid, data, output, dat_path);
  output.EnrollAnalysis(&AnalysisFunction);
  // Reset analysis if required
  if (!input.restartRequested) {
    analysis->ResetAnalysis();
  }
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

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real r = d.x[IDIR](i);
        real th = d.x[JDIR](j);
        real z = r * cos(th);
        real R = r * sin(th);
        if (R > Rin) {
          real Zh = FABS(z / R) / epsilonGlob;
          real csdisk = epsilonGlob / sqrt(R);
          real cs2 = csdisk * csdisk;
          d.Vc(RHO, k, j, i) = 1.0 / (R * sqrt(R)) * exp(1.0 / (csdisk * csdisk) * (1.0 / sqrt(R * R + z * z) - 1.0 / R));
          d.Vc(VX3, k, j, i) = 1.0 / sqrt(R) * sqrt(FMAX(R / sqrt(R * R + z * z) - 2.5 * csdisk * csdisk, 0.0));
          d.Vc(PRS, k, j, i) = cs2 * d.Vc(RHO, k, j, i);
          if (std::isnan(d.Vc(VX3, k, j, i))) {
            idfx::cout << "Nan in R>Rin at (i,j,k)=(" << i << "," << j << "," << k << "), (r,th,R,z)=(" << r << "," << th << "," << R << "," << z << ")" << std::endl;
            IDEFIX_ERROR("Nan!s");
          }
        } else {
          real Zh = FABS(z / Rin) / epsilonGlob;
          real csdisk = epsilonGlob / sqrt(Rin);
          real cs2 = csdisk * csdisk;
          d.Vc(RHO, k, j, i) = 1.0 / (Rin * sqrt(Rin)) * exp(1.0 / (csdisk * csdisk) * (1.0 / sqrt(Rin * Rin + z * z) - 1.0 / Rin));
          d.Vc(VX3, k, j, i) = 1.0 / sqrt(Rin) * sqrt(FMAX(Rin / sqrt(Rin * Rin + z * z) - 2.5 * csdisk * csdisk, 0.0));
          d.Vc(PRS, k, j, i) = cs2 * d.Vc(RHO, k, j, i);
          if (std::isnan(d.Vc(VX3, k, j, i))) {
            idfx::cout << "Nan in R<Rin at (i,j,k)=(" << i << "," << j << "," << k << "), (r,th,R,z)=(" << r << "," << th << "," << R << "," << z << ")" << std::endl;
            IDEFIX_ERROR("Nan!s");
          }
        }

        d.Vc(VX1, k, j, i) = ZERO_F;
        d.Vc(VX2, k, j, i) = ZERO_F;

        real densityFloor = Wind::computeDensityFloor(R, z, densityFloorGlob, Rin, epsilonGlob);
        if (d.Vc(RHO, k, j, i) < densityFloor) {
          d.Vc(RHO, k, j, i) = densityFloor;
          // d.Vc(PRS,k,j,i) = T2*d.Vc(RHO,k,j,i);
        }

        // Vector potential on the corner
        real s = sin(d.xl[JDIR](j));
        R = d.xl[IDIR](i) * s;

        A(IDIR, k, j, i) = ZERO_F;
        A(JDIR, k, j, i) = ZERO_F;

#ifdef EVOLVE_VECTOR_POTENTIAL
        if (R > Rin) {
          d.Ve(AX3e, k, j, i) = B0 * (pow(Rin, m + 2.0) / R * (-1.0 / (m + 2.0)) + pow(R, m + 1.0) / (m + 2.0) + Rin * Rin / (2.0 * R));
        } else {
          d.Ve(AX3e, k, j, i) = B0 * R / 2.0;
        }
#else
        if (R > Rin) {
          A(KDIR, k, j, i) = B0 * (pow(Rin, m + 2.0) / R * (-1.0 / (m + 2.0)) + pow(R, m + 1.0) / (m + 2.0));
          A(KDIR, k, j, i) = B0 * (pow(Rin, m + 2.0) / R * (-1.0 / (m + 2.0)) + pow(R, m + 1.0) / (m + 2.0) + Rin * Rin / (2.0 * R));
        } else {
          A(KDIR, k, j, i) = B0 * R / 2.0;
        }
#endif
      }
    }
  }

// Make the field from the vector potential
#ifndef EVOLVE_VECTOR_POTENTIAL
  d.MakeVsFromAmag(A);
#endif

  // Send it all, if needed
  d.SyncToDevice();
}
