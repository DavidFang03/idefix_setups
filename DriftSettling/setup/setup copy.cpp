#include "setup.hpp"
#include "idefix.hpp"

real sigma0Glob;
real sigmaSlopeGlob;
real csSlopeGlob;
real h0Glob;

real dtg;

// void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
//   IdefixArray4D<real> Vc = hydro->Vc;
//   auto *data = hydro->data;
//   IdefixArray1D<real> x1 = data->x[IDIR];
//   if (dir == IDIR) {
//     int ighost, ibeg, iend;
//     if (side == left) {
//       ighost = data->beg[IDIR];
//       ibeg = 0;
//       iend = data->beg[IDIR];
//       idefix_for(
//           "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
//             real R = x1(i);
//             real Vk = 1.0 / sqrt(R);

//             // ??????
//             Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
//             Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i + 1);
//             Vc(VX2, k, j, i) = Vk;
//             Vc(VX3, k, j, i) = Vc(VX3, k, j, 2 * ighost - i + 1);
//           });
//     } else if (side == right) {
//       ighost = data->end[IDIR] - 1;
//       ibeg = data->end[IDIR];
//       iend = data->np_tot[IDIR];
//       idefix_for(
//           "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
//             real R = x1(i);
//             real Vk = 1.0 / sqrt(R);

//             // Null gradient
//             Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
//             Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
//             Vc(VX2, k, j, i) = Vk;
//             Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
//           });
//     }
//   }
// }

void BoundaryDustNullGradient(Fluid<DustPhysics> *dust, int dir, BoundarySide side, real t) {
  auto *data = dust->data;
  IdefixArray4D<real> Vc = dust->Vc;
  // IdefixArray4D<real> Vs = hydro->Vs;
  if (dir == IDIR) {
    int iref, ibeg, iend;
    if (side == left) {
      iref = data->beg[IDIR];
      ibeg = 0;
      iend = data->beg[IDIR];
    } else {
      iref = data->end[IDIR] - 1;
      ibeg = data->end[IDIR];
      iend = data->np_tot[IDIR];
    }
    int nvar = Vc.extent(0);
    idefix_for(
        "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          for (int n = 0; n < nvar; n++) {
            Vc(n, k, j, i) = Vc(n, k, j, iref);
          }
        });
  }
}

void UserdefBoundaryDust(Fluid<DustPhysics> *dust, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = dust->Vc;
  auto data = dust->data;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];
  if (dir == IDIR) {
    int ighost, ibeg, iend;
    if (side == left) {
      ighost = data->beg[IDIR];
      ibeg = 0;
      iend = data->beg[IDIR];
      idefix_for(
          "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
            real R = x1(i);
            real Vk = 1.0 / sqrt(R);
            Vc(RHO, k, j, i) = Vc(RHO, k, j, 2 * ighost - i + 1);
            Vc(VX3, k, j, i) = Vk;
            Vc(VX2, k, j, i) = Vc(VX2, k, j, 2 * ighost - i + 1);

            if (Vc(VX1, k, j, i) >= ZERO_F) { // ?????????? No inflow?
              Vc(VX1, k, j, ighost) = 0.0;
            } else {
              Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            }
          });
    } else if (side == right) {
      ighost = data->end[IDIR] - 1;
      ibeg = data->end[IDIR];
      iend = data->np_tot[IDIR];
      idefix_for(
          "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
            real R = x1(i);
            real Vk = 1.0 / sqrt(R);
            Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
            Vc(VX2, k, j, i) = Vk;
            Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);

            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            if (Vc(VX1, k, j, ighost) <= ZERO_F) { // ? No inflow?
              Vc(VX1, k, j, i) = 0.0;
            }
          });
    }
  }
}

// void ComputeUserVars(DataBlock &data, UserDefVariablesContainer &variables) {
//   // Mirror data on Host
//   DataBlockHost d(data);

//   // Sync it
//   d.SyncFromDevice();

//   // Make references to the user-defined arrays (variables is a container of
//   // IdefixHostArray3D) Note that the labels should match the variable names in
//   // the input file
//   IdefixHostArray3D<real> InvDt = variables["InvDt"];

//   for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
//     for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
//       for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
//         InvDt(k, j, i) = d.InvDt(k, j, i);
//       }
//     }
//   }
// }

// Default constructor

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  // data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  // if (data.haveDust) {
  //   int nSpecies = data.dust.size();
  //   for (int n = 0; n < nSpecies; n++) {
  //     data.dust[n]->EnrollUserDefBoundary(&UserdefBoundaryDust);
  //     // data.dust[n]->drag->EnrollUserDrag(&MyDrag);
  //   }
  // }
  // data.hydro->EnrollUserSourceTerm(&MySourceTerm);
  // data.hydro->EnrollInternalBoundary(&InternalBoundary);
  // data.hydro->EnrollFluxBoundary(&FluxBoundary);
  // output.EnrollUserDefVariables(&ComputeUserVars);
  h0Glob = input.Get<real>("Setup", "h0", 0);
  sigmaSlopeGlob = input.Get<real>("Setup", "sigmaSlope", 0);
  sigma0Glob = input.Get<real>("Setup", "sigma0", 0);
  csSlopeGlob = input.Get<real>("Setup", "csSlope", 0);

  dtg = input.Get<real>("Dust", "DustToGas", 0);
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
  // Create a host copy
  DataBlockHost d(data);

  real h0 = h0Glob;
  real sigmaSlope = sigmaSlopeGlob;
  real sigma0 = sigma0Glob;
  real csSlope = csSlopeGlob;

  real dustDensityFloor = 1e-6;

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real R = d.x[IDIR](i);
        real theta = d.x[JDIR](j);
        real r = R * sin(theta);
        real z = R * cos(theta);
        real Vk = 1.0 / sqrt(R);
        real cs2 = h0 * h0 * pow(R, -csSlope);
        real Omega = pow(R, -1.5);
        real h2 = cs2 / Omega / Omega;

        real r0dust = 10;
        real theta0dust = 3.1415 / 3;

        // d.Vc(RHO, k, j, i) = sigma0 * pow(R, -sigmaSlope) * exp(R * R / h2 * (1.0 / sqrt(1.0 + z * z / R / R) - 1.0));
        d.Vc(RHO, k, j, i) = sigma0 * pow(R, -sigmaSlope) * exp(z * z / (2 * h2 * R * R));
        d.Vc(VX1, k, j, i) = 0.0; //+sin(8*d.x[JDIR](j));
        d.Vc(VX2, k, j, i) = 0.0;
        d.Vc(VX3, k, j, i) = Vk * (1 + 0.5 * (h2 / r / r) * (-sigmaSlope - 2 * csSlope - csSlope * z * z / 2.0 / h2));

        for (int n = 0; n < data.dust.size(); n++) {
          // d.dustVc[n](RHO, k, j, i) = (1e-5 + 3e-3 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.1 / 0.1)) * d.Vc(RHO, k, j, i); //?
          d.dustVc[n](RHO, k, j, i) = dustDensityFloor + d.Vc(RHO, k, j, i) * dtg * exp(-(r - r0dust) * (r - r0dust) / (1e-4)) * exp(-(theta - theta0dust) * (theta - theta0dust) / (1e-4));
          d.dustVc[n](VX1, k, j, i) = 0.0;
          d.dustVc[n](VX2, k, j, i) = 0.0;
          d.dustVc[n](VX3, k, j, i) = Vk;
        }
      }
    }
  }

  // Send it all, if needed
  d.SyncToDevice();
}
