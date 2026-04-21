#include "setup.hpp"
#include "idefix.hpp"

real sigmaSlopeGlob;
real csSlopeGlob;
real h0Glob;

real dtg;
real PM;

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

void UserdefBoundaryParticles(DataBlock &data, real t, int dir, BoundarySide side) {
  idfx::pushRegion("UserdefBoundaryParticles");
  auto states = data.particles->pack->states;
  IdefixArray1D<bool> isActive = data.particles->pack->isActive;
  int capacity = data.particles->pack->capacity;

  IdefixHostArray1D<int> host_counter = IdefixHostArray1D<int>("counter", 1);
  IdefixArray1D<int> device_counter = IdefixArray1D<int>("counter", 1);

  IdefixHostArray1D<real> host_mass_counter = IdefixHostArray1D<real>("mass_counter", 1);
  IdefixArray1D<real> device_mass_counter = IdefixArray1D<real>("counter", 1);

  if ((dir == IDIR) && (side == left)) {
    const real xl = data.mygrid->xbeg[IDIR];
    idefix_for(
        "UserdefParticlesBoundary", 0, capacity, KOKKOS_LAMBDA(int idx) {
          real x = states(dir, idx);
          if (isActive(idx) && x < xl) {
            isActive(idx) = false;
            Kokkos::atomic_add(&device_counter(0), 1);
            Kokkos::atomic_add(&device_mass_counter(0), states(PMASS, idx));
          }
        });
    Kokkos::deep_copy(host_counter, device_counter);

    data.particles->pack->activeCount -= host_counter(0);

    Kokkos::deep_copy(host_mass_counter, device_mass_counter);
    data.gravity->centralMass += 4 * M_PI * host_mass_counter(0);
  }

  if ((dir == IDIR) && (side == right)) {
    const real xr = data.mygrid->xend[IDIR];
    idefix_for(
        "UserdefParticlesBoundary", 0, capacity, KOKKOS_LAMBDA(int idx) {
          real x = states(dir, idx);
          if (isActive(idx) && x > xr) {
            isActive(idx) = false;
            Kokkos::atomic_add(&device_counter(0), 1);
          }
        });
    Kokkos::deep_copy(host_counter, device_counter);
    data.particles->pack->activeCount -= host_counter(0);
  }

  if (data.particles->pack->activeCount < 0)
    data.particles->pack->activeCount = 0;

  idfx::popRegion();
}

// Default constructor

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  // data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  // data.hydro->EnrollUserSourceTerm(&MySourceTerm);
  // data.hydro->EnrollInternalBoundary(&InternalBoundary);
  // data.hydro->EnrollFluxBoundary(&FluxBoundary);
  // output.EnrollUserDefVariables(&ComputeUserVars);
  h0Glob = input.Get<real>("Setup", "h0", 0);
  sigmaSlopeGlob = input.Get<real>("Setup", "sigmaSlope", 0);
  csSlopeGlob = input.Get<real>("Setup", "csSlope", 0);

  PM = input.Get<real>("Particles", "ParticleMass", 0);

  dtg = input.Get<real>("Particles", "DustToGas", 0);

  data.particles->EnrollUserDefBoundary(&UserdefBoundaryParticles);
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
  // real sigma0 = sigma0Glob;
  real sigma0 = PM / dtg * d.PactiveCount;
  real csSlope = csSlopeGlob;

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

        // real theta0dust = 3.1415 / 3;

        // d.Vc(RHO, k, j, i) = sigma0 * pow(R, -sigmaSlope) * exp(R * R / h2 * (1.0 / sqrt(1.0 + z * z / R / R) - 1.0));
        d.Vc(RHO, k, j, i) = sigma0 * pow(R, -sigmaSlope) * exp(z * z / (2 * h2 * R * R));
        d.Vc(VX1, k, j, i) = 0.0; //+sin(8*d.x[JDIR](j));
        d.Vc(VX2, k, j, i) = 0.0;
        d.Vc(VX3, k, j, i) = Vk * (1 + 0.5 * (h2 / r / r) * (-sigmaSlope - 2 * csSlope - csSlope * z * z / 2.0 / h2));
      }
    }
  };

  for (int n = 0; n < d.PactiveCount; n++) {
    // d.dustVc[n](RHO, k, j, i) = (1e-5 + 3e-3 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.1 / 0.1)) * d.Vc(RHO, k, j, i); //?
    real r0dust = 10;
    d.Ps(PX1, n) = r0dust;
    d.Ps(PX2, n) = n * M_PI / d.PactiveCount;
    d.Ps(PVX1, n) = 0.0;
    real Vk0 = 1 / sqrt(r0dust);
    d.Ps(PVX2, n) = 0.0;
    d.Ps(PVX3, n) = Vk0;
    d.Ps(PMASS, n) = PM;
  }

  // Send it all, if needed
  d.SyncToDevice();
}
