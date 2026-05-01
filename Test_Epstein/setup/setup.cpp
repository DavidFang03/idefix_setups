#include "setup.hpp"
// #include "dumpImage.hpp"
#include "idefix.hpp"

real dtg;
real PM;

void UserdefStoppingTime(DataBlock &data, const real t, IdefixArray1D<real> &tstop) {
  // a separate hook is needed for particles because gravity isn't in general
  // computed at the same time for particles and the fluid.

  // GPUS cannot capture static variables
  // auto states = data.particles->pack->states;

  // Assuming logarithmic spacing
  // i = [log(Ri) - log(R0)]/log(1+dR/R)

  auto states = data.particles->pack->states;
  auto isActive = data.particles->pack->isActive;
  // DataBlockHost d(data);
  auto tstop_array = data.particles->pack->fields.GetField<real>("t_stop");
  // auto r_indexes = data.particles->pack->fields.GetField<real>("r_indexes");
  // auto theta_indexes =
  // data.particles->pack->fields.GetField<real>("theta_indexes"); auto
  // phi_indexes = data.particles->pack->fields.GetField<real>("phi_indexes");

  // Below works only for uniform theta, phi and logarithmic r. It finds the
  // cell where the particle is in.
  // TODO: Interpolation?
  DataBlockHost d(data);
  // d.SyncFromDevice();

  // IdefixHostArray1D<real> x1 = d.x[IDIR];
  // IdefixHostArray1D<real> x2 = d.x[JDIR];
  // IdefixHostArray1D<real> x3 = d.x[KDIR];
  // IdefixHostArray4D<real> Vc = d.Vc;
  IdefixArray4D<real> Vc = data.hydro->Vc;

  int i_gbeg = d.gbeg[IDIR];
  int j_gbeg = d.gbeg[JDIR];
  int ibeg = d.beg[IDIR];
  int jbeg = d.beg[JDIR];
  int iend = d.end[IDIR];
  int kbeg = d.beg[KDIR];

  real iint = d.np_int[IDIR];
  real rstart = d.x[IDIR](ibeg);
  real rend = d.x[IDIR](iend);

  real theta_beg = d.x[JDIR](jbeg);
  real dtheta = d.x[JDIR](jbeg + 1) - d.x[JDIR](jbeg);
  real phi_beg = d.x[KDIR](kbeg);
  real dphi = d.x[KDIR](kbeg + 1) - d.x[KDIR](kbeg);

  // IdefixHostArray1D<real> host_tstop = IdefixHostArray1D<real>("host_tstop", 1);

  // for (int n = 0; n < d.PactiveCount; n++) {
  //   tstop(n) = 1;
  // }
  idefix_for(
      "StoppingTime", 0, data.particles->pack->maxActiveIndex + 1, KOKKOS_LAMBDA(int idx) {
        if (isActive(idx)) {

          real r = states(PX1, idx);
          real theta = states(PX2, idx);
          real phi = states(PX3, idx);

          int i = floor(iint * (log(r / rstart)) / log(rend / rstart)) + ibeg;
          int j = floor((theta - theta_beg) / dtheta) + jbeg;

          int k = floor((phi - phi_beg) / dphi) + kbeg;
          //         int k = 0;

          //         real rho = Vc(RHO, k, j, i);
          // real rho = Vc(RHO, k, j, i);
          tstop_array(idx) = Vc(RHO, kbeg, j, i);

          tstop(idx) = 1.0;
          // tstop(idx) = Vc(RHO, k, j, i);
        }
      });
}

// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  auto *data = hydro->data;
  if ((dir == IDIR) && (side == left)) {
    IdefixArray4D<real> Vc = hydro->Vc;

    int ighost = data->nghost[IDIR];
    hydro->boundary->BoundaryFor(
        "UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);

          if (Vc(VX1, k, j, ighost) >= ZERO_F) {
            Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i);
          } else {
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          }

          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
          Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
        });
  }

  if ((dir == IDIR) && (side == right)) {
    IdefixArray4D<real> Vc = hydro->Vc;

    int ighost = data->end[IDIR] - 1;

    hydro->boundary->BoundaryFor(
        "UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);

          if (Vc(VX1, k, j, ighost) <= ZERO_F)
            Vc(VX1, k, j, i) = 0.0;
          else
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);

          Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
        });
  }
  if (dir == JDIR) {
    IdefixArray4D<real> Vc = hydro->Vc;

    const int j_beg = data->beg[JDIR];
    const int j_end = data->end[JDIR];

    if (side == left) {
      // Cell-centered Loop
      idefix_for(
          "UserDefX2_Left_Vc", 0, data->np_tot[KDIR], 0, j_beg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            // Reflection across the interface j_beg - 0.5
            const int jrefl = 2 * j_beg - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
          });

      // Face-centered B field (BX2s)
      // For side == left, the loop covers j from 0 to j_beg (inclusive of the
      // boundary face)
    } else if (side == right) {
      // Cell-centered Loop
      idefix_for(
          "UserDefX2_Right_Vc", 0, data->np_tot[KDIR], j_end, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            // Reflection across the interface j_end - 0.5
            const int jrefl = 2 * j_end - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
          });
    }
  }
}

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

  if ((dir == JDIR) && (side == left)) {
    const real thetal = data.mygrid->xbeg[JDIR];
    idefix_for(
        "UserdefParticlesBoundary", 0, capacity, KOKKOS_LAMBDA(int idx) {
          real x = states(dir, idx);
          if (isActive(idx) && x < thetal) {
            isActive(idx) = false;
            Kokkos::atomic_add(&device_counter(0), 1);
          }
        });
    Kokkos::deep_copy(host_counter, device_counter);

    data.particles->pack->activeCount -= host_counter(0);
  }

  if ((dir == JDIR) && (side == right)) {
    const real thetar = data.mygrid->xend[JDIR];
    idefix_for(
        "UserdefParticlesBoundary", 0, capacity, KOKKOS_LAMBDA(int idx) {
          real x = states(dir, idx);
          if (isActive(idx) && x > thetar) {
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
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.particles->EnrollUserDefBoundary(&UserdefBoundaryParticles);
  data.particles->EnrollStoppingTime(&UserdefStoppingTime);

  PM = input.Get<real>("Particles", "ParticleMass", 0);
  dtg = input.Get<real>("Particles", "DustToGas", 0);
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
  // Create a host copy
  DataBlockHost d(data);

  int np_tot = data.np_tot[JDIR];
  int np_int = data.np_int[JDIR];
  printf("np_tot: %d\n", np_tot);
  printf("np_int: %d\n", np_int);
  printf("theta0: %f\n", d.x[JDIR](0));
  printf("thetabeg: %f\n", d.x[JDIR](d.beg[JDIR]));
  printf("thetagbeg: %f\n", d.x[JDIR](d.gbeg[JDIR]));

  real thetabeg = d.x[JDIR](d.beg[JDIR]);
  real thetabegp1 = d.x[JDIR](d.beg[JDIR] + 1);
  printf("dtheta1: %f\n", M_PI / np_int);         // 0.012272
  printf("dtheta2: %f\n", thetabegp1 - thetabeg); // 0.012272

  auto tstop_array = d.Pfields.GetField<real>("t_stop");

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real R = d.x[IDIR](i);
        real theta = d.x[JDIR](j);
        // d.Vc(RHO, k, j, i) = 0.2 * (1 + sin(theta * 20));
        d.Vc(RHO, k, j, i) = 30 - R;
        // d.Vc(RHO, k, j, i) = 0.2 * (1 + sin(R * 20));
        d.Vc(VX1, k, j, i) = 0.0;
        d.Vc(VX2, k, j, i) = 0.0;
        d.Vc(VX3, k, j, i) = 1 / sqrt(R);
      }
    }
  };

  real ntot = d.PactiveCount;
  real bunch = 5;

  for (int i = 0; i < ntot; i += bunch) {
    for (int j = 0; j < bunch && (i + j) < ntot; j++) {
      int n = i + j;
      real r = 5.0 + 10 * n / ntot;
      // real r = 5.0;
      // real theta = acos(z / r);
      // real theta = d.x[JDIR](1);
      real theta = 3.14 * (j + 0.5) / bunch;

      d.Ps(PX1, n) = r;
      d.Ps(PX2, n) = theta;
      d.Ps(PX3, n) = 0.0;
      d.Ps(PVX1, n) = 0.0;
      d.Ps(PVX2, n) = 0.0;
      d.Ps(PVX3, n) = 1 / sqrt(r);
      d.Ps(PMASS, n) = PM;
      tstop_array(n) = 1.0;
    }
  }

  // for (int n = 0; n < d.PactiveCount; n++) {
  //   // d.dustVc[n](RHO, k, j, i) = (1e-5 + 3e-3 * exp(-0.5 * (R - 2.0) * (R
  //   // - 2.0) / 0.1 / 0.1)) * d.Vc(RHO, k, j, i); // real z = 0.1;
  //   real r = 5.0 + 5.0 * n / d.PactiveCount;
  //   // real theta = acos(z / r);
  //   real theta = 3.14 * (n + 1) / d.PactiveCount;

  //   d.Ps(PX1, n) = r;
  //   d.Ps(PX2, n) = theta;
  //   d.Ps(PX3, n) = 0.0;
  //   d.Ps(PVX1, n) = 0.0;
  //   d.Ps(PVX2, n) = 0.0;
  //   d.Ps(PVX3, n) = 1 / sqrt(r);
  //   d.Ps(PMASS, n) = PM;
  //   tstop_array(n) = 1.0;
  // }

  // Send it all, if needed
  d.SyncToDevice();
}
