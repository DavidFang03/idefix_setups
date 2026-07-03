#include "setup.hpp"
#include "idefix.hpp"
#include <algorithm>

real sigmaSlopeGlob;
real CsSlopeGlob;
real sigma0Glob;
real h0Glob;
real HidealGlob;
real gammaGlob;
real densityFloorGlob;
real alphaGlob;
real sizeGlob;

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  real h0 = h0Glob;
  real CsSlope = CsSlopeGlob;
  idefix_for(
      "MySoundSpeed", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = x1(i);
        // real h = h0 * pow(R, CsSlope +)
        cs(k, j, i) = R * h0 * pow(R, CsSlope - 1);
      });
}

void ComputeUserVars(DataBlock &data, UserDefVariablesContainer &variables) {

  // Use Invdt as scratch array
  IdefixArray3D<real> scrh("Scratch", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  IdefixArray3D<real> scrh_cs("Scratch_cs", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

  // Ask for a computation of xA ambipolar in this scratch array
  MySoundSpeed(data, data.t, scrh_cs);

  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  // Make references to the user-defined arrays (variables is a container of
  // IdefixHostArray3D) Note that the labels should match the variable names in
  // the input file
  IdefixHostArray3D<real> cs = variables["cs"];

  IdefixHostArray4D<real> Vc = d.Vc;

  IdefixArray3D<real>::HostMirror scrhHost_cs = Kokkos::create_mirror_view(scrh_cs);
  Kokkos::deep_copy(scrhHost_cs, scrh_cs);

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        cs(k, j, i) = scrhHost_cs(k, j, i);
      }
    }
  }
}

// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  auto *data = hydro->data;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];

  real h0 = h0Glob;
  real sigmaSlope = sigmaSlopeGlob;
  real sigma0 = sigma0Glob;

  real CsSlope = CsSlopeGlob;

  if (dir == IDIR) {
    if (side == left) {
      const int ibeg = data->beg[IDIR];

      idefix_for(
          "UserDefBoundary_Left", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, ibeg, KOKKOS_LAMBDA(int k, int j, int i) {
            const int irefl = 2 * ibeg - 1 - i;
            real R = x1(i);
            real z = x3(i);
            real Omega = pow(R, -1.5);

            real cs2 = h0 * h0 * pow(R, 2 * CsSlope);
            real hg2 = cs2 / Omega / Omega;

            Vc(RHO, k, j, i) = Vc(RHO, k, j, irefl);

            // Vc(VX1, k, j, i) = Vc(VX1, k, j, irefl);
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ibeg);
            if (Vc(VX1, k, j, ibeg) > ZERO_F) {
              Vc(VX1, k, j, i) = -Vc(VX1, k, j, irefl); // Block inflow
            } else {
              Vc(VX1, k, j, i) = Vc(VX1, k, j, irefl); // Allow natural outflow (mirrored or copied)
            }

            Vc(VX3, k, j, i) = Vc(VX3, k, j, irefl);
            Vc(VX2, k, j, i) = R * Omega * (1 + 0.5 * (hg2 / R / R) * (sigmaSlope - 1.0 + 2 * CsSlope + 2 * CsSlope * z * z / 2.0 / hg2)); // Mirror vertical velocity
          });

    } else if (side == right) {
      const int iend = data->end[IDIR]; // First ghost cell on the right

      idefix_for(
          "UserDefBoundary_Right", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], iend, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int irefl = 2 * iend - 1 - i; // Correct mirrored index for right side
            real R = x1(i);
            real Vk = 1.0 / sqrt(R);

            Vc(RHO, k, j, i) = Vc(RHO, k, j, irefl);

            // Diode boundary: Allow outflow (positive VX1), kill inflow (negative VX1)
            if (Vc(VX1, k, j, iend - 1) < ZERO_F) {
              Vc(VX1, k, j, i) = -Vc(VX1, k, j, irefl);
            } else {
              Vc(VX1, k, j, i) = Vc(VX1, k, j, irefl);
            }
            // Vc(VX1, k, j, i) = 0.0;

            Vc(VX3, k, j, i) = Vc(VX3, k, j, irefl);
            Vc(VX2, k, j, i) = Vc(VX2, k, j, irefl);
          });
    }
  }
  if (dir == JDIR) {
    if (side == left) {
      const int jbeg = data->beg[JDIR];

      idefix_for(
          "UserDefBoundary_Left", 0, data->np_tot[KDIR], 0, jbeg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jbeg - 1 - j;
            real R = x1(i);
            real Omega = pow(R, -1.5);

            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

            if (Vc(VX3, k, jbeg, i) > ZERO_F) {
              Vc(VX3, k, j, i) = 0.0; // Block inflow
            } else {
              Vc(VX3, k, j, i) = Vc(VX3, k, jrefl, i); // Allow natural outflow (mirrored or copied)
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i); // Mirror vertical velocity
            Vc(VX2, k, j, i) = R * Omega;            // Mirror vertical velocity
          });

    } else if (side == right) {
      const int jend = data->end[JDIR]; // First ghost cell on the right

      idefix_for(
          "UserDefBoundary_Right", 0, data->np_tot[KDIR], jend, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jend - 1 - j; // Correct mirrored index for right side
            real R = x1(i);
            real Vk = 1.0 / sqrt(R);

            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

            if (Vc(VX3, k, jend - 1, i) < ZERO_F) {
              Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i); // Block inflow
            } else {
              Vc(VX3, k, j, i) = Vc(VX3, k, jrefl, i); // Allow natural outflow
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = Vc(VX2, k, jrefl, i);
          });
    }
  }
}

void UserdefBoundaryParticles(DataBlock &data, real t, int dir, BoundarySide side) {
  idfx::pushRegion("UserdefBoundaryParticles");
  auto states = data.particles->pack->states;
  IdefixArray1D<bool> isActive = data.particles->pack->isActive;
  int capacity = data.particles->pack->capacity;
  //
  IdefixHostArray1D<int> host_counter = IdefixHostArray1D<int>("counter", 1);
  IdefixArray1D<int> device_counter = IdefixArray1D<int>("counter", 1);
  //
  IdefixHostArray1D<real> host_mass_counter = IdefixHostArray1D<real>("mass_counter", 1);
  IdefixArray1D<real> device_mass_counter = IdefixArray1D<real>("counter", 1);
  //
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
    //
    data.particles->pack->activeCount -= host_counter(0);
    //
    // Kokkos::deep_copy(host_mass_counter, device_mass_counter);
    // data.gravity->centralMass += 4 * M_PI * host_mass_counter(0);
  }
  //
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
  //
  if (data.particles->pack->activeCount < 0)
    data.particles->pack->activeCount = 0;
  //
  idfx::popRegion();
}

// void MyDrag(DataBlock *data, int nSpecie, real beta, IdefixArray3D<real> &gamma) {
void MyDrag(DataBlock *data, real beta, IdefixArray3D<real> &gamma) {
  // Compute the drag coefficient gamma from the input beta
  auto VcGas = data->hydro->Vc;

  real rho0 = 6.0e-10;
  real rhos = 1.0; // 1 g/cm3
  real au = 1.5e11;
  real realbeta = rhos * sizeGlob / (rho0 * au);
  // auto VcDust = data->dust[nSpecie]->Vc;
  // auto cs = data->hydro->cs;
  IdefixArray1D<real> x1 = data->x[IDIR];

  real h0 = h0Glob;
  real CsSlope = CsSlopeGlob;

  idefix_for(
      "MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        // gamma(k, j, i) = 1 / (beta * VcGas(RHO, k, j, i));
        // real cs = sqrt(VcGas(PRS, k, j, i) / VcGas(RHO, k, j, i));
        real R = x1(i);
        real realbetaloc = realbeta;
        real cs = h0 * pow(R, CsSlope);

        gamma(k, j, i) = cs / realbetaloc;
      });
}

void UserdefBoundaryDust(Fluid<DustPhysics> *dust, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = dust->Vc;
  auto data = dust->data;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
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

            Vc(RHO, k, j, i) = 1e-5;
            if (Vc(VX1, k, j, ighost) >= ZERO_F) {
              Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i + 1);
            } else {
              // Vc(VX1,k,j,i) = - Vc(VX1,k,j,2*ighost - i +1);
              Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            }
            Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
            Vc(VX2, k, j, i) = Vk;
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
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            if (Vc(VX1, k, j, ighost) <= ZERO_F) {
              Vc(VX1, k, j, i) = 0.0;
            }
            // if(Vc(VX1,k,j,i)<=ZERO_F){
            //   Vc(VX1,k,j,ighost) = 0.0;
            // }
            Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
            Vc(VX2, k, j, i) = Vk;
          });
    }
  }
  if (dir == JDIR) {
    if (side == left) {
      const int jbeg = data->beg[JDIR];

      idefix_for(
          "UserDefBoundary_Left", 0, data->np_tot[KDIR], 0, jbeg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jbeg - 1 - j;
            real R = x1(i);
            real Omega = pow(R, -1.5);

            Vc(RHO, k, j, i) = Vc(RHO, k, jbeg, i);

            if (Vc(VX3, k, jbeg, i) > ZERO_F) {
              Vc(VX3, k, j, i) = 0.0; // Block inflow
            } else {
              Vc(VX3, k, j, i) = Vc(VX3, k, jbeg, i); // Allow natural outflow (mirrored or copied)
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jbeg, i); // Mirror vertical velocity
            Vc(VX2, k, j, i) = R * Omega;           // Mirror vertical velocity
          });

    } else if (side == right) {
      const int jend = data->end[JDIR]; // First ghost cell on the right

      idefix_for(
          "UserDefBoundary_Right", 0, data->np_tot[KDIR], jend, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jend - 1 - j; // Correct mirrored index for right side
            real R = x1(i);
            real Vk = 1.0 / sqrt(R);

            Vc(RHO, k, j, i) = Vc(RHO, k, jend, i);

            if (Vc(VX3, k, jend - 1, i) < ZERO_F) {
              Vc(VX3, k, j, i) = 0.0; // Block inflow
            } else {
              Vc(VX3, k, j, i) = Vc(VX3, k, jend, i); // Allow natural outflow
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jend, i);
            Vc(VX2, k, j, i) = Vk;
          });
    }
  }
}

// void UserdefStoppingTime(DataBlock &data, const real t, IdefixArray1D<real> &tstop) {
//   // a separate hook is needed for particles because gravity isn't in general
//   // computed at the same time for particles and the fluid.

//   // GPUS cannot capture static variables
//   // auto states = data.particles->pack->states;
//   auto states = data.particles->pack->states;
//   auto isActive = data.particles->pack->isActive;

//   DataBlockHost d(data);

//   IdefixArray4D<real> Vc = data.hydro->Vc;

//   int i_gbeg = d.gbeg[IDIR];
//   int j_gbeg = d.gbeg[JDIR];
//   int ibeg = d.beg[IDIR];
//   int jbeg = d.beg[JDIR];
//   int iend = d.end[IDIR];
//   int kbeg = d.beg[KDIR];

//   real iint = d.np_int[IDIR];
//   real r_beg = d.x[IDIR](ibeg);
//   real r_end = d.x[IDIR](iend);
//   real dr = d.x[IDIR](ibeg + 1) - d.x[IDIR](ibeg);

//   // real theta_beg = d.x[JDIR](jbeg);
//   // real dtheta = d.x[JDIR](jbeg + 1) - d.x[JDIR](jbeg);
//   // real phi_beg = d.x[KDIR](kbeg);
//   // real dphi = d.x[KDIR](kbeg + 1) - d.x[KDIR](kbeg);

//   real sigma0 = sigma0Glob;
//   real sigmaSlope = sigmaSlopeGlob;
//   real h0 = h0Glob;
//   real CsSlope = CsSlopeGlob;

//   idefix_for(
//       "StoppingTime", 0, data.particles->pack->maxActiveIndex + 1, KOKKOS_LAMBDA(int idx) {
//         if (isActive(idx)) {

//           real r = states(PX1, idx);
//           real theta = states(PX2, idx);
//           real phi = states(PX3, idx);

//           // int i = floor(iint * (log(r / r_beg)) / log(r_end / r_beg)) + ibeg; //TODO SWITCH FOR LOG
//           int i = floor((r - r_beg) / dr) + ibeg;
//           // int j = floor((theta - theta_beg) / dtheta) + jbeg;
//           int j = jbeg;
//           // int k = floor((phi - phi_beg) / dphi) + kbeg;
//           int k = kbeg;

//           real beta = 0.01;
//           real cs = h0 * pow(r, CsSlope); // TODO change that for wind

//           // tstop(idx) = 1.0;
//           tstop(idx) = beta / (cs * Vc(RHO, kbeg, j, i));
//         }
//       });
// }

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) // : m_planet(0)//, Planet &planet)
{
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
  data.particles->EnrollUserDefBoundary(&UserdefBoundaryParticles);
  if (data.haveDust) {
    int nSpecies = data.dust.size();
    for (int n = 0; n < nSpecies; n++) {
      data.dust[n]->EnrollUserDefBoundary(&UserdefBoundaryDust);
      data.dust[n]->drag->EnrollUserDrag(&MyDrag);
    }
  }

  sigmaSlopeGlob = input.Get<real>("Setup", "sigmaSlope", 0);
  sigma0Glob = input.Get<real>("Setup", "sigma0", 0);
  CsSlopeGlob = input.Get<real>("Setup", "CsSlope", 0);
  h0Glob = input.Get<real>("Setup", "h0", 0);

  sizeGlob = input.Get<real>("Particles", "stopping_time", 1);

  output.EnrollUserDefVariables(&ComputeUserVars);
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

  real CsSlope = CsSlopeGlob;

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real R = d.x[IDIR](i);
        real z = d.x[KDIR](k);

        real OmegaK = pow(R, -1.5);
        real cs2 = h0 * h0 * pow(R, 2 * CsSlope);
        real hg2 = cs2 / OmegaK / OmegaK;

        d.Vc(RHO, k, j, i) = 1e-6 + sigma0 * pow(R, sigmaSlope - 1.0) * exp(-z * z / (2 * hg2));
        d.Vc(VX1, k, j, i) = 0.0;
        d.Vc(VX3, k, j, i) = 0.0;
        // d.Vc(VX2, k, j, i) = R * OmegaK;
        d.Vc(VX2, k, j, i) = R * OmegaK * (1 + 0.5 * (hg2 / R / R) * (sigmaSlope - 1.0 + 2 * CsSlope + 2 * CsSlope * z * z / 2.0 / hg2));

        for (int n = 0; n < data.dust.size(); n++) {
          d.dustVc[n](RHO, k, j, i) = (1e-5 + 1e-2 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.05 / 0.05)) * exp(-0.5 * (z * z) / (0.2 * 0.2)) * d.Vc(RHO, k, j, i);
          d.dustVc[n](VX1, k, j, i) = 0.0;
          d.dustVc[n](VX3, k, j, i) = 0.0;
          d.dustVc[n](VX2, k, j, i) = R * OmegaK;
        }
      }
    }
  };

  real rho0 = 6.0e-10;
  real rhos = 1.0; // 1 g/cm3
  real au = 1.5e11;
  real beta = rhos * sizeGlob / (rho0 * au);

  for (int n = 0; n < d.PactiveCount; n++) {
    //   // d.dustVc[n](RHO, k, j, i) = (1e-5 + 3e-3 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.1 / 0.1)) * d.Vc(RHO, k, j, i); //
    real r = 2.0;
    d.Ps(PX1, n) = r;
    d.Ps(PX2, n) = 0.0;
    d.Ps(PX3, n) = 0.0;
    d.Ps(PVX1, n) = 0.0;
    d.Ps(PVX3, n) = 0.0;
    d.Ps(PVX2, n) = 1 / sqrt(r);
    d.Ps(PMASS, n) = 1e-3;
    d.Ps(DRAGCOEFF, n) = beta;
  }

  // Send it all, if needed
  d.SyncToDevice();
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock &data) {}