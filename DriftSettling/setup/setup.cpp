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

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  real h0 = h0Glob;
  real CsSlope = CsSlopeGlob;
  idefix_for(
      "MySoundSpeed", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = x1(i) * sin(x2(j));
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

  if (dir == IDIR) {
    if (side == left) {
      const int ibeg = data->beg[IDIR];

      idefix_for(
          "UserDefBoundary_Left", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, ibeg, KOKKOS_LAMBDA(int k, int j, int i) {
            const int irefl = 2 * ibeg - 1 - i;

            Vc(RHO, k, j, i) = Vc(RHO, k, j, irefl);

            if (Vc(VX1, k, j, ibeg) > ZERO_F) {
              Vc(VX1, k, j, i) = -Vc(VX1, k, j, irefl);
            } else {
              Vc(VX1, k, j, i) = Vc(VX1, k, j, irefl);
            }

            Vc(VX2, k, j, i) = Vc(VX2, k, j, irefl);
            Vc(VX3, k, j, i) = Vc(VX3, k, j, irefl);
          });

    } else if (side == right) {
      const int iend = data->end[IDIR];

      idefix_for(
          "UserDefBoundary_Right", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], iend, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int irefl = 2 * iend - 1 - i;

            Vc(RHO, k, j, i) = Vc(RHO, k, j, irefl);

            // Diode boundary: Allow outflow (positive VX1), kill inflow (negative VX1)
            if (Vc(VX1, k, j, iend - 1) < ZERO_F) {
              Vc(VX1, k, j, i) = -Vc(VX1, k, j, irefl);
            } else {
              Vc(VX1, k, j, i) = Vc(VX1, k, j, irefl);
            }

            Vc(VX2, k, j, i) = Vc(VX2, k, j, irefl);
            Vc(VX3, k, j, i) = Vc(VX3, k, j, irefl);
          });
    }
  }
  if (dir == JDIR) {
    if (side == left) {
      const int jbeg = data->beg[JDIR];

      idefix_for(
          "UserDefBoundary_Left", 0, data->np_tot[KDIR], 0, jbeg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jbeg - 1 - j;
            real r = x1(i);
            real theta = x2(j);
            real R = r * sin(theta);
            real Omega = pow(R, -1.5);

            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

            if (Vc(VX2, k, jbeg, i) > ZERO_F) {
              Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            } else {
              Vc(VX2, k, j, i) = Vc(VX2, k, jrefl, i);
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX3, k, j, i) = Vc(VX3, k, jrefl, i);
          });

    } else if (side == right) {
      const int jend = data->end[JDIR];

      idefix_for(
          "UserDefBoundary_Right", 0, data->np_tot[KDIR], jend, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jend - 1 - j;

            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

            if (Vc(VX2, k, jend - 1, i) < ZERO_F) {
              Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            } else {
              Vc(VX2, k, j, i) = Vc(VX2, k, jrefl, i);
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX3, k, j, i) = Vc(VX3, k, jrefl, i);
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
  // auto VcDust = data->dust[nSpecie]->Vc;
  // auto cs = data->hydro->cs;
  IdefixArray1D<real> x1 = data->x[IDIR];

  real h0 = h0Glob;
  real CsSlope = CsSlopeGlob;

  idefix_for(
      "MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        gamma(k, j, i) = 1 / (beta * VcGas(RHO, k, j, i));
        // // real cs = sqrt(VcGas(PRS, k, j, i) / VcGas(RHO, k, j, i));
        // real R = x1(i);
        // real cs = h0 * pow(R, CsSlope);

        // gamma(k, j, i) = cs / beta;
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
            int irefl = 2 * ighost - i - 1;

            Vc(RHO, k, j, i) = Vc(RHO, k, j, iref);
            if (Vc(VX1, k, j, ighost) >= ZERO_F) {
              Vc(VX1, k, j, i) = 0.0;
            } else {
              // Vc(VX1,k,j,i) = - Vc(VX1,k,j,2*ighost - i +1);
              Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            }
            Vc(VX2, k, j, i) = Vc(VX2, k, j, iref);
            Vc(VX3, k, j, i) = Vc(VX3, k, j, iref);
          });
    } else if (side == right) {
      ighost = data->end[IDIR] - 1;
      ibeg = data->end[IDIR];
      iend = data->np_tot[IDIR];
      idefix_for(
          "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
            Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            if (Vc(VX1, k, j, ighost) <= ZERO_F) {
              Vc(VX1, k, j, i) = 0.0;
            }
            // if(Vc(VX1,k,j,i)<=ZERO_F){
            //   Vc(VX1,k,j,ighost) = 0.0;
            // }
            Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
            Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
          });
    }
  }
  if (dir == JDIR) {
    if (side == left) {
      const int jbeg = data->beg[JDIR];

      idefix_for(
          "UserDefBoundary_Left", 0, data->np_tot[KDIR], 0, jbeg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jbeg - 1 - j;

            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

            if (Vc(VX2, k, jbeg, i) > ZERO_F) {
              Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            } else {
              Vc(VX2, k, j, i) = Vc(VX2, k, jrefl, i);
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX3, k, j, i) = Vc(VX3, k, jrefl, i);
          });

    } else if (side == right) {
      const int jend = data->end[JDIR];

      idefix_for(
          "UserDefBoundary_Right", 0, data->np_tot[KDIR], jend, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * jend - 1 - j;

            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

            if (Vc(VX2, k, jend - 1, i) < ZERO_F) {
              Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            } else {
              Vc(VX2, k, j, i) = Vc(VX2, k, jrefl, i);
            }

            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX3, k, j, i) = Vc(VX3, k, jrefl, i);
          });
    }
  }
}

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) // : m_planet(0)//, Planet &planet)
{
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
  // data.particles->EnrollUserDefBoundary(&UserdefBoundaryParticles);
  if (data.haveDust) {
    int nSpecies = data.dust.size();
    for (int n = 0; n < nSpecies; n++) {
      data.dust[n]->EnrollUserDefBoundary(&UserdefBoundaryDust);
      // data.dust[n]->drag->EnrollUserDrag(&MyDrag);
    }
  }

  sigmaSlopeGlob = input.Get<real>("Setup", "sigmaSlope", 0);
  sigma0Glob = input.Get<real>("Setup", "sigma0", 0);
  CsSlopeGlob = input.Get<real>("Setup", "CsSlope", 0);
  h0Glob = input.Get<real>("Setup", "h0", 0);

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
        real r = d.x[IDIR](i);
        real theta = d.x[JDIR](j);

        real z = r * cos(theta);
        real R = r * sin(theta);
        real OmegaK = pow(R, -1.5);
        real cs2 = h0 * h0 * pow(R, 2 * CsSlope);
        real hg2 = cs2 / OmegaK / OmegaK;

        d.Vc(RHO, k, j, i) = 1e-6 + sigma0 * pow(R, sigmaSlope - 1.0) * exp(-z * z / (2 * hg2));
        d.Vc(VX1, k, j, i) = 0.0;
        d.Vc(VX2, k, j, i) = 0.0;
        // d.Vc(VX3, k, j, i) = R * OmegaK;
        d.Vc(VX3, k, j, i) = R * OmegaK * (1 + 0.5 * (hg2 / R / R) * (sigmaSlope - 1.0 + 2 * CsSlope + 2 * CsSlope * z * z / 2.0 / hg2));

        for (int n = 0; n < data.dust.size(); n++) {
          d.dustVc[n](RHO, k, j, i) = 1e-8 + (1e-2 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.05 / 0.05)) * exp(-0.5 * (z * z) / (0.2 * 0.2)) * d.Vc(RHO, k, j, i);
          d.dustVc[n](VX1, k, j, i) = 0.0;
          d.dustVc[n](VX2, k, j, i) = 0.0;
          d.dustVc[n](VX3, k, j, i) = R * OmegaK;
        }
      }
    }
  };

  // for (int n = 0; n < d.PactiveCount; n++) {
  //   //   // d.dustVc[n](RHO, k, j, i) = (1e-5 + 3e-3 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.1 / 0.1)) * d.Vc(RHO, k, j, i); //
  //   real r = 2.0;
  //   d.Ps(PX1, n) = r;
  //   d.Ps(PX2, n) = 1.5707963267948966;
  //   d.Ps(PX3, n) = 0.0;
  //   d.Ps(PVX1, n) = 0.0;
  //   d.Ps(PVX2, n) = 0.0;
  //   d.Ps(PVX3, n) = 1 / sqrt(r);
  //   d.Ps(PMASS, n) = 1e-3;
  // }

  // Send it all, if needed
  d.SyncToDevice();
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock &data) {}