#include "setup.hpp"
#include "../../shared/wind/bc.hpp"
#include "../../shared/wind/floor.hpp"
#include "../../shared/wind/ic.hpp"
#include "../../shared/wind/ionisation.hpp"
#include "../../shared/wind/params.hpp"
#include "dumpImage.hpp"
#include "idefix.hpp"
#include <cmath>

using namespace Params;

real rmin;
real rmax;
real sizemin;
real sizemax;
real thetamin;
real thetamax;
real num_r;
real num_theta;
real num_size;

static std::string reload_path;
static std::string initialvelocity;

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
      real Omega = 1.0; // assuming Rin = 1.0
      idefix_for(
          "UserDefBoundaryDustIleft", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
            real R = x1(i) * sin(x2(j));

            Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
            if (Vc(VX1, k, j, ighost) >= ZERO_F) {
              Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i);
            } else {
              Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            }
            Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
            Vc(VX3, k, j, i) = R * Omega;
          });
    } else if (side == right) {
      ighost = data->end[IDIR] - 1;
      ibeg = data->end[IDIR];
      iend = data->np_tot[IDIR];
      idefix_for(
          "UserDefBoundaryDustIRight", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
            real R = x1(i);

            Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
            if (Vc(VX1, k, j, ighost) <= ZERO_F) {
              Vc(VX1, k, j, i) = 0.0;
            } else
              Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
            Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
          });
    }
  }

  if (dir == JDIR) {

    const int j_beg = data->beg[JDIR];
    const int j_end = data->end[JDIR];

    if (side == left) {
      idefix_for(
          "UserDefX2_Left_Vc_Dust", 0, data->np_tot[KDIR], 0, j_beg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_beg - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
          });

    } else if (side == right) {
      idefix_for(
          "UserDefX2_Right_Vc_Dust", 0, data->np_tot[KDIR], j_end, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_end - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
          });
    }
  }
}

void MyDrag(DataBlock *data, real beta, IdefixArray3D<real> &gamma) {
  // Compute the drag coefficient gamma from the input beta
  auto VcGas = data->hydro->Vc;

  idefix_for(
      "MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real cs = sqrt(VcGas(PRS, k, j, i) / VcGas(RHO, k, j, i));
        gamma(k, j, i) = cs / beta;
      });
}

void UserdefBoundaryParticles(DataBlock &data, real t, int dir, BoundarySide side) {
  idfx::pushRegion("UserdefBoundaryParticles");
  auto states = data.particles->pack->states;
  IdefixArray1D<bool> isActive = data.particles->pack->isActive;
  int capacity = data.particles->pack->capacity;

  IdefixHostArray1D<int> host_counter = IdefixHostArray1D<int>("counter", 1);
  IdefixArray1D<int> device_counter = IdefixArray1D<int>("counter", 1);

  // IdefixHostArray1D<real> host_mass_counter = IdefixHostArray1D<real>("mass_counter", 1);
  // IdefixArray1D<real> device_mass_counter = IdefixArray1D<real>("counter", 1);

  if ((dir == IDIR) && (side == left)) {
    const real xl = data.mygrid->xbeg[IDIR];
    idefix_for(
        "UserdefParticlesBoundary", 0, capacity, KOKKOS_LAMBDA(int idx) {
          real x = states(dir, idx);
          if (isActive(idx) && x < xl) {
            isActive(idx) = false;
            Kokkos::atomic_add(&device_counter(0), 1);
            // Kokkos::atomic_add(&device_mass_counter(0), states(PMASS, idx));
          }
        });
    Kokkos::deep_copy(host_counter, device_counter);

    data.particles->pack->activeCount -= host_counter(0);

    // Kokkos::deep_copy(host_mass_counter, device_mass_counter);
    // data.gravity->centralMass += 4 * M_PI * host_mass_counter(0);
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

void ParticlesBodyForce(DataBlock &data, const real t, const real dt, IdefixArray2D<real> &force) {
  idfx::pushRegion("ParticlesBodyForce");

  auto states = data.particles->pack->states;
  auto isActive = data.particles->pack->isActive;
  auto phi_array = data.particles->pack->fields.GetField<real>("phi");

  idefix_for(
      "BodyForce", 0, data.particles->pack->maxActiveIndex + 1, KOKKOS_LAMBDA(int idx) {
        if (isActive(idx)) {
          force(IDIR, idx) = ZERO_F;
          force(JDIR, idx) = ZERO_F;
          force(KDIR, idx) = ZERO_F;
          real r = states(PX1, idx);
          real theta = states(PX2, idx);
          phi_array(idx) = phi_array(idx) + states(PVX3, idx) / (r * sin(theta)) * dt;
        }
      });
  idfx::popRegion();
}

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");

  idefix_for(
      "BodyForce", data.beg[KDIR], data.end[KDIR], data.beg[JDIR], data.end[JDIR], data.beg[IDIR], data.end[IDIR], KOKKOS_LAMBDA(int idx, int j, int i) {
        force(IDIR, idx, j, i) = ZERO_F;
        force(JDIR, idx, j, i) = ZERO_F;
        force(KDIR, idx, j, i) = ZERO_F;
      });

  idfx::popRegion();
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

  IdefixHostArray3D<real> cs = variables["cs"];

#ifdef ISOTHERMAL
  IdefixArray3D<real> scrh_cs("Scratch_cs", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  MySoundSpeed(data, data.t, scrh_cs);
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
#else
        cs(k, j, i) = sqrt(gammaGlob * d.Vc(PRS, k, j, i) / d.Vc(RHO, k, j, i));
#endif
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
  data.hydro->EnrollAmbipolarDiffusivity(&Wind::Ambipolar);
  data.hydro->EnrollOhmicDiffusivity(&Wind::Resistivity);
  data.hydro->EnrollUserSourceTerm(&Wind::MySourceTerm);
  data.hydro->EnrollInternalBoundary(&Wind::InternalBoundary);
  data.hydro->EnrollEmfBoundary(&Wind::EmfBoundary);
  data.particles->EnrollUserDefBoundary(&UserdefBoundaryParticles);

  data.gravity->EnrollBodyForce(BodyForce);
  data.particles->EnrollBodyForce(ParticlesBodyForce);

  if (data.haveDust) {
    int nSpecies = data.dust.size();
    for (int n = 0; n < nSpecies; n++) {
      data.dust[n]->EnrollUserDefBoundary(&UserdefBoundaryDust);
      data.dust[n]->drag->EnrollUserDrag(&MyDrag);
    }
  }

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

  reload_path = input.Get<std::string>("Setup", "reload_path", 0);
  initialvelocity = input.Get<std::string>("Particles", "initialvelocity", 0);

  num_r = input.Get<real>("Particles", "num_r", 0);
  num_theta = input.Get<real>("Particles", "num_theta", 0);
  num_size = input.Get<real>("Particles", "num_size", 0);
  rmin = input.Get<real>("Particles", "rmin", 0);
  rmax = input.Get<real>("Particles", "rmax", 0);
  hmin = input.Get<real>("Particles", "hmin", 0);
  hmax = input.Get<real>("Particles", "hmax", 0);
  sizemin = input.Get<real>("Particles", "sizemin", 0);
  sizemax = input.Get<real>("Particles", "sizemax", 0);
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
  // Create a host copy
  DataBlockHost d(data);
  DumpImage image(reload_path, &data);

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
        d.Vc(BX3, k, j, i) = image.arrays["Vc-BX3"](kglob, jglob, iglob);
      }
    }
  }

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR] + IOFFSET; i++) {
        int iglob = i - 2 * d.beg[IDIR] + d.gbeg[IDIR];
        int jglob = j - 2 * d.beg[JDIR] + d.gbeg[JDIR];
        int kglob = k - 2 * d.beg[KDIR] + d.gbeg[KDIR];
        d.Vs(BX1s, k, j, i) = image.arrays["Vs-BX1s"](kglob, jglob, iglob);
      }
    }
  }

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR] + JOFFSET; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        int iglob = i - 2 * d.beg[IDIR] + d.gbeg[IDIR];
        int jglob = j - 2 * d.beg[JDIR] + d.gbeg[JDIR];
        int kglob = k - 2 * d.beg[KDIR] + d.gbeg[KDIR];
        d.Vs(BX2s, k, j, i) = image.arrays["Vs-BX2s"](kglob, jglob, iglob);
      }
    }
  }

  real ntot = d.PactiveCount;

  real rho0 = 6.0e-10;
  real rhos = 1.0; // 1 g/cm3
  real au = 1.5e11;

  real betamin = rhos * sizemin / (rho0 * au);
  real betamax = rhos * sizemax / (rho0 * au);

  real massmin = betamin * (sizemin / au) * (sizemin / au);
  real massmax = betamax * (sizemax / au) * (sizemax / au);

  real thetamin = M_PI / 2.0 - atan(epsilonGlob * hmax);
  real thetamax = M_PI / 2.0 - atan(epsilonGlob * hmin);

  printf("Number of particles: %.1e\n", ntot);
  printf("dust betas range from: %.1e\n", betamin);
  printf(" to: %.1e\n", betamax);
  printf("dust masses range from: %.1e\n", massmin);
  printf(" to: %.1e\n", massmax);

  // int npart = 0;
  auto x1 = d.x[IDIR];
  auto x2 = d.x[JDIR];

  auto phi_part = d.Pfields.GetField<real>("phi");

  // Many particles
  // default state
  for (int n = 0; n < ntot; n++) {
    d.Ps(PX1, n) = 10.0;
    d.Ps(PX2, n) = 1.0;
    d.Ps(PX3, n) = 0;
    d.Ps(PVX1, n) = 0.0;
    d.Ps(PVX2, n) = 0.0;
    d.Ps(PVX3, n) = 0.0;
    d.Ps(PMASS, n) = 1.0e-3;
    d.Ps(DRAGCOEFF, n) = 1.0;
    phi_part(n) = 0.0;
  }

  int n = -1;
  for (int ii = 0; ii < num_r; ii++) {
    real r0 = rmin + (rmax - rmin) * ii / (num_r - 1);

    for (int jj = 0; jj < num_theta; jj++) {
      real theta0 = thetamin + (thetamax - thetamin) * jj / (num_theta - 1);

      for (int kk = 0; kk < num_size; kk++) {

        // logspace
        real log_betamin = log10(betamin); //
        real log_betamax = log10(betamax); //

        real log_beta = log_betamin + (log_betamax - log_betamin) * kk / (num_size - 1);

        // real beta = betamin + (betamax - betamin) * kk / (num_size - 1);
        real beta = pow(10, log_beta);

        n++;

        for (int k = 0; k < d.np_tot[KDIR]; k++) {
          for (int j = 0; j < d.np_tot[JDIR]; j++) {
            for (int i = 0; i < d.np_tot[IDIR]; i++) {
              real r = d.x[IDIR](i);
              real theta = d.x[JDIR](j);
              real R = r * sin(theta);
              real dr = d.dx[IDIR](i);
              real dtheta = d.dx[JDIR](j);

              if (abs(r - r0) < dr / 2 && abs(theta - theta0) < dtheta / 2 && n < ntot) {

                real OmegaK = pow(R, -1.5);

                real vr;
                real vtheta;
                real vphi;
                real cs = sqrt(d.Vc(PRS, k, j, i) / d.Vc(RHO, k, j, i));

                real tstop = beta / (d.Vc(RHO, k, j, i) * cs);

                if (initialvelocity == "tv") {
                  real delta = tstop * (pow(d.Vc(VX2, k, j, i), 2) / r - 1.0 / pow(r, 2));
                  vr = d.Vc(VX1, k, j, i) + delta;
                  vtheta = d.Vc(VX2, k, j, i);
                  vphi = d.Vc(VX3, k, j, i);
                } else if (initialvelocity == "zero") {
                  vr = 0.0;
                  vtheta = 0.0;
                  vphi = R * OmegaK;
                } else {
                  vr = d.Vc(VX1, k, j, i);
                  vtheta = d.Vc(VX2, k, j, i);
                  vphi = R * OmegaK;
                }

                d.Ps(PX1, n) = r;
                d.Ps(PX2, n) = theta;
                d.Ps(PX3, n) = 0.0;
                d.Ps(PVX1, n) = vr;
                d.Ps(PVX2, n) = vtheta;
                d.Ps(PVX3, n) = vphi;
                d.Ps(PMASS, n) = 1.0e-10;
                d.Ps(DRAGCOEFF, n) = beta;
                d.Ps(TSTOP, n) = tstop;
                // }
              }
            }
          }
        }
      }
    }
  }

  // delete image;

  // Send it all, if needed
  d.SyncToDevice();
}
