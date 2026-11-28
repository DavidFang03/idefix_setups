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
real hmin;
real hmax;
real num_r;
real num_theta;
real num_size;

real rho0 = 6.0e-10; // rho = 1 code units corresponds to this value in g/cm3
real rhos = 1.0;     // 1 g/cm3
real au = 1.5e11;    // r=1 is 1 au i.e 1.5e11 m

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

void MyDrag(DataBlock *data, real size, IdefixArray3D<real> &gamma) {
  // Compute the drag coefficient gamma = 1/(rhos a) with Esptein regime
  // input value is mass in meter.
  real beta = (rhos * size) / (rho0 * au);

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
  IdefixArray3D<real> scrh_eta("Scratch_eta", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

  // Ask for a computation of xA ambipolar in this scratch array

#ifndef DISABLE_MHD
  Wind::Resistivity(data, data.t, scrh_eta);
  Wind::Ambipolar(data, data.t, scrh);
#endif

  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  // Make references to the user-defined arrays (variables is a container of IdefixHostArray3D)
  // Note that the labels should match the variable names in the input file
  IdefixHostArray3D<real> Eta = variables["eta"];
  IdefixHostArray3D<real> Am = variables["Am"];
  IdefixHostArray3D<real> InvDt = variables["InvDt"];
  IdefixHostArray3D<real> Ephi = variables["Ephi"];

  IdefixHostArray1D<real> x1 = d.x[IDIR];
  IdefixHostArray1D<real> x2 = d.x[JDIR];
  IdefixHostArray1D<real> x3 = d.x[KDIR];
  IdefixHostArray4D<real> Vc = d.Vc;
  IdefixArray3D<real>::host_mirror_type scrhHost = Kokkos::create_mirror_view(scrh);
  Kokkos::deep_copy(scrhHost, scrh);
  IdefixArray3D<real>::host_mirror_type scrhHost_eta = Kokkos::create_mirror_view(scrh_eta);
  Kokkos::deep_copy(scrhHost_eta, scrh_eta);

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
        Eta(k, j, i) = scrhHost_eta(k, j, i);
        Am(k, j, i) = scrhHost(k, j, i); // eta_A / B^2
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

real get_beta(real size) { return rhos * sizemin / (rho0 * au); }
real get_tstop(real size, real rho, real prs) {
  // = beta/(rho cs) = beta/sqrt(rho prs)
  return get_beta(size) / sqrt(rho * prs);
}

std::pair<real, real> v_tv(real ur, real utheta, real r, real theta, real tstop) {
  real ux = sin(theta) * ur + cos(theta) * utheta;
  real uz = cos(theta) * ur - sin(theta) * utheta;
  real deltax = tstop * (-sin(theta) / pow(r, 2));
  real deltaz = tstop * (-cos(theta) / pow(r, 2));
  real vx = ux + deltax;
  real vz = uz + deltaz;

  real vr = sin(theta) * vx + cos(theta) * vz;
  real vtheta = cos(theta) * vx - sin(theta) * vz;

  return {vr, vtheta};
}

void fill_all_upper_cells_with_particles(DataBlockHost &d) {
  real size = sizemin;
  int k = d.beg[KDIR];
  int n = -1;
  for (int j = 0; j < d.np_tot[JDIR]; j++) {
    for (int i = 0; i < d.np_tot[IDIR]; i++) {
      real r = d.x[IDIR](i);
      real theta = d.x[JDIR](j);
      real phi = d.x[KDIR](k);
      if (n < d.PactiveCount and theta < M_PI / 2) {
        n++;
        real tstop = get_tstop(size, d.Vc(RHO, k, j, i), d.Vc(PRS, k, j, i));
        auto [vr, vtheta] = v_tv(d.Vc(VX1, k, j, i), d.Vc(VX2, k, j, i), r, theta, tstop);

        d.Ps(PX1, n) = r;
        d.Ps(PX2, n) = theta;
        d.Ps(PX3, n) = phi;
        d.Ps(PVX1, n) = vr;
        d.Ps(PVX2, n) = vtheta;
        d.Ps(PVX3, n) = d.Vc(VX3, k, j, i);
        d.Ps(PMASS, n) = size;
        // the real value would be around 1e-40 in code units. I'm afraid of underflows so I rather put a wrong value (the size in m instead). Dynamics should be independent of mass anyway, as long as
        // there's no feedback
        d.Ps(DRAGCOEFF, n) = get_beta(size);
        d.Ps(TSTOP, n) = tstop;
      }
    }
  }
  printf("Initialized %d particles\n", n + 1);

  while (n < d.PactiveCount) { // toss em
    n++;

    d.Ps(PX1, n) = 2.0;
    d.Ps(PX2, n) = M_PI;
    d.Ps(PX3, n) = 0.0;
    d.Ps(PVX1, n) = 0.0;
    d.Ps(PVX2, n) = 0.0;
    d.Ps(PVX3, n) = 0.0;
    d.Ps(PMASS, n) = 1.0;
    // the real value would be around 1e-40 in code units. I'm afraid of underflows so I rather put a wrong value (the size in m instead). Dynamics should be independent of mass anyway, as long as
    // there's no feedback
    d.Ps(DRAGCOEFF, n) = 1.0;
    d.Ps(TSTOP, n) = 1.0;
  }
  printf("Initialized %d particles\n", n + 1);
}

void wedge(DataBlockHost &d) {
  real ntot = d.PactiveCount;

  real betamin = rhos * sizemin / (rho0 * au);
  real betamax = rhos * sizemax / (rho0 * au);

  real massmin = betamin * (sizemin / au) * (sizemin / au); // for logs only
  real massmax = betamax * (sizemax / au) * (sizemax / au); // for logs only

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

  // auto phi_part = d.Pfields.GetField<real>("phi");
  int n = -1;
  int k = d.beg[KDIR];

  for (int ii = 0; ii < num_r; ii++) {
    real frac_r = (num_r > 1) ? (real)ii / (num_r - 1) : 0.0;
    real r0 = rmin + (rmax - rmin) * frac_r;

    // 1. Find closest active radial cell
    int best_i = d.beg[IDIR];
    real min_dr = fabs(d.x[IDIR](best_i) - r0);
    for (int i = d.beg[IDIR] + 1; i <= d.end[IDIR]; i++) {
      real dist = fabs(d.x[IDIR](i) - r0);
      if (dist < min_dr) {
        min_dr = dist;
        best_i = i;
      }
    }

    for (int jj = 0; jj < num_theta; jj++) {
      real frac_th = (num_theta > 1) ? (real)jj / (num_theta - 1) : 0.0;
      real theta0 = thetamin + (thetamax - thetamin) * frac_th;

      // 2. Find closest active theta cell
      int best_j = d.beg[JDIR];
      real min_dth = fabs(d.x[JDIR](best_j) - theta0);
      for (int j = d.beg[JDIR] + 1; j <= d.end[JDIR]; j++) {
        real dist = fabs(d.x[JDIR](j) - theta0);
        if (dist < min_dth) {
          min_dth = dist;
          best_j = j;
        }
      }

      // Sanity check: warn if (r0, theta0) was actually outside the domain
      if (min_dr > 0.5 * d.dx[IDIR](best_i) * 1.01 || min_dth > 0.5 * d.dx[JDIR](best_j) * 1.01) {
        printf("Warning: (r0=%g, theta0=%g) is outside active grid!\n", r0, theta0);
      }

      real r = d.x[IDIR](best_i);
      real theta = d.x[JDIR](best_j);
      real R = r * sin(theta);
      real OmegaK = pow(R, -1.5);
      real cs = sqrt(d.Vc(PRS, k, best_j, best_i) / d.Vc(RHO, k, best_j, best_i));

      real ur = d.Vc(VX1, k, best_j, best_i);
      real utheta = d.Vc(VX2, k, best_j, best_i);
      real uphi = d.Vc(VX3, k, best_j, best_i);

      real ux = sin(theta) * ur + cos(theta) * utheta;
      // real uy = uphi;
      real uz = cos(theta) * ur - sin(theta) * utheta;

      // 3. Assign all sizes at this exact cell
      for (int kk = 0; kk < num_size; kk++) {
        n++;
        if (n >= ntot)
          break;

        real frac_s = (num_size > 1) ? (real)kk / (num_size - 1) : 0.0;
        real log_beta = log10(betamin) + (log10(betamax) - log10(betamin)) * frac_s;
        real beta = pow(10.0, log_beta);
        real log_size = log10(sizemin) + (log10(sizemax) - log10(sizemin)) * frac_s;
        real size = pow(10.0, log_size);
        real tstop = beta / (d.Vc(RHO, k, best_j, best_i) * cs);

        real vr, vtheta, vphi;
        if (initialvelocity == "tv") {
          // spherical -> cartesian -> spherical
          // u = gas velocity
          // v = dust velocity
          // v = u + delta
          // terminal velocity: assuming F + Fgrav = 0 in cartesian
          // with F = -delta/tstop => delta = tstop * Fgrav
          real deltax = tstop * (-sin(theta) / pow(r, 2));
          // real deltay = 0.0;
          real deltaz = tstop * (-cos(theta) / pow(r, 2));
          real vx = ux + deltax;
          // real vy = uy + deltay;
          real vz = uz + deltaz;

          vr = sin(theta) * vx + cos(theta) * vz;
          vtheta = cos(theta) * vx - sin(theta) * vz;
          vphi = uphi;
        } else if (initialvelocity == "zero") {
          vr = 0.0;
          vtheta = 0.0;
          vphi = R * OmegaK;
        } else {
          vr = ur;
          vtheta = utheta;
          vphi = R * OmegaK;
        }

        d.Ps(PX1, n) = r;
        d.Ps(PX2, n) = theta;
        d.Ps(PX3, n) = 0.0;
        d.Ps(PVX1, n) = vr;
        d.Ps(PVX2, n) = vtheta;
        d.Ps(PVX3, n) = vphi;
        d.Ps(PMASS, n) = size;
        // the real value would be around 1e-40 in code units. I'm afraid of underflows so I rather put a wrong value (the size in m instead). Dynamics should be independent of mass anyway, as long as
        // there's no feedback
        d.Ps(DRAGCOEFF, n) = beta;
        d.Ps(TSTOP, n) = tstop;
      }
    }
    printf("Initialized %d particles\n", n + 1);
  }

  // Verify every single particle slot was filled
  if (n + 1 != (int)ntot) {
    printf("ERROR: Initialized %d particles, but PactiveCount is %d!\n", n + 1, (int)ntot);
  } else {
    printf("All %d / %d particles initialized\n", n + 1, (int)ntot);
  }
}

// Default constructor

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {

  // Set the function for userdefboundary
  if (data.haveDust) {
    int nSpecies = data.dust.size();
    for (int n = 0; n < nSpecies; n++) {
      data.dust[n]->EnrollUserDefBoundary(&UserdefBoundaryDust);
      data.dust[n]->drag->EnrollUserDrag(&MyDrag);
    }
  }
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

  fill_all_upper_cells_with_particles(d);

  if (data.haveDust) {
    for (int k = 0; k < d.np_tot[KDIR]; k++) {
      for (int j = 0; j < d.np_tot[JDIR]; j++) {
        for (int i = 0; i < d.np_tot[IDIR]; i++) {
          for (int n = 0; n < data.dust.size(); n++) {
            // d.dustVc[n](RHO, k, j, i) = d.Vc(RHO, k, j, i);
            real r = d.x[IDIR][i];
            real theta = d.x[JDIR][j];

            // auto [vr, vtheta] = v_tv(d.Vc(VX1, k, j, i), d.Vc(VX2, k, j, i), r, theta, tstop);

            d.dustVc[n](RHO, k, j, i) = 1.0 / (r * r); // because one particle in each cell, with uniform theta spacing and log radial spacing => rho propto 1/r^2
            d.dustVc[n](VX1, k, j, i) = 0.0;
            d.dustVc[n](VX2, k, j, i) = 0.0;
            d.dustVc[n](VX3, k, j, i) = d.Vc(VX3, k, j, i);
          }
        }
      }
    }
  }
  // Send it all, if needed
  d.SyncToDevice();
}
