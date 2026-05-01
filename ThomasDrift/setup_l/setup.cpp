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

real dtg;
real PM;

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  IdefixArray1D<real> x1 = data.x[IDIR];
  real h0 = h0Glob;
  real CsSlope = CsSlopeGlob;
  idefix_for(
      "MySoundSpeed", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = x1(i);
        cs(k, j, i) = h0 * pow(R, CsSlope);
      });
}

// drag coefficient assuming vertical hydrostatic equilirbium
// And using the fact that Vc(RHO) is the surface density Sigma

// Here, we start from the definition of the stopping time (see doc):
// gamma_i = 1 / (tau_s * Sigma) = Omega / (St * Sigma)
// by definition of the Stokes number (assuming Epstein)
// St = Omega * taus = Omega * rho_s * a / (rho_mid * c_s)
// Using the vertical equilibrium: rho_mid = Sigma/(sqrt(2pi)*h)
// St = sqrt(2*pi) * rho_s a / Sigma
// Hence the product St*Sigma is a constant for each specie.
// We are free to define beta = St*Sigma/sigma0 so that gamma_i = Omega/(beta_i*sigma0)

// In this setup, we assume Sigma = sigma0 @ R=1, so that St(R=1) = beta
// Note that in this setup, sigma0 is entirely scale-free because there is no
// gravitational interraction due to the gas.

// Assuming Epstein drag and a disk with physical surface density Sigma_phys,
// the particle size is related to beta through:
// a = 20 cm *  beta * (Sigma_phys/100 g.cm^-2) / (rho_s / 2 g.cm^-3)
// NB: checked against A. Johansen+ (2007)

// void MyDrag(DataBlock *data, real beta, IdefixArray3D<real> &gamma) {
//   // Compute the drag coefficient gamma from the input beta
//   auto x1 = data->x[IDIR];
//   real sigma0 = sigma0Glob;
//   real sigmaSlope = sigmaSlopeGlob;
//   //DataBlockHost d(data);

//   idefix_for("MyDrag",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
//     KOKKOS_LAMBDA (int k, int j, int i) {
//       real Omega = pow(x1(i),-1.5);
//       real h0 = h0Glob;
//       //gamma(k,j,i)=pow(x1(i),-CsSlopeGlob/2.0)/sigma0/beta;
//       //gamma(k,j,i)=0.0;
//       //gamma(k,j,i) = 1.0/(beta*sigma0*sqrt(x1(i)));
//       //gamma(k,j,i) = 1/(beta*sigma0*pow(x1(i),-sigmaSlope-1)*exp(R*R/hg2*(1.0/sqrt(1.0+z*z/R/R)-1.0)));
//       gamma(k,j,i) = 1/(beta*data->hydro->Vc(RHO,k,j,i));
//       //gamma(k,j,i) = 1.0 / (beta*sigma0*pow(x1(i),-sigmaSlope-1));
//       //gamma(k,j,i) = h0*0.0001 / (sigma0*pow(x1(i),-sigmaSlope-1)*beta);
//     });
// }

// void MyDrag(DataBlock *data, int nSpecie, real beta, IdefixArray3D<real> &gamma) {
//  Compute the drag coefficient gamma from the input beta
//  auto VcGas = data->hydro->Vc;
//  idefix_for("MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) { gamma(k, j, i) = 1 / (beta * VcGas(RHO, k, j, i)); });
//}

// void MyDrag(DataBlock *data, int nSpecie, real beta, IdefixArray3D<real> &gamma) {
//   // Compute the drag coefficient gamma from the input beta
//   auto x1 = data->x[IDIR];
//   real sigma0 = sigma0Glob;
//   real CsSlope = CsSlopeGlob;
//   IdefixArray4D<real> Vc = hydro->Vc;

//   idefix_for("MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) { gamma(k, j, i) = pow(x1(i), CsSlope) / sigma0 / beta; });
// }

// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  auto *data = hydro->data;
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
            real z = x3(k);
            real Vk = 1.0 / sqrt(R);

            Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
            Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i - 1);
            Vc(VX2, k, j, i) = Vk;
            Vc(VX3, k, j, i) = Vc(VX3, k, j, 2 * ighost - i - 1);
          });
    } else if (side == right) {
      ighost = data->end[IDIR] - 1;
      ibeg = data->end[IDIR];
      iend = data->np_tot[IDIR];
      idefix_for(
          "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
            real R = x1(i);
            real z = x3(k);
            real Vk = 1.0 / sqrt(R);

            Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            Vc(VX2, k, j, i) = Vk;
            Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
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

  if (data.particles->pack->activeCount < 0)
    data.particles->pack->activeCount = 0;

  idfx::popRegion();
}

void UserdefStoppingTime(DataBlock &data, const real t, IdefixArray1D<real> &tstop) {
  // a separate hook is needed for particles because gravity isn't in general
  // computed at the same time for particles and the fluid.

  // GPUS cannot capture static variables
  // auto states = data.particles->pack->states;
  auto isActive = data.particles->pack->isActive;
  // DataBlockHost d(data);
  auto tstop_array = data.particles->pack->fields.GetField<real>("t_stop");

  idefix_for(
      "StoppingTime", 0, data.particles->pack->maxActiveIndex + 1, KOKKOS_LAMBDA(int idx) {
        if (isActive(idx)) {
          tstop(idx) = tstop_array(idx);
        }
      });
}

// void UserdefBoundaryDust(Fluid<DustPhysics> *dust, int dir, BoundarySide side, real t) {
//   IdefixArray4D<real> Vc = dust->Vc;
//   auto data = dust->data;
//   IdefixArray1D<real> x1 = data->x[IDIR];
//   IdefixArray1D<real> x3 = data->x[KDIR];
//   if (dir == IDIR) {
//     int ighost, ibeg, iend;
//     if (side == left) {
//       ighost = data->beg[IDIR];
//       ibeg = 0;
//       iend = data->beg[IDIR];
//       idefix_for(
//           "UserDefBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
//             real R = x1(i);
//             real z = x3(k);
//             real Vk = 1.0 / sqrt(R);

//             Vc(RHO, k, j, i) = Vc(RHO, k, j, 2 * ighost - i + 1);
//             if (Vc(VX1, k, j, i) >= ZERO_F) {
//               Vc(VX1, k, j, ighost) = 0.0;
//               // Vc(VX1,k,j,2*ighost - i +1)=0.0;
//             } else {
//               // Vc(VX1,k,j,i) = - Vc(VX1,k,j,2*ighost - i +1);
//               Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
//             }
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
//             real z = x3(k);
//             real Vk = 1.0 / sqrt(R);

//             Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
//             Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
//             // if(Vc(VX1,k,j,ighost)<=ZERO_F){
//             //   Vc(VX1,k,j,i) = 0.0;
//             // }
//             // if(Vc(VX1,k,j,i)<=ZERO_F){
//             //   Vc(VX1,k,j,ighost) = 0.0;
//             // }
//             Vc(VX2, k, j, i) = Vk;
//             Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
//           });
//     }
//   }
// }

// void userdefstoppingtime(DataBlock &data, const real t, IdefixArray1D<real> &tstop) {
//   // a separate hook is needed for particles because gravity isn't in general
//   // computed at the same time for particles and the fluid.

//   // GPUS cannot capture static variables
//   auto states = data.particles->pack->states;
//   auto isActive = data.particles->pack->isActive;
//   idefix_for(
//       "StoppingTime", 0, data.particles->pack->maxActiveIndex + 1, KOKKOS_LAMBDA(int idx) {
//         if (isActive(idx)) {
//           tstop(idx) = states(PX1, idx) * states(PX1, idx);
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
  data.particles->EnrollStoppingTime(&UserdefStoppingTime);

  sigmaSlopeGlob = input.Get<real>("Setup", "sigmaSlope", 0);
  sigma0Glob = input.Get<real>("Setup", "sigma0", 0);
  CsSlopeGlob = input.Get<real>("Setup", "CsSlope", 0);
  h0Glob = input.Get<real>("Setup", "h0", 0);

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

  real h0 = h0Glob;
  real sigmaSlope = sigmaSlopeGlob;
  // real sigma0 = sigma0Glob;
  // real PM = sigma0 * dtg / d.activeCount;
  real sigma0 = PM / dtg * d.PactiveCount; // Not correct but not important

  real CsSlope = CsSlopeGlob;

  auto size = d.Pfields.GetField<real>("size");
  auto tstop_array = d.Pfields.GetField<real>("t_stop");

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real R = d.x[IDIR](i);
        real z = d.x[KDIR](k);
        real Vk = 1.0 / sqrt(R);
        real cs2 = h0 * h0 * pow(R, 2 * CsSlope);
        real Omega = pow(R, -1.5);
        real hg2 = cs2 / Omega / Omega;

        d.Vc(RHO, k, j, i) = sigma0 * pow(R, sigmaSlope - 1.0) * exp(R * R / hg2 * (1.0 / sqrt(1.0 + z * z / R / R) - 1.0));
        d.Vc(VX1, k, j, i) = 0.0;
        d.Vc(VX3, k, j, i) = 0.0;
        d.Vc(VX2, k, j, i) = Vk * (1 + 0.5 * (hg2 / R / R) * (sigmaSlope - 1.0 + 2 * CsSlope + 2 * CsSlope * z * z / 2.0 / hg2));
      }
    }
  };

  for (int n = 0; n < d.PactiveCount; n++) {
    // d.dustVc[n](RHO, k, j, i) = (1e-5 + 3e-3 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.1 / 0.1)) * d.Vc(RHO, k, j, i); //
    real r = 2.0;
    d.Ps(PX1, n) = r;
    d.Ps(PX2, n) = 0.0;
    d.Ps(PVX1, n) = 0.0;
    real Vk0 = 1 / sqrt(r);
    d.Ps(PVX2, n) = Vk0;
    d.Ps(PVX3, n) = 0.0;
    d.Ps(PMASS, n) = PM;
    size(n) = pow(10.0, -n);
  }
  printf("%f", tstop_array(0));
  printf("%f", tstop_array(1));
  printf("%f", tstop_array(2));

  // Send it all, if needed
  d.SyncToDevice();
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock &data) {}