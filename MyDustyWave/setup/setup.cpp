#include "setup.hpp"
#include "idefix.hpp"

static real Omega;
static real shear;
static real vs;
static real n;
static real Sigma0;
real PM;
real tau;

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  // GPUS cannot capture static variables
  real OmegaLocal = Omega;
  real shearLocal = shear;

  idefix_for(
      "BodyForce", data.beg[KDIR], data.end[KDIR], data.beg[JDIR], data.end[JDIR], data.beg[IDIR], data.end[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        force(IDIR, k, j, i) = -2.0 * OmegaLocal * shearLocal * x(i);
        force(JDIR, k, j, i) = ZERO_F;
        force(KDIR, k, j, i) = ZERO_F;
      });

  idfx::popRegion();
}

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
  // auto tstop_array = data.particles->pack->fields          // real r = states(PX1, idx);
  // real theta = states(PX2, idx);
  // real phi = states(PX3, idx);

  // int i = floor(iint * (log(r / rstart)) / log(rend / rstart)) + ibeg;
  // int j = floor((theta - theta_beg) / dtheta) + jbeg;

  // int k = floor((phi - phi_beg) / dphi) + kbeg;.GetField<real>("t_stop");
  // auto r_indexes = data.particles->pack->fields.GetField<real>("r_indexes");
  // auto theta_indexes =
  // data.particles->pack->fields.GetField<real>("theta_indexes"); auto
  // phi_indexes = data.particles->pack->fields.GetField<real>("phi_indexes");

  // Below works only for uniform theta, phi and logarithmic r. It finds the
  // cell where the particle is in.
  // TODO: Interpolation?
  DataBlockHost d(data);

  IdefixArray4D<real> Vc = data.hydro->Vc;

  // int i_gbeg = d.gbeg[IDIR];
  // int j_gbeg = d.gbeg[JDIR];
  // int ibeg = d.beg[IDIR];
  // int jbeg = d.beg[JDIR];
  // int iend = d.end[IDIR];
  // int kbeg = d.beg[KDIR];

  // real iint = d.np_int[IDIR];
  // real rstart = d.x[IDIR](ibeg);
  // real rend = d.x[IDIR](iend);

  // real theta_beg = d.x[JDIR](jbeg);
  // real dtheta = d.x[JDIR](jbeg + 1) - d.x[JDIR](jbeg);
  // real phi_beg = d.x[KDIR](kbeg);
  // real dphi = d.x[KDIR](kbeg + 1) - d.x[KDIR](kbeg);

  real tau_local = tau;

  idefix_for(
      "StoppingTime", 0, data.particles->pack->maxActiveIndex + 1, KOKKOS_LAMBDA(int idx) {
        if (isActive(idx)) {

          // real r = states(PX1, idx);
          // real theta = states(PX2, idx);
          // real phi = states(PX3, idx);

          // int i = floor(iint * (log(r / rstart)) / log(rend / rstart)) + ibeg;
          // int j = floor((theta - theta_beg) / dtheta) + jbeg;

          // int k = floor((phi - phi_beg) / dphi) + kbeg;

          // tstop_array(idx) = Vc(RHO, kbeg, j, i);

          tstop(idx) = 0.1;
          // tstop(idx) = Vc(RHO, k, j, i);
        }
      });
}

void ParticleBodyForce(DataBlock &data, const real t, IdefixArray2D<real> &force) {
  idfx::pushRegion("ParticleBodyForce");
  // a separate hook is needed for particles because gravity isn't in general
  // computed at the same time for particles and the fluid.

  // GPUS cannot capture static variables
  real OmegaLocal = Omega;
  real shearLocal = shear;

  auto states = data.particles->pack->states;
  auto isActive = data.particles->pack->isActive;
  idefix_for(
      "BodyForce", 0, data.particles->pack->maxActiveIndex + 1, KOKKOS_LAMBDA(int idx) {
        if (isActive(idx)) {
          force(IDIR, idx) = -2.0 * OmegaLocal * shearLocal * states(PX1, idx);
          force(JDIR, idx) = ZERO_F;
          force(KDIR, idx) = ZERO_F;
        }
      });
  idfx::popRegion();
}

void MyDrag(DataBlock *data, real beta, IdefixArray3D<real> &gamma) {
  // Compute the drag coefficient gamma from the input beta
  auto VcGas = data->hydro->Vc;
  idefix_for("MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) { gamma(k, j, i) = 1 / (beta * VcGas(RHO, k, j, i)); });
}

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {

  // Get rotation rate along vertical axis
  Omega = input.Get<real>("Hydro", "rotation", 0);
  shear = input.Get<real>("Hydro", "shearingBox", 0);
  vs = input.Get<real>("Hydro", "csiso", 1);
  Sigma0 = input.Get<real>("Setup", "Sigma0", 0);
  n = input.Get<real>("Setup", "n", 0);

  PM = input.Get<real>("Particles", "ParticleMass", 0);
  tau = input.Get<real>("Particles", "tau", 0);
  // Add our userstep to the timeintegrator
  data.gravity->EnrollBodyForce(BodyForce);
  data.particles->EnrollStoppingTime(&UserdefStoppingTime);
  data.particles->EnrollBodyForce(ParticleBodyForce);

  if (data.haveDust) {
    int nSpecies = data.dust.size();
    for (int n = 0; n < nSpecies; n++) {
      data.dust[n]->drag->EnrollUserDrag(&MyDrag);
    }
  }
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
  // Create a host copy
  DataBlockHost d(data);
  real x;

  real L = d.x[IDIR](d.end[IDIR] - 1) - d.x[IDIR](d.beg[IDIR]);
  real kx = 2 * M_PI * n / L;
  real kappa = 2 * Omega * (2 * Omega + shear);

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        x = d.x[IDIR](i);

        d.Vc(RHO, k, j, i) = Sigma0 + 1.0e-3 * Sigma0 * cos(kx * x);
        d.Vc(VX1, k, j, i) = 1.0e-3 * sqrt(kappa * kappa + (kx * vs) * (kx * vs)) / kx * cos(kx * x);
        d.Vc(VX2, k, j, i) = shear * x + 1.0e-3 * (shear + 2 * Omega) / kx * sin(kx * x);
        d.Vc(VX3, k, j, i) = 0.0;

        for (int n = 0; n < data.dust.size(); n++) {
          d.dustVc[n](RHO, k, j, i) = 1e-6 + exp(-0.5 * x * x / (0.001 * 0.001));
          d.dustVc[n](VX1, k, j, i) = 0.0;
          d.dustVc[n](VX2, k, j, i) = shear * x;
          d.dustVc[n](VX3, k, j, i) = 0.0;
        }
      }
    }
  }

  for (int n = 0; n < d.PactiveCount; n++) {

    d.Ps(PX1, n) = 0.0;
    d.Ps(PX2, n) = 0.0;
    d.Ps(PX3, n) = 0.0;
    d.Ps(PVX1, n) = 0.0;
    d.Ps(PVX2, n) = 0.0;
    d.Ps(PVX3, n) = 0.0;
    d.Ps(PMASS, n) = PM;
  }

  // Send it all, if needed
  d.SyncToDevice();
}
