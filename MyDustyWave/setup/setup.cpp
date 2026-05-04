#include "setup.hpp"
#include "idefix.hpp"

static real Omega;
static real shear;
static real vs;
static real n;
static real Sigma0;

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

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {

  // Get rotation rate along vertical axis
  Omega = input.Get<real>("Hydro", "rotation", 0);
  shear = input.Get<real>("Hydro", "shearingBox", 0);
  vs = input.Get<real>("Hydro", "csiso", 1);
  Sigma0 = input.Get<real>("Setup", "Sigma0", 0);
  n = input.Get<real>("Setup", "n", 0);

  // Add our userstep to the timeintegrator
  data.gravity->EnrollBodyForce(BodyForce);
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
      }
    }
  }

  // Send it all, if needed
  d.SyncToDevice();
}
