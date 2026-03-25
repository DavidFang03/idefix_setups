#include "setup.hpp"
#include "analysis.hpp"
#include "idefix.hpp"

static real gammaIdeal;
static real omega;
static real shear;
static real v0;
static real B0;
static std::string datdir;
static real n;
static real vx_r;
static real vx_i;
static real vy_r;
static real vy_i;
static real Bx_r;
static real Bx_i;
static real By_r;
static real By_i;

Analysis *analysis;

/* This file has been adjusted to allow for a new paramter k0 whose value is set
 * in the setup section of the idefix.ini file*/

// #define STRATIFIED

/*********************************************/
/**
 Customized random number generator
 Allow one to have consistant random numbers
 generators on different architectures.
 **/
/*********************************************/
real randm(void) {
  const int a = 16807;
  const int m = 2147483647;
  static int in0 = 13763 + 2417 * idfx::prank;
  int q;

  /* find random number  */
  q = (int)fmod((double)a * in0, m);
  in0 = q;

  return ((real)((double)q / (double)m));
}

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  // GPUS cannot capture static variables
  real omegaLocal = omega;
  real shearLocal = shear;

  idefix_for(
      "BodyForce", data.beg[KDIR], data.end[KDIR], data.beg[JDIR], data.end[JDIR], data.beg[IDIR],
      data.end[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        force(IDIR, k, j, i) = -2.0 * omegaLocal * shearLocal * x(i);
        force(JDIR, k, j, i) = ZERO_F;
#ifdef STRATIFIED
        force(KDIR, k, j, i) = -omegaLocal * omegaLocal * z(k);
#else
        force(KDIR, k, j, i) = ZERO_F;
#endif
      });

  idfx::popRegion();
}

void AnalysisFunction(DataBlock &data) { analysis->PerformAnalysis(data); }

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  gammaIdeal = data.hydro->eos->GetGamma();

  // Get rotation rate along vertical axis
  omega = input.Get<real>("Hydro", "rotation", 0);
  shear = input.Get<real>("Hydro", "shearingBox", 0);
  v0 = input.Get<real>("Setup", "v0", 0);
  B0 = input.Get<real>("Setup", "B0", 0);
  datdir = input.Get<std::string>("Output", "dat_dir", 0);
  n = input.Get<real>("Setup", "n", 0);

  vx_r = input.Get<real>("Setup", "vx-r", 0);
  vx_i = input.Get<real>("Setup", "vx-i", 0);
  vy_r = input.Get<real>("Setup", "vy-r", 0);
  vy_i = input.Get<real>("Setup", "vy-i", 0);
  Bx_r = input.Get<real>("Setup", "Bx-r", 0);
  Bx_i = input.Get<real>("Setup", "Bx-i", 0);
  By_r = input.Get<real>("Setup", "By-r", 0);
  By_i = input.Get<real>("Setup", "By-i", 0);

  // Add our userstep to the timeintegrator
  data.gravity->EnrollBodyForce(BodyForce);

  analysis = new Analysis(input, grid, data, output, datdir);
  output.EnrollAnalysis(&AnalysisFunction);
  // Reset analysis if required
  if (!input.restartRequested) {
    analysis->ResetAnalysis();
  }
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
  // Create a host copy
  DataBlockHost d(data);
  real x, y, z;

  real cs2 = omega * omega;
  real kz = n * 3.1415 / 1;

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        x = d.x[IDIR](i);
        y = d.x[JDIR](j);
        z = d.x[KDIR](k);

#ifdef STRATIFIED
        d.Vc(RHO, k, j, i) = 1.0 * exp(-z * z / (2.0));
#else
        d.Vc(RHO, k, j, i) = 1.0;
#endif
        // d.Vc(PRS, k, j, i) = d.Vc(RHO, k, j, i) / cs2 * gammaIdeal;
        // d.Vc(VX1, k, j, i) = v0 * (0.5 - idfx::randm());
        // d.Vc(VX2, k, j, i) = shear * x + v0 * (0.5 - idfx::randm());
        // d.Vc(VX3, k, j, i) = v0 * (0.5 - idfx::randm());
        d.Vc(VX1, k, j, i) = v0 * (vx_r * cos(kz * z) - vx_i * sin(kz * z));
        d.Vc(VX2, k, j, i) = shear * x + v0 * (vy_r * cos(kz * z) - vy_i * sin(kz * z));
        d.Vc(VX3, k, j, i) = 0;
        d.Vs(BX1s, k, j, i) = v0 * (Bx_r * cos(kz * z) - Bx_i * sin(kz * z));
        d.Vs(BX2s, k, j, i) = v0 * (By_r * cos(kz * z) - By_i * sin(kz * z));
        d.Vs(BX3s, k, j, i) = B0;
      }
    }
  }

  // Send it all, if needed
  d.SyncToDevice();
}

// Analyse data to produce an output

void MakeAnalysis(DataBlock &data) {}
