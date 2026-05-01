#include "setup.hpp"
#include "idefix.hpp"
#include <algorithm>
#include <vector>

real sigmaSlopeGlob;
real CsSlopeGlob;
real sigma0Glob;
real h0Glob;
real HidealGlob;
real gammaGlob;
real densityFloorGlob;
real alphaGlob;

std::vector<real> betas;

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

// void MyDrag(DataBlock *data, int nSpecie, real beta, IdefixArray3D<real> &gamma) {
//   // Compute the drag coefficient gamma from the input beta
//   auto x1 = data->x[IDIR];
//   real sigma0 = sigma0Glob;
//   real CsSlope = CsSlopeGlob;

//   idefix_for("MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) { gamma(k, j, i) = pow(x1(i), CsSlope) / sigma0 / beta; });
// }

void MyDrag(DataBlock *data, int nSpecie, real beta, IdefixArray3D<real> &gamma) {
  // Compute the drag coefficient gamma from the input beta
  auto VcGas = data->hydro->Vc;
  // auto VcDust = data->dust[nSpecie]->Vc;
  // auto cs = data->hydro->cs;
  IdefixArray1D<real> x1 = data->x[IDIR];

  idefix_for(
      "MyDrag", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        // gamma(k, j, i) = 1 / (beta * VcGas(RHO, k, j, i));
        // real cs = 1.0;
        // real cs = sqrt(VcGas(PRS, k, j, i));
        // real cs = sqrt(VcGas(PRS, k, j, i) / VcGas(RHO, k, j, i));
        real R = x1(i);
        real h0 = h0Glob;
        real CsSlope = CsSlopeGlob;
        real cs = h0 * pow(R, CsSlope);

        // Epstein: beta = s*rhopart
        // real tau = beta * VcDust(RHO, k, j, i) / (cs(k, j, i) * VcGas(RHO, k, j, i));
        gamma(k, j, i) = cs / beta;
      });
}

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

void UserdefBoundaryDust(Fluid<DustPhysics> *dust, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = dust->Vc;
  auto data = dust->data;
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

            Vc(RHO, k, j, i) = Vc(RHO, k, j, 2 * ighost - i - 1);
            if (Vc(VX1, k, j, ighost) >= ZERO_F) {
              Vc(VX1, k, j, i) = 0.0;
            } else {
              // Vc(VX1,k,j,i) = - Vc(VX1,k,j,2*ighost - i +1);
              Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
            }
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
            if (Vc(VX1, k, j, ighost) <= ZERO_F) {
              Vc(VX1, k, j, i) = 0.0;
            }
            // if(Vc(VX1,k,j,i)<=ZERO_F){
            //   Vc(VX1,k,j,ighost) = 0.0;
            // }
            Vc(VX2, k, j, i) = Vk;
            Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
          });
    }
  }
}

// void FargoVelocity(DataBlock &data, IdefixArray2D<real> &Vphi) {
//   IdefixArray1D<real> x1 = data.x[IDIR];

//  idefix_for("FargoVphi",0,data.np_tot[KDIR], 0, data.np_tot[IDIR],
//      KOKKOS_LAMBDA (int k, int i) {
//      Vphi(k,i) = 1.0/sqrt(x1(i));
//  });
//}

void ComputeUserVars(DataBlock &data, UserDefVariablesContainer &variables) {

  // Use Invdt as scratch array
  // IdefixArray3D<real> scrh("Scratch", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

  // Ask for a computation of xA ambipolar in this scratch array
  // MySoundSpeed(data, data.t, scrh_st);
  IdefixArray4D<real> scrh_st("Scratch_st", data.dust.size(), data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
  auto Vc = data.hydro->Vc;
  auto x1 = data.x[IDIR];
  for (int n = 0; n < data.dust.size(); n++) {
    IdefixArray3D<real> scrh_gamma1("Scratch_gamma1", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    MyDrag(&data, n, betas[n], scrh_gamma1);
    idefix_for(
        "StokesUserVar", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR],
        KOKKOS_LAMBDA(int k, int j, int i) { scrh_st(n, k, j, i) = pow(x1(i), -1.5) * Vc(RHO, k, j, i) / scrh_gamma1(k, j, i); });
  }

  DataBlockHost d(data);
  d.SyncFromDevice();

  // Mirror data on Host

  // Sync it

  // Make references to the user-defined arrays (variables is a container of
  // IdefixHostArray3D) Note that the labels should match the variable names in
  // the input file
  IdefixHostArray3D<real> st0 = variables["st0"];
  IdefixHostArray3D<real> st1 = variables["st1"];
  IdefixHostArray3D<real> st2 = variables["st2"];
  // IdefixHostArray4D<real> Vc = d.Vc;

  IdefixArray4D<real>::HostMirror scrhHost_st = Kokkos::create_mirror_view(scrh_st);
  Kokkos::deep_copy(scrhHost_st, scrh_st);

  // for (int n = 0; n < data.dust.size(); n++) {

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        st0(k, j, i) = scrhHost_st(0, k, j, i);
        st1(k, j, i) = scrhHost_st(1, k, j, i);
        st2(k, j, i) = scrhHost_st(2, k, j, i);
      }
    }
  }
  // }
}

// Default constructor
// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) // : m_planet(0)//, Planet &planet)
{
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  if (data.haveDust) {
    int nSpecies = data.dust.size();
    for (int n = 0; n < nSpecies; n++) {
      data.dust[n]->EnrollUserDefBoundary(&UserdefBoundaryDust);
      data.dust[n]->drag->EnrollUserDrag(&MyDrag);
    }
  }

  data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
  // output.EnrollUserDefVariables(&ComputeUserVars);

  // if(data.haveFargo)
  //   data.fargo->EnrollVelocity(&FargoVelocity);
  sigmaSlopeGlob = input.Get<real>("Setup", "sigmaSlope", 0);
  sigma0Glob = input.Get<real>("Setup", "sigma0", 0);
  CsSlopeGlob = input.Get<real>("Setup", "CsSlope", 0);
  h0Glob = input.Get<real>("Setup", "h0", 0);

  int nSpecies = input.Get<int>("Dust", "nSpecies", 0);
  // IdefixHostArray1D<real> betas("betas", nSpecies);
  for (int i = 0; i < nSpecies; i++) {
    betas.push_back(input.Get<real>("Dust", "drag", i + 1));
  }
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
        real Vk = 1.0 / sqrt(R);
        real cs2 = h0 * h0 * pow(R, 2 * CsSlope);
        real Omega = pow(R, -1.5);
        real hg2 = cs2 / Omega / Omega;

        d.Vc(RHO, k, j, i) = 1e-6 + sigma0 * pow(R, sigmaSlope - 1.0) * exp(R * R / hg2 * (1.0 / sqrt(1.0 + z * z / R / R) - 1.0));
        d.Vc(VX1, k, j, i) = 0.0; //+sin(8*d.x[JDIR](j));
        d.Vc(VX3, k, j, i) = 0.0;
        d.Vc(VX2, k, j, i) = Vk * (1 + 0.5 * (hg2 / R / R) * (sigmaSlope - 1.0 + 2 * CsSlope + 2 * CsSlope * z * z / 2.0 / hg2));

        for (int n = 0; n < data.dust.size(); n++) {
          // d.dustVc[n](RHO, k, j, i) = 0.01 * sigma0;
          d.dustVc[n](RHO, k, j, i) = (1e-5 + 3e-3 * exp(-0.5 * (R - 2.0) * (R - 2.0) / 0.1 / 0.1)) * d.Vc(RHO, k, j, i);
          d.dustVc[n](VX1, k, j, i) = 0.0;
          d.dustVc[n](VX2, k, j, i) = Vk;
          d.dustVc[n](VX3, k, j, i) = 0.0;
        }
      }
    }
  }

  // Send it all, if needed
  d.SyncToDevice();
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock &data) {}