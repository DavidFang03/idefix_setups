#include "setup.hpp"
#include "analysis.hpp"
#include "dumpImage.hpp"
#include "idefix.hpp"
#include <cmath>

real epsilonGlob;
real epsilonTopGlob;
real betaGlob;
real HidealGlob;
real AmMidGlob;
real gammaGlob;
real densityFloorGlob;
real trSmoothingGlob;
real trSmoothingTempGlob;
real Rm0;
real etab0;

real rmin;
real rmax;
real sizemin;
real sizemax;
real thetamin;
real thetamax;
real num_r;
real num_theta;
real num_size;

static std::string dat_path;
static std::string reload_path;

static std::string initialvelocity;
Analysis *analysis;

KOKKOS_INLINE_FUNCTION real computeDensityFloor(real R, real z, real d_floor_0, real Rin, real c0) { return d_floor_0; }

KOKKOS_INLINE_FUNCTION real computeVaMax(real t_change, real Va_ini_max, real Va_fin_max, real t) {
  real Va_lim;

  if (t < t_change) {
    Va_lim = t * (Va_fin_max - Va_ini_max) / t_change + Va_ini_max;
  } else {
    Va_lim = Va_fin_max;
  }

  return Va_lim;
}

void Ambipolar(DataBlock &data, real t, IdefixArray3D<real> &xAin) {
  IdefixArray3D<real> xA = xAin;
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  real Hideal = HidealGlob;
  real epsilon = epsilonGlob;
  real AmMid = AmMidGlob;
  real etamax = 10 * epsilon * epsilon; // Corresponds to Rm=0.1
  real Rin = 1.0;
  real waveKillWidth = 0.1;
  real trSmoothing = trSmoothingGlob;

  idefix_for(
      "Ambipolar", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = x1(i) * cos(x2(j));
        real R = FMAX(FABS(x1(i) * sin(x2(j))), ONE_F);
        real Omega = pow(R, -1.5);

        real zh = z / (R * epsilon); // z in units of disc scale height h=R*epsilon
        real Am;

        Am = AmMid / (0.5 * (1 - tanh((fabs(zh) - Hideal) / trSmoothing)));

        real B2 = Vc(BX1, k, j, i) * Vc(BX1, k, j, i) + Vc(BX2, k, j, i) * Vc(BX2, k, j, i) + Vc(BX3, k, j, i) * Vc(BX3, k, j, i);
        real eta = B2 / (Omega * Am * Vc(RHO, k, j, i));
        if (eta > etamax)
          xA(k, j, i) = etamax / B2; //! SAME THING????
        else
          xA(k, j, i) = 1.0 / (Omega * Am * Vc(RHO, k, j, i)); //! SAME THING????

        // Kill it at the radial boundaryloop
        if (x1(i) / Rin < Rin * (1 + waveKillWidth)) {
          real w = (x1(i) - Rin) / (Rin * waveKillWidth);

          xA(k, j, i) = xA(k, j, i) * w;
        }
      });
}

void Resistivity(DataBlock &data, real t, IdefixArray3D<real> &etain) {
  IdefixArray3D<real> eta = etain;
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  real trSmoothing = trSmoothingGlob;
  real Hideal = HidealGlob;
  real epsilon = epsilonGlob;

  real R0 = data.mygrid->xbeg[IDIR]; // =1
  // The constant pre factor for R_m in the dead zone
  real Rm0copy = Rm0;
  real etaBuffer0 = etab0;

  idefix_for(
      "Resistivity", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real z = x1(i) * cos(x2(j));
        real R = x1(i) * sin(x2(j)); // cylindric R
        real Ri = FMAX(R0, R);
        real r = x1(i);
        real zh = z / (R * epsilon); //=1 ??? =z/H
        real Omega = pow(R, -1.5);
        // Inner region damping. Buffer region
        real EtaBuffer = etaBuffer0 * epsilon * epsilon * 0.05 * FMAX((1.25 * R0 - r), 0.0); // # [R0, R0+0.25R0]

        // Transition across disk and corona (want eta to be zero outside the
        // disk dead zone)
        real TransDC = 0.5 * (1 - tanh((fabs(zh) - Hideal) / (trSmoothing)));
        // Transition across the DZI (want eta to be zero outside the disk dead
        // zone)
        // real TransDZI = 0.5 * (1 + tanh((R - 10.0) / (0.1 * trSmoothing)));
        // //! cause eta to diverge->crash
        // // The expression for the magnetic Reynolds number (R_m) in the dead
        // // zone of the disk
        // real RmDZ = RmDZ0 * 1 / (Vc(RHO, k, j, i) * Ri);
        // // The expression for the Ohmic resistivity in the dead zone of the
        // disk real etaDZ = pow(epsilon * Ri, 2) * Omega /
        //              RmDZ; // exact expresion to get eta from Rm.
        // The final expression for the Ohmic resistivity (includes the buffer
        // zone contribution). Makes R_M = 50 at the inner edge of the DZI
        // (feels appropriate - maybe disk slightly heavy? - discuss) eta(k,j,i)
        // = (pow(Ri,1.5))*Vc(RHO,k,j,i)/(10.0*10.0*pow(10,0.5))*TransDC +
        // EtaBuffer;
        // eta(k,j,i) = EtaBuffer +
        // (pow(Ri,1.5))*Vc(RHO,k,j,i)/2500*TransDZI*TransDC; eta(k,j,i) =
        // EtaBuffer
        // + etaDZ;
        // eta(k, j, i) = etaDZ * TransDC * TransDZI + EtaBuffer;
        // eta(k, j, i) = EtaBuffer;
        // real eta0 = pow(epsilon * Ri, 2) * Omega / Rm0copy;
        // eta0 = 0;
        // Precription of Roberts,Latter,Lesur (2026): Rm propto 1/(rho R)
        eta(k, j, i) = epsilon * epsilon * pow(Ri, 1.5) * Vc(RHO, k, j, i) / Rm0copy * TransDC + EtaBuffer;
        // eta(k, j, i) = eta0 * (pow(Ri, 1.5)) * Vc(RHO, k, j, i) /
        //                    (10.0 * 10.0 * pow(10, 0.5)) * TransDC +
        //                EtaBuffer;
      });
}

void MySourceTerm(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  real epsilonTop = epsilonTopGlob;
  real epsilon = epsilonGlob;
  real tauGlob = 0.1;
  real gamma_m1 = gammaGlob - 1.0;
  real dt = dtin;
  real Hideal = HidealGlob;
  real Rin = 1.0;
  real trSmoothingTemp = trSmoothingTempGlob;

  idefix_for(
      "MySourceTerm", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real r = x1(i);
        real th = x2(j);
        real z = r * cos(th);
        real R = r * sin(th);
        real R0 = FMAX(R, Rin);
        real tau;

        real Zh = FABS(z / R0) / epsilon;
        real Tdisk = epsilon * epsilon / R0;
        real Tcorona = epsilonTop * epsilonTop / R0;
        real Teff = 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * tanh((Zh - Hideal) / (trSmoothingTemp));

        tau = tauGlob * (FMIN(pow(R, 1.5), 1.0));

        // Cooling /heatig function
        real Ptarget = Teff * Vc(RHO, k, j, i);

        Uc(ENG, k, j, i) += -dt * (Vc(PRS, k, j, i) - Ptarget) / (tau * gamma_m1);
      });
}

void InternalBoundary(Hydro *hydro, const real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Vs = hydro->Vs;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];

  real vAmax = computeVaMax(4.0, 50.0, 8.0, t);
  real densityFloor0 = densityFloorGlob;
  real Rin = 1.0;
  real epsilon = epsilonGlob;

  idefix_for(
      "InternalBoundary", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real R = x1(i) * sin(x2(j));
        real z = x1(i) * cos(x2(j));
        real zh = FABS(z / R) / epsilon;

        real b2 = EXPAND(Vc(BX1, k, j, i) * Vc(BX1, k, j, i), +Vc(BX2, k, j, i) * Vc(BX2, k, j, i), +Vc(BX3, k, j, i) * Vc(BX3, k, j, i));
        real va2 = b2 / Vc(RHO, k, j, i);
        real myMax = vAmax;
        // if(x1(i)<1.1) myMax=myMax/50.0;
        if (va2 > myMax * myMax) {
          real T = Vc(PRS, k, j, i) / Vc(RHO, k, j, i);
          Vc(RHO, k, j, i) = b2 / (myMax * myMax);
          Vc(PRS, k, j, i) = T * Vc(RHO, k, j, i);
        }
        real densityFloor = computeDensityFloor(R, z, densityFloor0, Rin, epsilon);
        if (Vc(RHO, k, j, i) < densityFloor) {
          real T = Vc(PRS, k, j, i) / Vc(RHO, k, j, i);
          Vc(RHO, k, j, i) = densityFloor;
        }
      });
}
// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  auto *data = hydro->data;
  if ((dir == IDIR) && (side == left)) {
    IdefixArray4D<real> Vc = hydro->Vc;
    IdefixArray4D<real> Vs = hydro->Vs;
    IdefixArray1D<real> x1 = data->x[IDIR];
    IdefixArray1D<real> x2 = data->x[JDIR];

    int ighost = data->nghost[IDIR];
    real Omega = 1.0;
    real Rin = 1.0;
    real csdisk = epsilonGlob / sqrt(Rin);
    real cscorona = epsilonTopGlob / sqrt(Rin);
    real densityFloor0 = densityFloorGlob;
    real epsilon = epsilonGlob;

    hydro->boundary->BoundaryFor(
        "UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          real R = x1(i) * sin(x2(j));
          real z = x1(i) * cos(x2(j));
          /*
          Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
          Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost);*/

          Vc(RHO, k, j, i) = 1.0 / (Rin * sqrt(Rin)) * exp(1.0 / (csdisk * csdisk) * (1.0 / sqrt(Rin * Rin + z * z) - 1.0 / Rin));
          real densityFloor = computeDensityFloor(R, z, densityFloor0, Rin, epsilon);
          if (Vc(RHO, k, j, i) < densityFloor)
            Vc(RHO, k, j, i) = densityFloor;

          Vc(PRS, k, j, i) = Vc(RHO, k, j, i) * csdisk * csdisk;

          if (Vc(VX1, k, j, ighost) >= ZERO_F)
            Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i);
          else
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
          // real Rmin = FMAX(0.3,R);

          // Vc(VX3,k,j,i) = 1.0/sqrt(Rmin) * sqrt( Rmin / sqrt(Rmin*Rmin + z*z));
          Vc(VX3, k, j, i) = Omega * R;
          Vc(BX3, k, j, i) = -Vc(BX3, k, j, 2 * ighost - i);
          // Vc(BX3,k,j,i) = Vc(BX3,k,j,ighost);
        });
    hydro->boundary->BoundaryForX2s("UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) { Vs(BX2s, k, j, i) = Vs(BX2s, k, j, ighost); });
  }

  if ((dir == IDIR) && (side == right)) {
    IdefixArray4D<real> Vc = hydro->Vc;
    IdefixArray4D<real> Vs = hydro->Vs;
    IdefixArray1D<real> x1 = data->x[IDIR];
    IdefixArray1D<real> x2 = data->x[JDIR];

    int ighost = data->end[IDIR] - 1;
    real Rin = 1.0;
    real csdisk = epsilonGlob / sqrt(Rin);
    real cscorona = epsilonTopGlob / sqrt(Rin);

    hydro->boundary->BoundaryFor(
        "UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          real R = x1(i) * sin(x2(j));
          real z = x1(i) * cos(x2(j));

          Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
          Vc(PRS, k, j, i) = Vc(PRS, k, j, ighost);

          if (Vc(VX1, k, j, ighost) <= ZERO_F)
            Vc(VX1, k, j, i) = 0.0;
          else
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
          // real Rmin = FMAX(0.3,R);

          // Vc(VX3,k,j,i) = 1.0/sqrt(Rmin) * sqrt( Rmin / sqrt(Rmin*Rmin + z*z));
          Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
          Vc(BX3, k, j, i) = -Vc(BX3, k, j, 2 * ighost - i);
          // Vc(BX3,k,j,i) = Vc(BX3,k,j,ighost);
        });
    hydro->boundary->BoundaryForX2s("UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) { Vs(BX2s, k, j, i) = Vs(BX2s, k, j, ighost); });
  }
  if (dir == JDIR) {
    IdefixArray4D<real> Vc = hydro->Vc;
    IdefixArray4D<real> Vs = hydro->Vs;

    const int j_beg = data->beg[JDIR];
    const int j_end = data->end[JDIR];

    if (side == left) {
      // Cell-centered Loop
      idefix_for(
          "UserDefX2_Left_Vc", 0, data->np_tot[KDIR], 0, j_beg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_beg - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
            Vc(BX3, k, j, i) = -Vc(BX3, k, jrefl, i); // https://github.com/idefix-code/idefix/issues/203
          });

      hydro->boundary->BoundaryForX1s(
          "UserDefX1_Left_Vs", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_beg - 1 - j;
            Vs(BX1s, k, j, i) = Vs(BX1s, k, jrefl, i);
          });
    } else if (side == right) {
      idefix_for(
          "UserDefX2_Right_Vc", 0, data->np_tot[KDIR], j_end, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_end - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
            Vc(BX3, k, j, i) = 0.0;
          });

      hydro->boundary->BoundaryForX1s(
          "UserDefX1_Right_Vs", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_end - 1 - j;
            Vs(BX1s, k, j, i) = Vs(BX1s, k, jrefl, i);
          });
    }
  }
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

void EmfBoundary(DataBlock &data, const real t) {
  IdefixArray3D<real> Ex1 = data.hydro->emf->ex;
  IdefixArray3D<real> Ex2 = data.hydro->emf->ey;
  IdefixArray3D<real> Ex3 = data.hydro->emf->ez;
  if (data.lbound[IDIR] == userdef) {

    int ighost = data.beg[IDIR];

    idefix_for("EMFBoundary", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], KOKKOS_LAMBDA(int k, int j) { Ex3(k, j, ighost) = ZERO_F; });
  }
  // additional zero EMF on the boundary
  if (data.lbound[JDIR] == axis || data.lbound[JDIR] == userdef) {
    int jghost = data.beg[JDIR];
    idefix_for("EMFBoundary", 0, data.np_tot[KDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int i) { Ex3(k, jghost, i) = ZERO_F; });
  }
  if (data.rbound[JDIR] == axis || data.rbound[JDIR] == userdef) {
    int jghost = data.end[JDIR];
    idefix_for("EMFBoundary", 0, data.np_tot[KDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int i) { Ex3(k, jghost, i) = ZERO_F; });
  }
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
  Resistivity(data, data.t, scrh_eta);
  Ambipolar(data, data.t, scrh);

  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  // Make references to the user-defined arrays (variables is a container of
  // IdefixHostArray3D) Note that the labels should match the variable names in
  // the input file
  IdefixHostArray3D<real> eta = variables["eta"];
  IdefixHostArray3D<real> Am = variables["Am"];
  IdefixHostArray3D<real> InvDt = variables["InvDt"];

  IdefixHostArray1D<real> x1 = d.x[IDIR];
  IdefixHostArray1D<real> x2 = d.x[JDIR];
  IdefixHostArray4D<real> Vc = d.Vc;
  IdefixArray3D<real>::HostMirror scrhHost = Kokkos::create_mirror_view(scrh);
  Kokkos::deep_copy(scrhHost, scrh);
  IdefixArray3D<real>::HostMirror scrhHost_eta = Kokkos::create_mirror_view(scrh_eta);
  Kokkos::deep_copy(scrhHost_eta, scrh_eta);

  for (int k = d.beg[KDIR]; k < d.end[KDIR]; k++) {
    for (int j = d.beg[JDIR]; j < d.end[JDIR]; j++) {
      for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++) {
        real z = x1(i) * cos(x2(j));
        real R = FMAX(FABS(x1(i) * sin(x2(j))), ONE_F);
        real Omega = pow(R, -1.5);
        eta(k, j, i) = scrhHost_eta(k, j, i);
        Am(k, j, i) = 1.0 / (Omega * scrhHost(k, j, i) * Vc(RHO, k, j, i));
        InvDt(k, j, i) = d.InvDt(k, j, i);
      }
    }
  }
}

void AnalysisFunction(DataBlock &data) { analysis->PerformAnalysis(data); }

// Default constructor

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.hydro->EnrollAmbipolarDiffusivity(&Ambipolar);
  data.hydro->EnrollOhmicDiffusivity(&Resistivity);
  data.hydro->EnrollUserSourceTerm(&MySourceTerm);
  data.hydro->EnrollInternalBoundary(&InternalBoundary);
  data.hydro->EnrollEmfBoundary(&EmfBoundary);
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
  epsilonGlob = input.Get<real>("Setup", "epsilon", 0);
  epsilonTopGlob = input.Get<real>("Setup", "epsilonTop", 0);
  betaGlob = input.Get<real>("Setup", "beta", 0);
  HidealGlob = input.Get<real>("Setup", "Hideal", 0);
  AmMidGlob = input.Get<real>("Setup", "Am", 0);
  densityFloorGlob = input.Get<real>("Setup", "densityFloor", 0);
  trSmoothingGlob = input.Get<real>("Setup", "transitionSmoothing", 0);
  Rm0 = input.Get<real>("Setup", "Rm0", 0);
  etab0 = input.Get<real>("Setup", "etab0", 0);

  reload_path = input.Get<std::string>("Setup", "reload_path", 0);
  initialvelocity = input.Get<std::string>("Particles", "initialvelocity", 0);

  num_r = input.Get<real>("Particles", "num_r", 0);
  num_theta = input.Get<real>("Particles", "num_theta", 0);
  num_size = input.Get<real>("Particles", "num_size", 0);
  rmin = input.Get<real>("Particles", "rmin", 0);
  rmax = input.Get<real>("Particles", "rmax", 0);
  thetamin = input.Get<real>("Particles", "thetamin", 0);
  thetamax = input.Get<real>("Particles", "thetamax", 0);
  sizemin = input.Get<real>("Particles", "sizemin", 0);
  sizemax = input.Get<real>("Particles", "sizemax", 0);

  dat_path = input.Get<std::string>("Output", "dat_path", 0);
  analysis = new Analysis(input, grid, data, output, dat_path);
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
