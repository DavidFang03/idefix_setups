#pragma once

#include "../../shared/wind/params.hpp"
#include "floor.hpp"
#include "ionisation.hpp"

using namespace Params;

namespace Wind {
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  /* User-defined boundaries for a 2.5D disk.

  Radial boundary: inspired from test/MHD/AmbipolarWind
  - Inner: solid body
  - Outer: outflow

  Polar boundary: "axis", see (Zhu & Stone 2018)
  */

  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;

#ifndef DISABLE_MHD
  IdefixArray4D<real> Vs = hydro->Vs;
#endif
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  if ((dir == IDIR) && (side == left)) {

    int ighost = data->nghost[IDIR];
    real Rin = 1.0;
    real cscorona = epsilonTopGlob / sqrt(Rin);
    real densityFloor0 = densityFloorGlob;
    real epsilon = epsilonGlob;
    real trSmoothingTemp = trSmoothingTempGlob;
    real epsilonTop = epsilonTopGlob;
    real Hideal = HidealGlob;
    // real csdisk = epsilonGlob / sqrt(Rin);

    hydro->boundary->BoundaryFor(
        "UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          real R = x1(i) * sin(x2(j));
          real z = x1(i) * cos(x2(j));
          real Rmin = FMAX(R, 1.0);

          real r = x1(i);
          real th = x2(j);

          real temp = Wind::temperature(r, th, epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);

          Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);
          real densityFloor = Wind::computeDensityFloor(R, z, densityFloor0, Rin, epsilon);
          if (Vc(RHO, k, j, i) < densityFloor)
            Vc(RHO, k, j, i) = densityFloor;

#ifndef ISOTHERMAL
          Vc(PRS, k, j, i) = Vc(RHO, k, j, i) * temp;
#endif

          if (Vc(VX1, k, j, ighost) >= ZERO_F)
            Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i - 1);
          else
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
          // Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);
          Vc(VX3, k, j, i) = R * pow(Rmin, 0);
#ifndef DISABLE_MHD
          Vc(BX3, k, j, i) = -Vc(BX3, k, j, 2 * ighost - i - 1);
#endif
        });

#ifndef DISABLE_MHD
    hydro->boundary->BoundaryForX2s("UserDefX2s", dir, side, KOKKOS_LAMBDA(int k, int j, int i) { Vs(BX2s, k, j, i) = Vs(BX2s, k, j, ighost); });
#endif
  }

  if ((dir == IDIR) && (side == right)) {
    int ighost = data->end[IDIR] - 1;

    hydro->boundary->BoundaryFor(
        "UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          real R = x1(i) * sin(x2(j));
          real z = x1(i) * cos(x2(j));

          Vc(RHO, k, j, i) = Vc(RHO, k, j, ighost);

#ifndef ISOTHERMAL
          Vc(PRS, k, j, i) = Vc(PRS, k, j, ighost);
#endif

          if (Vc(VX1, k, j, ighost) <= ZERO_F)
            Vc(VX1, k, j, i) = 0.0;
          else
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);

          Vc(VX3, k, j, i) = Vc(VX3, k, j, ighost);

#ifndef DISABLE_MHD
          Vc(BX3, k, j, i) = 0.0; // Jannaud 2026
#endif
        });
#ifndef DISABLE_MHD
    hydro->boundary->BoundaryForX2s("UserDefX2s", dir, side, KOKKOS_LAMBDA(int k, int j, int i) { Vs(BX2s, k, j, i) = Vs(BX2s, k, j, ighost); });
#endif
  }
  if (dir == JDIR) {

    const int j_beg = data->beg[JDIR];
    const int j_end = data->end[JDIR];

    if (side == left) {
      idefix_for(
          "UserDefX2_Left_Vc", 0, data->np_tot[KDIR], 0, j_beg, 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_beg - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

#ifndef ISOTHERMAL
            Vc(PRS, k, j, i) = Vc(PRS, k, jrefl, i);
#endif
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            // Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
            Vc(VX3, k, j, i) = ZERO_F;
#ifndef DISABLE_MHD
            Vc(BX3, k, j, i) = -Vc(BX3, k, jrefl, i); // https://github.com/idefix-code/idefix/issues/203
#endif
          });

#ifndef DISABLE_MHD
      hydro->boundary->BoundaryForX1s(
          "UserDefX1_Left_Vs", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_beg - 1 - j;
            Vs(BX1s, k, j, i) = Vs(BX1s, k, jrefl, i);
          });
#endif
    } else if (side == right) {
      idefix_for(
          "UserDefX2_Right_Vc", 0, data->np_tot[KDIR], j_end, data->np_tot[JDIR], 0, data->np_tot[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_end - 1 - j;
            Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);
#ifndef ISOTHERMAL
            Vc(PRS, k, j, i) = Vc(PRS, k, jrefl, i);
#endif
            Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
            Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
            Vc(VX3, k, j, i) = ZERO_F;
        // Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);

#ifndef DISABLE_MHD
            Vc(BX3, k, j, i) = -Vc(BX3, k, jrefl, i);
#endif
          });
#ifndef DISABLE_MHD
      hydro->boundary->BoundaryForX1s(
          "UserDefX1_Right_Vs", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
            const int jrefl = 2 * j_end - 1 - j;
            Vs(BX1s, k, j, i) = Vs(BX1s, k, jrefl, i);
          });
#endif
    }
  }
}

#ifndef DISABLE_MHD
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
    int jref = data.beg[JDIR];
    idefix_for("EMFBoundary", 0, data.np_tot[KDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int i) { Ex3(k, jref, i) = ZERO_F; });
  }
  if (data.rbound[JDIR] == axis || data.rbound[JDIR] == userdef) {
    int jref = data.end[JDIR];
    idefix_for("EMFBoundary", 0, data.np_tot[KDIR], 0, data.np_tot[IDIR], KOKKOS_LAMBDA(int k, int i) { Ex3(k, jref, i) = ZERO_F; });
  }
}
#endif
} // namespace Wind