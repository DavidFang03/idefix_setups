#pragma once

#include "../../shared/wind/params.hpp"
#include "floor.hpp"
#include "ic.hpp"
#include "ionisation.hpp"

using namespace Params;

namespace Wind {
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  /* User-defined boundaries for a 2.5D disk.

  Radial boundary: inspired from test/MHD/AmbipolarWind
  - Inner: solid body rotation + outflow
  - Outer: outflow

  Polar boundary: "axis", (Zhu & Stone 2018)
  */

  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;

#ifndef DISABLE_MHD
  IdefixArray4D<real> Vs = hydro->Vs;
#endif
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];
  if ((dir == IDIR) && (side == left)) {

    int ighost = data->nghost[IDIR];
    real Rin = RinGlob;
    real epsilon = epsilonGlob;
    real trSmoothingTemp = trSmoothingTempGlob;
    real epsilonTop = epsilonTopGlob;
    real Hideal = HidealGlob;
    real densityFloor0 = densityFloorGlob;

    hydro->boundary->BoundaryFor(
        "UserDefX1", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          real rho = Wind::density_initial(x1(i), x2(j), x3(k), epsilon, Rin, densityFloor0);
          Vc(RHO, k, j, i) = rho;

#ifndef ISOTHERMAL
          real temp = Wind::temperature(x1(i), x2(j), x3(k), epsilon, epsilonTop, Rin, Hideal, trSmoothingTemp);
          Vc(PRS, k, j, i) = rho * temp;
#endif

          if (Vc(VX1, k, j, ighost) >= ZERO_F)
            Vc(VX1, k, j, i) = -Vc(VX1, k, j, 2 * ighost - i - 1);
          else
            Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);

          // Vc(VX1, k, j, i) = Vc(VX1, k, j, ighost);
          Vc(VX2, k, j, i) = Vc(VX2, k, j, ighost);
          Vc(VX3, k, j, i) = Wind::vphi_initial(x1(i), x2(j), x3(k), epsilon, Rin);

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
          Vc(BX3, k, j, i) = 0.0; // Jannaud 2026 important
#endif
        });
#ifndef DISABLE_MHD
    hydro->boundary->BoundaryForX2s("UserDefX2s", dir, side, KOKKOS_LAMBDA(int k, int j, int i) { Vs(BX2s, k, j, i) = Vs(BX2s, k, j, ighost); });
#endif
  }

  if (dir == JDIR) {
    int j_ghost;
    if (side == left) {
      j_ghost = data->beg[JDIR];
    } else if (side == right) {
      j_ghost = data->beg[JDIR];
    }
    const int jrefl = 2 * j_ghost - 1;

    hydro->boundary->BoundaryFor(
        "UserDefX2", dir, side, KOKKOS_LAMBDA(int k, int j, int i) {
          Vc(RHO, k, j, i) = Vc(RHO, k, jrefl, i);

#ifndef ISOTHERMAL
          Vc(PRS, k, j, i) = Vc(PRS, k, jrefl, i);
#endif
          Vc(VX1, k, j, i) = Vc(VX1, k, jrefl, i);
          Vc(VX2, k, j, i) = -Vc(VX2, k, jrefl, i);
          Vc(VX3, k, j, i) = -Vc(VX3, k, jrefl, i);
#ifndef DISABLE_MHD
          Vc(BX3, k, j, i) = -Vc(BX3, k, jrefl, i); // https://github.com/idefix-code/idefix/issues/203
#endif
        });

#ifndef DISABLE_MHD
    hydro->boundary->BoundaryForX1s("UserDefX1s", dir, side, KOKKOS_LAMBDA(int k, int j, int i) { Vs(BX1s, k, j, i) = Vs(BX1s, k, jrefl, i); });
#endif
  };
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