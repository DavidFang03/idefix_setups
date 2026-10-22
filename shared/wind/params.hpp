#pragma once

namespace Params {
inline real alphaGlob;
inline real gammaGlob;
inline real tauGlob;
inline real epsilonGlob;
inline real epsilonTopGlob;
inline real betaGlob;
inline real HidealGlob;
inline real AmMidGlob;
inline real densityFloorGlob;
inline real trSmoothingGlob;
inline real trSmoothingTempGlob;
inline real Rm0;
inline real etab0;

KOKKOS_INLINE_FUNCTION real get_r(real x1, real x2, real x3) {
#if GEOMETRY == SPHERICAL
  real r = x1;
#elif GEOMETRY == CYLINDRICAL
  real r = sqrt(x1 * x1 + x2 * x2);
#elif GEOMETRY == POLAR
  real r = sqrt(x1 * x1 + x3 * x3);
#endif
  return r;
}

KOKKOS_INLINE_FUNCTION real get_R(real x1, real x2, real x3) {
#if GEOMETRY == SPHERICAL
  real R = FABS(x1 * sin(x2));
#elif GEOMETRY == CYLINDRICAL
  real R = x1;
#elif GEOMETRY == POLAR
  real R = x1;
#endif
  return R;
}

KOKKOS_INLINE_FUNCTION real get_z(real x1, real x2, real x3) {
#if GEOMETRY == SPHERICAL
  real z = x1 * cos(x2);
#elif GEOMETRY == CYLINDRICAL
  real z = x2;
#elif GEOMETRY == POLAR
  real z = x3;
#endif
  return z;
}

#if GEOMETRY == SPHERICAL
int HOR = IDIR;
int VER = JDIR;
int PHIDIR = KDIR;
int VHOR = VX1;
int VVER = VX2;
int VPHI = VX3;
#elif GEOMETRY == POLAR
int HOR = IDIR;
int VER = KDIR;
int PHIDIR = JDIR;
int VHOR = VX1;
int VVER = VX3;
int VPHI = VX2;
#endif

} // namespace Params