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
#ifdef RELOAD
inline std::string reload_path;
#endif
} // namespace Params