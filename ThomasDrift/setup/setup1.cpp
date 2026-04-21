#include "setup.hpp"
#include "idefix.hpp"
#include <algorithm>

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {}

void Setup::InitFlow(DataBlock &data) {

  DataBlockHost d(data);
  IdefixArray3D<real> myArray("debugRHO", d.np_tot[KDIR], d.np_tot[JDIR], d.np_tot[IDIR]); // GPU
  IdefixArray3D<real>::HostMirror myArrayHost = Kokkos::create_mirror_view(myArray);       // CPU

  for (int k = 0; k < d.np_tot[KDIR]; k++) {
    for (int j = 0; j < d.np_tot[JDIR]; j++) {
      for (int i = 0; i < d.np_tot[IDIR]; i++) {
        real R = d.x[IDIR](i);
        real Vk = 1.0 / sqrt(R);

        d.Vc(RHO, k, j, i) = 0.125 * R;

        myArrayHost(k, j, i) = d.Vc(RHO, k, j, i);
        d.Vc(VX1, k, j, i) = 0.0;
        d.Vc(VX3, k, j, i) = 0.0;
        d.Vc(VX2, k, j, i) = Vk;
      }
    }
  }

  // Send it all, if needed
  d.SyncToDevice();

  Kokkos::deep_copy(myArray, myArrayHost);

  // IdefixArray3D<real> myArray("debugMe", d.np_tot[KDIR], d.np_tot[JDIR], d.np_tot[IDIR]);
  std::string filename = "debugRHO_proc" + std::to_string(idfx::prank) + ".npy";
  idfx::DumpArray(filename, myArrayHost);
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock &data) {}