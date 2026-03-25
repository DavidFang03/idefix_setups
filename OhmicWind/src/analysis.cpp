#include "analysis.hpp"
#include "fluid.hpp"
#include "idefix.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

Analysis::Analysis(
    Input &input, Grid &grid, DataBlock &data, Output &output, std::string filename) {
    this->d        = new DataBlockHost(data);
    this->grid     = &grid;
    this->filename = filename;
    // this->shear    = data.hydro->sbS;
    // this->rotation = data.hydro->OmegaZ;
    this->precision = 10;
}

/* **************************************************************** */
void Analysis::WriteField(double data) {
    /*
     * Write a global profile to a file
     *
     *
     **************************************************************** */
    if (idfx::prank == 0) {
        int col_width = precision + 10;
        this->file << std::scientific << std::setw(col_width) << data;
    }
    return;
}

void Analysis::ResetAnalysis() {
    GridHost gh(*this->grid);
    gh.SyncFromDevice();
    int col_width = precision + 10;
    if (idfx::prank == 0) {
        file.open(filename, std::ios::trunc);
        file << std::setw(col_width) << "t";
        file << std::setw(col_width) << "divB";
        file << std::endl;
        file.close();
    }
}

void Analysis::PerformAnalysis(DataBlock &data) {
    idfx::pushRegion("Analysis::PerformAnalysis");
    d->SyncFromDevice();
    // int fields[3];
    if (idfx::prank == 0) {
        file.open(filename, std::ios::app);
        file.precision(precision);
    }

    // const int nx0 = data.beg[IDIR] + data.np_int[IDIR] / 2;
    // const int ny0 = data.beg[JDIR] + 3 * data.np_int[JDIR] / 4;
    // const int nz0 = data.beg[KDIR];

    WriteField(data.t);
    WriteField(data.hydro->CheckDivB());

    if (idfx::prank == 0) {
        file << std::endl;
        file.close();
    }
    idfx::popRegion();
}
