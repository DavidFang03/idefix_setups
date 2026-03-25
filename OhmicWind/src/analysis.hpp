#ifndef ANALYSIS_HPP_
#define ANALYSIS_HPP_

#include "dataBlock.hpp"
#include "dataBlockHost.hpp"
#include "grid.hpp"
#include "idefix.hpp"
#include "input.hpp"
#include "output.hpp"
#include <fstream>
#include <iostream>

class Analysis {
    public:
    // Constructor from Setup arguments
    Analysis(Input &, Grid &, DataBlock &, Output &, std::string);
    void ResetAnalysis();
    void PerformAnalysis(DataBlock &);

    private:
    void WriteField(double);
    DataBlockHost *d;
    Grid *grid;

    int precision;
    // real shear;
    // real rotation;
    std::string filename;

    std::ofstream file;
};

#endif // ANALYSIS_HPP__
