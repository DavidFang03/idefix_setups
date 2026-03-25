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
    this->shear    = data.hydro->sbS;
    // this->rotation = data.hydro->OmegaZ;
    this->precision = 10;
}

/* **************************************************************** */
double Analysis::Average(const int nfields, int fields[])
/*
 * compute the weighted average: int dphi dz rho *infield/int dphi dz rho
 *
 **************************************************************** */
{
    real outfield = 0;

    for (int k = d->beg[KDIR]; k < d->end[KDIR]; k++) {
        for (int j = d->beg[JDIR]; j < d->end[JDIR]; j++) {
            for (int i = d->beg[IDIR]; i < d->end[IDIR]; i++) {

                real q = 1.0;
                for (int n = 0; n < nfields; n++) {
                    // Substract Keplerian flow if vphi
                    if (fields[n] == VX2) {
                        q = q * (d->Vc(fields[n], k, j, i) - d->x[IDIR](i) * this->shear);
                    } else {
                        q = q * d->Vc(fields[n], k, j, i);
                    }
                }
                outfield += q;
            }
        }
    }

    // Reduce
#ifdef WITH_MPI
    real reducedValue;
    MPI_Reduce(&outfield, &reducedValue, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    outfield = reducedValue;
#endif

    outfield = outfield / ((double) grid->np_int[IDIR] * grid->np_int[JDIR] * grid->np_int[KDIR]);

    return outfield;
}

double Analysis::CorrelationLength(const int field)
/*
 * compute the correlation length along the z direction
 *
 **************************************************************** */
{
    real Integral        = 0;
    real SquaredIntegral = 0;
    real outfield        = 0;
    int j                = (int) grid->np_int[JDIR] / 2 + 1;

    for (int i = d->beg[IDIR]; i < d->end[IDIR]; i++) {
        for (int k = d->beg[KDIR]; k < d->end[KDIR]; k++) {
            Integral += d->Vc(field, k, j, i) * d->dx[KDIR][k];
            SquaredIntegral += d->Vc(field, k, j, i) * d->Vc(field, k, j, i) * d->dx[KDIR][k];
        }
    }
    // Reduce
#ifdef WITH_MPI
    real reducedIntegral;
    MPI_Reduce(&Integral, &reducedIntegral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    Integral = reducedIntegral;
    real reducedSquaredIntegral;
    MPI_Reduce(
        &SquaredIntegral, &reducedSquaredIntegral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    SquaredIntegral = reducedSquaredIntegral;
#endif

    outfield = Integral * Integral / SquaredIntegral / ((double) grid->np_int[IDIR]);

    return outfield;
}

/* **************************************************************** */
double Analysis::QualityFactorX()
/*
 * compute the quality factor along the X direction
 *
 **************************************************************** */
{
    real outfield = 0;

    for (int k = d->beg[KDIR]; k < d->end[KDIR]; k++) {
        for (int j = d->beg[JDIR]; j < d->end[JDIR]; j++) {
            for (int i = d->beg[IDIR]; i < d->end[IDIR]; i++) {
                outfield += 2 * 3.1415 * abs(d->Vc(BX1, k, j, i)) / (d->dx[IDIR][i])
                            / sqrt(d->Vc(RHO, k, j, i));
            }
        }
    }

    // Reduce
#ifdef WITH_MPI
    real reducedValue;
    MPI_Reduce(&outfield, &reducedValue, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    outfield = reducedValue;
#endif

    outfield = outfield / ((double) grid->np_int[IDIR] * grid->np_int[JDIR] * grid->np_int[KDIR]);

    return outfield;
}

/* **************************************************************** */
double Analysis::QualityFactorY()
/*
 * compute the quality factor along the Y direction
 *
 **************************************************************** */
{
    real outfield = 0;

    for (int k = d->beg[KDIR]; k < d->end[KDIR]; k++) {
        for (int j = d->beg[JDIR]; j < d->end[JDIR]; j++) {
            for (int i = d->beg[IDIR]; i < d->end[IDIR]; i++) {
                outfield += 2 * 3.1415 * abs(d->Vc(BX2, k, j, i)) / d->dx[JDIR][j]
                            / sqrt(d->Vc(RHO, k, j, i));
            }
        }
    }

    // Reduce
#ifdef WITH_MPI
    real reducedValue;
    MPI_Reduce(&outfield, &reducedValue, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    outfield = reducedValue;
#endif

    outfield = outfield / ((double) grid->np_int[IDIR] * grid->np_int[JDIR] * grid->np_int[KDIR]);

    return outfield;
}

/* **************************************************************** */
double Analysis::QualityFactorZ()
/*
 * compute the quality factor along the Z direction
 *
 **************************************************************** */
{
    real outfield = 0;

    for (int k = d->beg[KDIR]; k < d->end[KDIR]; k++) {
        for (int j = d->beg[JDIR]; j < d->end[JDIR]; j++) {
            for (int i = d->beg[IDIR]; i < d->end[IDIR]; i++) {
                outfield += 2 * 3.1415 * abs(d->Vc(BX3, k, j, i)) / d->dx[KDIR][k]
                            / sqrt(d->Vc(RHO, k, j, i));
            }
        }
    }

    // Reduce
#ifdef WITH_MPI
    real reducedValue;
    MPI_Reduce(&outfield, &reducedValue, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    outfield = reducedValue;
#endif

    outfield = outfield / ((double) grid->np_int[IDIR] * grid->np_int[JDIR] * grid->np_int[KDIR]);

    return outfield;
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
        file << std::setw(col_width) << "vx2";
        file << std::setw(col_width) << "vy2";
        file << std::setw(col_width) << "vz2";
        file << std::setw(col_width) << "kinx";
        file << std::setw(col_width) << "kiny";
        file << std::setw(col_width) << "kinz";
        file << std::setw(col_width) << "rhovrvphi";
        file << std::setw(col_width) << "bx";
        file << std::setw(col_width) << "by";
        file << std::setw(col_width) << "bz";
        file << std::setw(col_width) << "bxby";
        file << std::setw(col_width) << "prs";
        file << std::setw(col_width) << "rho";
        file << std::setw(col_width) << "Qx";
        file << std::setw(col_width) << "Qy";
        file << std::setw(col_width) << "Qz";
        file << std::setw(col_width) << "Lz";
        file << std::endl;
        file.close();
    }
}

void Analysis::PerformAnalysis(DataBlock &data) {
    idfx::pushRegion("Analysis::PerformAnalysis");
    d->SyncFromDevice();
    int fields[3];
    if (idfx::prank == 0) {
        file.open(filename, std::ios::app);
        file.precision(precision);
    }

    const int nx0 = data.beg[IDIR] + data.np_int[IDIR] / 2;
    const int ny0 = data.beg[JDIR] + 3 * data.np_int[JDIR] / 4;
    const int nz0 = data.beg[KDIR];

    WriteField(data.t);

    fields[0] = VX1;
    fields[1] = VX1;
    WriteField(Average(2, fields));

    fields[0] = VX2;
    fields[1] = VX2;
    WriteField(Average(2, fields));

    fields[0] = VX3;
    fields[1] = VX3;
    WriteField(Average(2, fields));

    fields[0] = RHO;
    fields[1] = VX1;
    fields[2] = VX1;
    WriteField(HALF_F * Average(3, fields));

    fields[0] = RHO;
    fields[1] = VX2;
    fields[2] = VX2;
    WriteField(HALF_F * Average(3, fields));

    fields[0] = RHO;
    fields[1] = VX3;
    fields[2] = VX3;
    WriteField(HALF_F * Average(3, fields));

    fields[0] = RHO;
    fields[1] = VX1;
    fields[2] = VX2;
    WriteField(HALF_F * Average(3, fields));

    fields[0] = BX1;
    fields[1] = BX1;
    WriteField(Average(2, fields));

    fields[0] = BX2;
    fields[1] = BX2;
    WriteField(Average(2, fields));

    fields[0] = BX3;
    fields[1] = BX3;
    WriteField(Average(2, fields));

    fields[0] = BX1;
    fields[1] = BX2;
    WriteField(Average(2, fields));

    fields[0] = RHO;
    fields[1] = RHO;
    WriteField(Average(2, fields));

    fields[0] = RHO;
    fields[1] = RHO;
    WriteField(Average(2, fields));

    WriteField(QualityFactorX());
    WriteField(QualityFactorY());
    WriteField(QualityFactorZ());

    WriteField(CorrelationLength(BX2));

    if (idfx::prank == 0) {
        file << std::endl;
        file.close();
    }
    idfx::popRegion();
}
