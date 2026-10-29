#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void block_power_iter(
    MKL_INT N, MKL_INT M, int d, int n_iter, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    double *D_final, unsigned int random_state
);

void print_csr_as_dense(
    MKL_INT N, MKL_INT M, 
    const MKL_INT *row_ptr, const MKL_INT *col_ind, const double *values,
    MKL_INT max_rows, MKL_INT max_cols
);

int load_data(
    MKL_INT *N, MKL_INT *M, MKL_INT *nnz,
    MKL_INT **row_ptr, MKL_INT **col_ind,
    double **values, const char *file_path
);

void compute_pairwise_distances(MKL_INT N, int d, const double *D_final, double *Dist);
