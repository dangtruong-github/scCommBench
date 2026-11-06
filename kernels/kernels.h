#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>

#include <math.h>
#include <time.h>
#include <mkl.h>
#include <omp.h>

#include "../utils.h"

void naive_seq_lib_baseline(
    const MKL_INT N, const MKL_INT M, const int d, const int n_iter,
    const MKL_INT *row_ptr, const MKL_INT *col_ind,
    const double *values, double *Dist_matrix,
    float *block_power_time, float *knn_time,
    bool print_result, const unsigned int seed
);
