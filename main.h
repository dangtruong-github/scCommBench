#pragma once

#include "kernels/kernels.h"
#include "utils.h"

typedef void (*algo_func_t)(
    const MKL_INT, const MKL_INT, const MKL_INT,
    const int, const int, const int, const int,
    const MKL_INT*, const MKL_INT*, const double*,
    double*, float*, float*, bool, unsigned int
);

int run_mode(
    const MKL_INT N, const MKL_INT M_local,
    const MKL_INT M_global, const int rank,
    const int world_size, const int d, const int n_iter,
    const unsigned int seed, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    algo_func_t func1, const char *save_path
);
