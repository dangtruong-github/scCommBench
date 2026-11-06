#pragma once

#include "kernels/kernels.h"
#include "utils.h"

typedef void (*algo_func_t)(
    int32_t, int32_t, int, int,
    const int32_t*, const int32_t*, const double*,
    double*, float*, float*, bool, unsigned int
);

int run_mode(
    const int32_t N, const int32_t M, const int d,
    const int n_iter, const unsigned int seed,
    const int32_t *row_ptr, const int32_t *col_ind,
    const double *values, algo_func_t func1,
    const char *save_path
);
