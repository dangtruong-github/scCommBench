#pragma once

#include "kernels/kernels.h"
#include "utils.h"

typedef void (*algo_func_t)(
    int32_t, int32_t, int, int,
    const int32_t*, const int32_t*, const double*,
    double*, float*, float*, bool, unsigned int
);
