#pragma once

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>
#include <stdbool.h>

typedef struct {
    double start_time;
    const char *function_name;
    bool print_out;
} CPUTimer;

// Helper function to get current time in seconds
double cpu_timer_get_time(void);

// Replaces constructor
void cpu_timer_init(CPUTimer *timer, const char *name, bool print_out);

// Replaces destructor (calculates elapsed time and optionally prints)
double cpu_timer_destroy(CPUTimer *timer);

static inline void* aligned_alloc_64(size_t size) {
    size_t padded = ((size + 63) / 64) * 64;
    return mkl_malloc(padded, 64);
}

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

int load_data_plink(
    int32_t *out_N, int32_t *out_M, int32_t *out_nnz,
    int32_t **out_row_ptr, int32_t **out_col_ind,
    double **out_values, const char *prefix
);
int save_dist_mat_bin(
    const char *file_path, const int32_t N,
    const unsigned int seed, const int d,
    const int n_iter, const double *Dist_matrix_new
);

bool validate(
    double *dist_mat_truth, double *dist_mat,
    const int32_t M, const int32_t N,
    const double atol, const double rtol
);
