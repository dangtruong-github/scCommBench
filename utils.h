#pragma once

#define _XOPEN_SOURCE 700

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>
#include <stdbool.h>

#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <mpi.h>

// Helper to resolve the correct MPI datatype for MKL_INT automatically
#if defined(MKL_ILP64) || (SIZEOF_MKL_INT == 8)
    #define MPI_MKL_INT MPI_INT64_T
#else
    #define MPI_MKL_INT MPI_INT
#endif

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

int load_data_plink(
    MKL_INT *out_N, MKL_INT *out_M, MKL_INT *out_nnz,
    MKL_INT **out_row_ptr, MKL_INT **out_col_ind,
    double **out_values, const char *prefix
);
int load_plink_partition_mpi(
    const char *prefix, int rank, int world_size,
    MKL_INT *out_N, MKL_INT *out_M_local, MKL_INT *out_M_glob,
    MKL_INT *out_nnz, MKL_INT **out_row_ptr,
    MKL_INT **out_col_ind, double **out_values
);


int save_dist_mat_bin(
    const char *file_path, const MKL_INT N,
    const unsigned int seed, const int d,
    const int n_iter, const double *Dist_matrix_new
);

bool validate(
    double *dist_mat_truth, double *dist_mat,
    const MKL_INT M, const MKL_INT N,
    const double atol, const double rtol
);
