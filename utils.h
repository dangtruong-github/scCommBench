#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <mkl.h>
#include <mpi.h>

// Helper to resolve the correct MPI datatype for MKL_INT automatically
#if defined(MKL_ILP64) || (SIZEOF_MKL_INT == 8)
    #define MPI_MKL_INT MPI_INT64_T
#else
    #define MPI_MKL_INT MPI_INT
#endif

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
    MKL_INT N, MKL_INT M, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    MKL_INT max_rows, MKL_INT max_cols
);

bool validate(
    double *dist_mat_truth, double *dist_mat,
    const MKL_INT M, const MKL_INT N,
    const double atol, const double rtol
);

void get_start_end(MKL_INT N, int rank, int world_size, MKL_INT *N_start, MKL_INT *N_end);

double get_theoretical_flop(MKL_INT N, MKL_INT M, int d, int n_iter, MKL_INT nnz);
