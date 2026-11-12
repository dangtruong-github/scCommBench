#include "utils.h"

// Function to print a CSR matrix in dense 2D grid format
void print_csr_as_dense(
    MKL_INT N, MKL_INT M, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    MKL_INT max_rows, MKL_INT max_cols
) {
    MKL_INT r_limit = (max_rows > 0 && max_rows < N) ? max_rows : N;
    MKL_INT c_limit = (max_cols > 0 && max_cols < M) ? max_cols : M;

    printf("\n=== Sparse Matrix Printed in Dense View (%lld x %lld preview of %lld x %lld) ===\n", 
           (long long)r_limit, (long long)c_limit, (long long)N, (long long)M);

    for (MKL_INT i = 0; i < r_limit; ++i) {
        MKL_INT row_start = row_ptr[i];
        MKL_INT row_end   = row_ptr[i + 1];
        MKL_INT ptr = row_start;

        printf("[");
        for (MKL_INT j = 0; j < c_limit; ++j) {
            double val = 0.0;

            // Check if current column matches the stored non-zero entry
            if (ptr < row_end && col_ind[ptr] == j) {
                val = values[ptr];
                ptr++;
            }

            printf("%6.2f", val);
            if (j < c_limit - 1) printf(", ");
        }
        if (c_limit < M) printf("  ...");
        printf("]");
        if (i < r_limit) printf(",");
        printf("\n");
    }

    if (r_limit < N) {
        printf("...\n");
    }
    printf("\n");
}

bool validate(
    double *dist_mat_truth, double *dist_mat,
    const MKL_INT M, const MKL_INT N,
    const double atol, const double rtol
) {
    if (dist_mat_truth == NULL || dist_mat == NULL) {
        return dist_mat_truth == dist_mat;
    }

    bool matches = true;

    for (MKL_INT r = 0; r < M; ++r) {
        for (MKL_INT c = 0; c < N; ++c) {
            uint64_t i = (uint64_t)r * N + c;
            double truth = dist_mat_truth[i];
            double val = dist_mat[i];

            if (truth == val) {
                continue;
            }

            if (isnan(truth) || isnan(val)) {
                if (isnan(truth) && isnan(val)) {
                    continue;
                }
                printf("Mismatch at index %lld (row %d, col %d): truth = %g, val = %g (NaN mismatch)\n",
                       (long long)i, r, c, truth, val);
                matches = false;
                continue;
            }

            double diff = fabs(truth - val);
            double max_val = fmax(fabs(truth), fabs(val));

            if (diff > atol && diff > rtol * max_val) {
                printf("Mismatch at index %lld (row %d, col %d): truth = %g, val = %g, diff = %e (atol = %e, rtol * max_val = %e)\n",
                       (long long)i, r, c, truth, val, diff, atol, rtol * max_val);
                matches = false;
            }
        }
    }

    return matches;
}

// timer
double cpu_timer_get_time(void) {
    struct timespec ts;
    // CLOCK_MONOTONIC ensures time doesn't jump backwards if system time updates
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void cpu_timer_init(CPUTimer *timer, const char *name, bool print_out) {
    if (!timer) return;
    
    timer->function_name = name;
    timer->print_out = print_out;
    timer->start_time = cpu_timer_get_time();
}

double cpu_timer_destroy(CPUTimer *timer) {
    if (!timer) return 0.0;
    
    double elapsed = cpu_timer_get_time() - timer->start_time;
    
    if (timer->print_out) {
        printf("[%s] Elapsed time: %.6f seconds\n", 
               timer->function_name ? timer->function_name : "Timer", 
               elapsed);
    }
    
    return elapsed;
}

void get_start_end(MKL_INT N, int rank, int world_size, MKL_INT *N_start, MKL_INT *N_end) {
    if (!N_start || !N_end) return;
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        *N_start = 0;
        *N_end = 0;
        return;
    }

    MKL_INT base_N = N / world_size;
    MKL_INT rem    = N % world_size;

    // Distribute remainder 1 by 1 to the first 'rem' ranks
    MKL_INT local_count = base_N + (rank < rem ? 1 : 0);
    MKL_INT start       = rank * base_N + (rank < rem ? rank : rem);

    *N_start = start;
    *N_end   = start + local_count;
}

#include <stdio.h>
#include <mkl.h>

/**
 * Computes theoretical FLOPs for block_power_iter.
 *
 * Parameters:
 *   N      : Number of rows in sparse matrix X
 *   M      : Number of columns in sparse matrix X
 *   d      : Number of principal components
 *   n_iter : Number of power iterations
 *   nnz    : Total non-zero entries in CSR matrix X
 */
double compute_block_power_iter_flops(MKL_INT N, MKL_INT M, int d, int n_iter, MKL_INT nnz) {
    double d_dbl = (double)d;
    double N_dbl = (double)N;
    double M_dbl = (double)M;
    double nnz_dbl = (double)nnz;

    // 1. Initial QR factorization (dgeqrf + dorgqr) on random matrix V (M x d)
    double flops_initial_qr = 4.0 * M_dbl * d_dbl * d_dbl - (4.0 / 3.0) * d_dbl * d_dbl * d_dbl;

    // 2. Iteration Loop (n_iter iterations)
    // - Step 1: Forward SpMM U = X @ V (N x d) -> 2 * nnz * d
    // - Step 2: Backward SpMM W = X^T @ U (M x d) -> 2 * nnz * d
    // - Step 3&4: GEMM Vt_R = W^T @ W (d x d) -> 2 * M * d^2
    // - Step 3&4: Cholesky dpotrf on (d x d) -> (1/3) * d^3
    // - Step 3&4: TRSM V = W @ R^-1 (M x d) -> M * d^2
    double flops_per_iter = (4.0 * nnz_dbl * d_dbl) 
                          + (3.0 * M_dbl * d_dbl * d_dbl) 
                          + ((1.0 / 3.0) * d_dbl * d_dbl * d_dbl);
    double flops_loop = (double)n_iter * flops_per_iter;

    // 3. Rayleigh-Ritz Step & Final Computations
    // - SpMM D_temp = X @ V (N x d) -> 2 * nnz * d
    // - Thin SVD dgesvd on D_temp (N x d) -> ~ 6 * N * d^2 + 11 * d^3
    // - GEMM V_final = V @ V_R (M x d) -> 2 * M * d^2
    // - svd_flip sign checks -> N * d
    // - SpMM D_final = X @ Vh_final^T (N x d) -> 2 * nnz * d
    double flops_spmm_rr    = 2.0 * nnz_dbl * d_dbl;
    double flops_svd        = 6.0 * N_dbl * d_dbl * d_dbl + 11.0 * d_dbl * d_dbl * d_dbl;
    double flops_gemm_rr   = 2.0 * M_dbl * d_dbl * d_dbl;
    double flops_svd_flip   = N_dbl * d_dbl;
    double flops_spmm_final = 2.0 * nnz_dbl * d_dbl;

    double total_flops = flops_initial_qr + flops_loop + flops_spmm_rr 
                       + flops_svd + flops_gemm_rr + flops_svd_flip + flops_spmm_final;

    return total_flops;
}

/**
 * Computes theoretical FLOPs for knn_extract.
 *
 * Parameters:
 *   N : Number of rows/samples in D_final
 *   d : Feature dimension
 */
double compute_knn_extract_flops(MKL_INT N, int d) {
    double N_dbl = (double)N;
    double d_dbl = (double)d;

    // 1. Squared row norms sum_k(D[i, k]^2): d multiplies + d adds per row
    double flops_norms = 2.0 * N_dbl * d_dbl;

    // 2. Gram Matrix GEMM: D_final (N x d) * D_final^T (d x N) -> (N x N)
    double flops_gemm = 2.0 * N_dbl * N_dbl * d_dbl;

    // 3. Distance calculation: d_sq = sq_i + sq_j - 2.0 * g_ij (3 ops per pair)
    double flops_dist = 3.0 * N_dbl * N_dbl;

    return flops_norms + flops_gemm + flops_dist;
}

/**
 * Helper to compute throughput in GFLOP/s (or TFLOP/s)
 */
double get_theoretical_flop(MKL_INT N, MKL_INT M, int d, int n_iter, MKL_INT nnz) {
    double bpi_flops = compute_block_power_iter_flops(N, M, d, n_iter, nnz);
    double knn_flops = compute_knn_extract_flops(N, d);
    double total_flops = bpi_flops + knn_flops;

    return total_flops;
}
