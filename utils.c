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
