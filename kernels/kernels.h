#pragma once

#include <string.h>
#include <stddef.h>
#include <omp.h>

#include "../utils.h"

typedef struct {
    double dist_sq;
    MKL_INT index;
} DistPair;

void mpi_impl(
    const MKL_INT N, const MKL_INT M_local,
    const MKL_INT M_global, const int rank,
    const int world_size, const int d, const int n_iter,
    const MKL_INT num_K_nbs, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    MKL_INT *K_nb, float *block_power_time, float *knn_time,
    bool print_result, const unsigned int seed
);
void naive_seq_lib_baseline(
    const MKL_INT N, const MKL_INT M_local,
    const MKL_INT M_global, const int rank,
    const int world_size, const int d, const int n_iter,
    const MKL_INT num_K_nbs, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    MKL_INT *K_nb, float *block_power_time, float *knn_time,
    bool print_result, const unsigned int seed
);

// Standard binary max-heap sift-down
static inline void heapify_down(DistPair *heap, MKL_INT size, MKL_INT idx) {
    while (1) {
        MKL_INT largest = idx;
        MKL_INT left = 2 * idx + 1;
        MKL_INT right = 2 * idx + 2;

        if (left < size && heap[left].dist_sq > heap[largest].dist_sq) {
            largest = left;
        }
        if (right < size && heap[right].dist_sq > heap[largest].dist_sq) {
            largest = right;
        }
        if (largest == idx) break;

        DistPair tmp = heap[idx];
        heap[idx] = heap[largest];
        heap[largest] = tmp;

        idx = largest;
    }
}

// Standard binary max-heap sift-up
static inline void heapify_up(DistPair *heap, MKL_INT idx) {
    while (idx > 0) {
        MKL_INT parent = (idx - 1) / 2;
        if (heap[idx].dist_sq > heap[parent].dist_sq) {
            DistPair tmp = heap[idx];
            heap[idx] = heap[parent];
            heap[parent] = tmp;
            idx = parent;
        } else {
            break;
        }
    }
}

// Files I/O methods

int load_plink_partition_mpi(
    const char *prefix, int rank, int world_size,
    MKL_INT *out_N, MKL_INT *out_M_local, MKL_INT *out_M_glob,
    MKL_INT *out_nnz, MKL_INT **out_row_ptr,
    MKL_INT **out_col_ind, double **out_values
);
int save_dist_mat_bin_mpi(
    const char *file_path, const MKL_INT N,
    const unsigned int seed, const int d,
    const int n_iter, const MKL_INT num_K_nbs,
    const MKL_INT *K_nb, const int rank, const int world_size
);
