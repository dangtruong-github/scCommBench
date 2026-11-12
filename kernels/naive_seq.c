#include "kernels.h"

/**
 * Block Power Iteration (Subspace Iteration) for Sparse Matrix in CSR format.
 *
 * Inputs:
 *   N, M         : Matrix dimensions (N x M)
 *   d            : Number of principal components / subspace dimension
 *   n_iter       : Number of power iterations
 *   row_ptr      : CSR row pointer array (length N + 1)
 *   col_ind      : CSR column index array (length nnz)
 *   values       : CSR values array (length nnz)
 *   random_state : Seed for reproducibility
 *   print_time   : Whether to print step timings
 *
 * Returns:
 *   U            : Orthonormal-subspace projection X @ V (N x d), Row-Major (caller frees with mkl_free)
 */
void block_power_iter_naive(
    MKL_INT N, MKL_INT M, int d, int n_iter, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values, double *U,
    unsigned int random_state, bool print_time
) {
    CPUTimer timer;

    // 1. Create MKL Sparse Matrix Handle (0-based indexing)
    sparse_matrix_t X_handle;
    mkl_sparse_d_create_csr(
        &X_handle,
        SPARSE_INDEX_BASE_ZERO,
        N, M,
        (MKL_INT*)row_ptr,
        (MKL_INT*)(row_ptr + 1),
        (MKL_INT*)col_ind,
        (double*)values
    );

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // 2. Minimal allocations (U is returned directly for KNN extraction)
    double *V    = (double*)aligned_alloc_64(M * d * sizeof(double));
    double *Vt_R = (double*)aligned_alloc_64(d * d * sizeof(double));
    double *tau  = (double*)aligned_alloc_64(d * sizeof(double));

    // 3. Initialize random matrix V of size M x d with standard Gaussian noise
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, random_state);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, M * d, V, 0.0, 1.0);
    vslDeleteStream(&stream);

    // Initial QR factorization to orthonormalize V
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, d, V, d, tau);
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, d, d, V, d, tau);

    // 4. Block Power Iteration Loop
    for (int iter = 0; iter < n_iter; ++iter) {
        // Step 1: Forward pass U = X @ V (N x d)
        cpu_timer_init(&timer, "Step 1: Forward SpMM", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, U, d);
        cpu_timer_destroy(&timer);

        // Step 2: Backward pass V = X.T @ U (M x d)
        cpu_timer_init(&timer, "Step 2: Backward SpMM", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, U, d, d, 0.0, V, d);
        cpu_timer_destroy(&timer);

        // Step 3 & 4: Orthonormalize V in-place via Cholesky QR
        cpu_timer_init(&timer, "Step 3: Gram Matrix", print_time);
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            d, d, M,
            1.0, V, d,
            V, d,
            0.0, Vt_R, d
        );
        cpu_timer_destroy(&timer);

        // Cholesky factorization: G = R^T @ R
        cpu_timer_init(&timer, "Step 4a: Cholesky Factorization", print_time);
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', d, Vt_R, d);
        cpu_timer_destroy(&timer);

        // V = V @ R^{-1} (in-place solve)
        cpu_timer_init(&timer, "Step 4b: Triangular Solve (V Update)", print_time);
        cblas_dtrsm(
            CblasRowMajor, CblasRight, CblasUpper,
            CblasNoTrans, CblasNonUnit,
            M, d,
            1.0, Vt_R, d,
            V, d
        );
        cpu_timer_destroy(&timer);
    }

    // 5. Final subspace projection U = X @ V (N x d)
    // Orthogonal rotation/sign-flip from Rayleigh-Ritz SVD preserves pairwise distances
    cpu_timer_init(&timer, "Step 5: Local Rayleigh-Ritz SpMM", print_time);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, U, d);

    // Cleanup temporary buffers
    mkl_sparse_destroy(X_handle);
    mkl_free(V);
    mkl_free(Vt_R);
    mkl_free(tau);
}

/**
 * Computes memory-efficient K-nearest neighbor indices for U (N x d) in pure C.
 *
 * Parameters:
 *   N          : Number of samples (rows)
 *   d          : Feature dimension (cols)
 *   num_K_nbs  : Number of nearest neighbors (K) to extract per sample
 *   U          : Input projected matrix of shape (N, d)
 *   K_nbs      : Output matrix of neighbor indices of shape (N, num_K_nbs) [Pre-allocated]
 */
void knn_extract_naive(
    const MKL_INT N, const int d, const MKL_INT num_K_nbs,
    const double *U, MKL_INT *K_nbs
) {
    if (N <= 0 || num_K_nbs <= 0) return;

    MKL_INT actual_k = (num_K_nbs < N) ? num_K_nbs : N;

    // 1. Compute squared row norms: sq_norms[i] = sum_k (U[i, k]^2)
    double *sq_norms = (double*)aligned_alloc_64(N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < N; ++i) {
        double norm_sq = 0.0;
        for (int k = 0; k < d; ++k) {
            double val = U[i * d + k];
            norm_sq += val * val;
        }
        sq_norms[i] = norm_sq;
    }

    // 2. Row Tiling: Memory footprint capped to min(1024, N) x N
    const MKL_INT TILE_SIZE = (N < 1024) ? N : 1024;
    double *Dist_block = (double*)aligned_alloc_64(TILE_SIZE * N * sizeof(double));

    for (MKL_INT block_start = 0; block_start < N; block_start += TILE_SIZE) {
        MKL_INT B_curr = (TILE_SIZE < N - block_start) ? TILE_SIZE : (N - block_start);
        const double *U_block = U + block_start * d;

        // Compute sub-matrix Gram products G_block (B_curr x N) = U_block (B_curr x d) * U^T (d x N)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B_curr, N, d,
                    1.0, U_block, d,
                    U, d,
                    0.0, Dist_block, N);

        // 3. Extract Top-K Nearest Neighbors per row using thread-local max-heaps
        #pragma omp parallel
        {
            DistPair *heap = (DistPair*)malloc(actual_k * sizeof(DistPair));

            #pragma omp for schedule(dynamic)
            for (MKL_INT r = 0; r < B_curr; ++r) {
                MKL_INT i = block_start + r;
                const double *G_row = Dist_block + r * N;
                double sq_norm_i = sq_norms[i];

                MKL_INT heap_size = 0;

                for (MKL_INT j = 0; j < N; ++j) {
                    double d_sq = (j == i) ? 0.0 : (sq_norm_i + sq_norms[j] - 2.0 * G_row[j]);
                    if (d_sq < 0.0) d_sq = 0.0;

                    if (heap_size < actual_k) {
                        heap[heap_size].dist_sq = d_sq;
                        heap[heap_size].index = j;
                        heapify_up(heap, heap_size);
                        heap_size++;
                    } else if (d_sq < heap[0].dist_sq) {
                        heap[0].dist_sq = d_sq;
                        heap[0].index = j;
                        heapify_down(heap, actual_k, 0);
                    }
                }

                // Extract sorted indices from heap into output (nearest first)
                MKL_INT current_heap_size = actual_k;
                for (MKL_INT k = actual_k - 1; k >= 0; --k) {
                    K_nbs[i * num_K_nbs + k] = heap[0].index;
                    heap[0] = heap[current_heap_size - 1];
                    current_heap_size--;
                    if (current_heap_size > 0) {
                        heapify_down(heap, current_heap_size, 0);
                    }
                }

                // Fill padding if num_K_nbs > N
                for (MKL_INT k = actual_k; k < num_K_nbs; ++k) {
                    K_nbs[i * num_K_nbs + k] = -1;
                }
            }

            free(heap);
        }
    }

    mkl_free(Dist_block);
    mkl_free(sq_norms);
}

void naive_seq_lib_baseline(
    const MKL_INT N, const MKL_INT M_local,
    const MKL_INT M_global, const int rank,
    const int world_size, const int d, const int n_iter,
    const MKL_INT num_K_nbs, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    MKL_INT *K_nb, double *block_power_time, double *knn_time,
    bool print_result, const unsigned int seed
) {
    printf("M_global=%d, rank=%d, world_size=%d\n", M_global, rank, world_size);

    if (world_size > 1) {
        printf("This only works for np=1");
        exit(1);
    }

    CPUTimer timer;
    double *U = (double*)aligned_alloc_64(N * d * sizeof(double));

    // 1. Execute Block Power Iteration and obtain projected embedding U (N x d)
    cpu_timer_init(&timer, "Block Power Iteration", print_result);
    block_power_iter_naive(
        N, M_local, d, n_iter, row_ptr,
        col_ind, values, U, seed, print_result
    );
    *block_power_time = cpu_timer_destroy(&timer);

    // 2. Compute K-Nearest Neighbors directly from U
    cpu_timer_init(&timer, "KNN", print_result);
    knn_extract_naive(N, d, num_K_nbs, U, K_nb);
    *knn_time = cpu_timer_destroy(&timer);

    // 3. Print Results Summary
    printf("Top 5x5 Submatrix of K neighbor indices:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Row %d: [", i);
        for (int j = 0; j < 5; ++j) {
            printf("%lld%s", (long long)K_nb[i * num_K_nbs + j], (j == 4) ? "" : ", ");
        }
        printf("]\n");
    }

    mkl_free(U);
}
