#include "kernels.h"


static void svd_flip(MKL_INT N, MKL_INT M, MKL_INT d, 
                     double *U, double *Vh, double *V_final) {
    #pragma omp parallel for schedule(static)
    for (MKL_INT k = 0; k < d; ++k) {
        double max_val = -1.0;
        double sign = 1.0;

        // Find max absolute value in column k of U
        for (MKL_INT i = 0; i < N; ++i) {
            double val = U[i * d + k];
            double abs_val = fabs(val);
            if (abs_val > max_val) {
                max_val = abs_val;
                sign = (val < 0.0) ? -1.0 : 1.0;
            }
        }

        if (sign < 0.0) {
            // Flip column k of U
            for (MKL_INT i = 0; i < N; ++i) {
                U[i * d + k] = -U[i * d + k];
            }

            // Flip row k of Vh (contiguous memory, vectorizable)
            #pragma omp simd
            for (MKL_INT j = 0; j < M; ++j) {
                Vh[k * M + j] = -Vh[k * M + j];
            }

            // Flip column k of V_final
            if (V_final != NULL) {
                for (MKL_INT j = 0; j < M; ++j) {
                    V_final[j * d + k] = -V_final[j * d + k];
                }
            }
        }
    }
}

/**
 * Block Power Iteration (Subspace Iteration) for Sparse Matrix in CSR format.
 *
 * Inputs:
 *   N, M         : Matrix dimensions (N x M)
 *   d            : Number of principal components / singular vectors
 *   random_state : Seed for reproducibility
 *   n_iter       : Number of power iterations
 *   row_ptr      : CSR row pointer array (length N + 1)
 *   col_ind      : CSR column index array (length nnz)
 *   values       : CSR values array (length nnz)
 *
 * Outputs (pre-allocated by caller):
 *   U_final      : Left singular vectors (N x d), Row-Major
 *   S            : Singular values (d)
 *   Vh_final     : Right singular vectors transposed (d x M), Row-Major
 *   D_final      : Reconstructed product X @ Vh_final.T (N x d), Row-Major
 */
void block_power_iter(
    MKL_INT N, MKL_INT M, int d, int n_iter, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    double *D_final, unsigned int random_state, bool print_time
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

    // 2. Memory allocations for temporary buffers
    double *V        = (double*)aligned_alloc_64(M * d * sizeof(double));
    double *U        = (double*)aligned_alloc_64(N * d * sizeof(double));
    double *W        = (double*)aligned_alloc_64(M * d * sizeof(double));
    double *D_temp   = (double*)aligned_alloc_64(N * d * sizeof(double));
    double *Vt_R     = (double*)aligned_alloc_64(d * d * sizeof(double));
    double *tau      = (double*)aligned_alloc_64(d * sizeof(double));
    double *superb   = (double*)aligned_alloc_64(d * sizeof(double));
    double *U_final  = (double*)aligned_alloc_64(N * d * sizeof(double));
    double *S        = (double*)aligned_alloc_64(d * sizeof(double));
    double *Vh_final = (double*)aligned_alloc_64(d * M * sizeof(double));

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
        cpu_timer_init(&timer, "Step 1", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, U, d);
        cpu_timer_destroy(&timer);

        // Step 2: Backward pass W = X.T @ U (M x d)
        cpu_timer_init(&timer, "Step 2", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, U, d, d, 0.0, W, d);
        cpu_timer_destroy(&timer);

        cpu_timer_init(&timer, "Step 3&4 new", print_time);
        // Step 3 & 4: Orthonormalize d vectors and update V
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            d, d, M,
            1.0, W, d,
            W, d,
            0.0, Vt_R, d
        );

        // 2. Cholesky factorization: G = R^T @ R
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', d, Vt_R, d);

        // 3. V = W @ R^{-1} (in-place solve)
        cblas_dtrsm(
            CblasRowMajor, CblasRight, CblasUpper,
            CblasNoTrans, CblasNonUnit,
            M, d,
            1.0, Vt_R, d,
            W, d
        );
        cpu_timer_destroy(&timer);
        

        memcpy(V, W, M * d * sizeof(double));
    }

    // 5. Rayleigh-Ritz Step
    // D_temp = X @ V (N x d)
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, D_temp, d);

    // Thin SVD on D_temp: D_temp = U_tilde @ diag(S) @ V_R_t
    // Output: U_tilde -> U_final (N x d), S -> S (d), V_R_t -> Vt_R (d x d)
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', N, d, D_temp, d, S, U_final, d, Vt_R, d, superb);

    // Reconstruct V_final (M x d) = V (M x d) @ V_R (d x d)
    // Reusing array 'W' to hold V_final
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d, d, 1.0, V, d, Vt_R, d, 0.0, W, d);

    // Transpose V_final (M x d) -> Vh_final (d x M)
    mkl_domatcopy('R', 'T', M, d, 1.0, W, d, Vh_final, M);

    // 6. Align signs
    svd_flip(N, M, d, U_final, Vh_final, W);

    // 7. Compute d_final = X @ Vh_final.T (N x d)
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, W, d, d, 0.0, D_final, d);

    // Cleanup
    mkl_sparse_destroy(X_handle);
    mkl_free(V);
    mkl_free(U);
    mkl_free(W);
    mkl_free(D_temp);
    mkl_free(Vt_R);
    mkl_free(tau);
    mkl_free(superb);
}

/**
 * Computes memory-efficient K-nearest neighbor indices for D_final (N x d) in pure C.
 *
 * Parameters:
 *   N          : Number of samples (rows)
 *   d          : Feature dimension (cols)
 *   D_final    : Input matrix of shape (N, d)
 *   K_nbs      : Output matrix of neighbor indices of shape (N, num_K_nbs) [Pre-allocated]
 *   num_K_nbs  : Number of nearest neighbors (K) to extract per sample
 */
void knn_extract(
    const MKL_INT N, const int d, const MKL_INT num_K_nbs,
    const double *D_final, MKL_INT *K_nbs
) {
    if (N <= 0 || num_K_nbs <= 0) return;

    MKL_INT actual_k = (num_K_nbs < N) ? num_K_nbs : N;

    // 1. Compute squared row norms: sq_norms[i] = sum_k (D[i, k]^2)
    double *sq_norms = (double*)aligned_alloc_64(N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < N; ++i) {
        double norm_sq = 0.0;
        for (int k = 0; k < d; ++k) {
            double val = D_final[i * d + k];
            norm_sq += val * val;
        }
        sq_norms[i] = norm_sq;
    }

    // 2. Row Tiling: Memory footprint capped to (TILE_SIZE x N)
    const MKL_INT TILE_SIZE = 1024;
    double *Dist_block = (double*)aligned_alloc_64(TILE_SIZE * N * sizeof(double));

    for (MKL_INT block_start = 0; block_start < N; block_start += TILE_SIZE) {
        MKL_INT B_curr = (TILE_SIZE < N - block_start) ? TILE_SIZE : (N - block_start);
        const double *D_block = D_final + block_start * d;

        // Compute sub-matrix Gram products G_block (B_curr x N) = D_block (B_curr x d) * D_final^T (d x N)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B_curr, N, d,
                    1.0, D_block, d,
                    D_final, d,
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
                    double g_ij = G_row[j];
                    double d_sq = sq_norm_i + sq_norms[j] - 2.0 * g_ij;
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

    // Allocate output matrix D_final (N x d)
    double *D_final = (double*)aligned_alloc_64(N * d * sizeof(double));

    // 2. Execute Block Power Iteration
    printf("\n=== Running Block Power Iteration (%d iterations, d=%d) ===\n", n_iter, d);

    cpu_timer_init(&timer, "Block Power Iteration", print_result);
    block_power_iter(
        N, M_local, d, n_iter, row_ptr, col_ind,
        values, D_final, seed, print_result
    );
    *block_power_time = cpu_timer_destroy(&timer);

    printf("Execution completed in: %.4f seconds\n\n", *block_power_time);

    // 3. Compute N x N Distance Matrix
    printf("=== Computing Pairwise Distance Matrix (%lld x %lld) ===\n", (long long)N, (long long)N);

    cpu_timer_init(&timer, "KNN", print_result);
    knn_extract(N, d, num_K_nbs, D_final, K_nb);
    *knn_time = cpu_timer_destroy(&timer);

    printf("Distance matrix computation completed in: %.4f seconds\n\n", *knn_time);

    // 4. Print Results Summary
    printf("Top 5x5 Submatrix of K neighbor indices:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Row %d: [", i);
        for (int j = 0; j < 5; ++j) {
            printf("%lld%s", (long long)K_nb[i * num_K_nbs + j], (j == 4) ? "" : ", ");
        }
        printf("]\n");
    }

    mkl_free(D_final);
}
