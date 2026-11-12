#include "kernels.h"


static void svd_flip_mpi(MKL_INT N, MKL_INT M, MKL_INT d, 
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
void block_power_iter_mpi(
    const MKL_INT N, const MKL_INT M_local, const int d,
    const int n_iter, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    double *D_final, unsigned int random_state,
    bool print_time
) {
    CPUTimer timer;

    // 1. Create MKL Sparse Matrix Handle (0-based indexing)
    sparse_matrix_t X_handle;
    mkl_sparse_d_create_csr(
        &X_handle,
        SPARSE_INDEX_BASE_ZERO,
        N, M_local,
        (MKL_INT*)row_ptr,
        (MKL_INT*)(row_ptr + 1),
        (MKL_INT*)col_ind,
        (double*)values
    );

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // 2. Memory allocations for temporary buffers
    double *V        = (double*)aligned_alloc_64(M_local * d * sizeof(double));
    double *U        = (double*)aligned_alloc_64(N * d * sizeof(double));
    double *W        = (double*)aligned_alloc_64(M_local * d * sizeof(double));
    double *D_temp   = (double*)aligned_alloc_64(N * d * sizeof(double));
    double *Vt_R     = (double*)aligned_alloc_64(d * d * sizeof(double));
    double *tau      = (double*)aligned_alloc_64(d * sizeof(double));
    double *superb   = (double*)aligned_alloc_64(d * sizeof(double));
    double *U_final  = (double*)aligned_alloc_64(N * d * sizeof(double));
    double *S        = (double*)aligned_alloc_64(d * sizeof(double));
    double *Vh_final = (double*)aligned_alloc_64(d * M_local * sizeof(double));

    // 3. Initialize random matrix V of size M x d with standard Gaussian noise
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, random_state);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, M_local * d, V, 0.0, 1.0);
    vslDeleteStream(&stream);

    // Initial QR factorization to orthonormalize V
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M_local, d, V, d, tau);
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M_local, d, d, V, d, tau);

    // 4. Block Power Iteration Loop
    for (int iter = 0; iter < n_iter; ++iter) {
        // Step 1a: Local Forward SpMM (D_temp = X_local @ V_local)
        cpu_timer_init(&timer, "Step 1a: Local Forward SpMM", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, D_temp, d);
        cpu_timer_destroy(&timer);

        // Step 1b: AllReduce Forward Pass (U = sum(D_temp))
        cpu_timer_init(&timer, "Step 1b: AllReduce Forward Pass", print_time);
        MPI_Allreduce(D_temp, U, N * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cpu_timer_destroy(&timer);

        // Step 2: Local Backward SpMM (W_local = X_local^T @ U)
        cpu_timer_init(&timer, "Step 2: Local Backward SpMM", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, U, d, d, 0.0, W, d);
        cpu_timer_destroy(&timer);

        // Step 3a: Compute Local Gram Matrix (G_local = W_local^T @ W_local)
        cpu_timer_init(&timer, "Step 3a: Local Gram Matrix", print_time);
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            d, d, M_local,
            1.0, W, d,
            W, d,
            0.0, Vt_R, d
        );
        cpu_timer_destroy(&timer);

        // Step 3b: AllReduce Gram Matrix (G_global = sum(G_local))
        cpu_timer_init(&timer, "Step 3b: AllReduce Gram Matrix", print_time);
        MPI_Allreduce(MPI_IN_PLACE, Vt_R, d * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cpu_timer_destroy(&timer);

        // Step 4a: Cholesky Factorization (G_global = R^T @ R)
        cpu_timer_init(&timer, "Step 4a: Cholesky Factorization", print_time);
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', d, Vt_R, d);
        cpu_timer_destroy(&timer);

        // Step 4b: Triangular Solve & Update V (V_local = W_local @ R^-1)
        cpu_timer_init(&timer, "Step 4b: Triangular Solve (V Update)", print_time);
        cblas_dtrsm(
            CblasRowMajor, CblasRight, CblasUpper,
            CblasNoTrans, CblasNonUnit,
            M_local, d,
            1.0, Vt_R, d,
            W, d
        );
        memcpy(V, W, M_local * d * sizeof(double));
        cpu_timer_destroy(&timer);
    }

    // Step 5a: Local Rayleigh-Ritz SpMM (U = X_local @ V_local)
    cpu_timer_init(&timer, "Step 5a: Local Rayleigh-Ritz SpMM", print_time);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, U, d);
    cpu_timer_destroy(&timer);

    // Step 5b: AllReduce Rayleigh-Ritz Pass (D_temp = sum(U))
    cpu_timer_init(&timer, "Step 5b: AllReduce Rayleigh-Ritz Pass", print_time);
    MPI_Allreduce(U, D_temp, N * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    cpu_timer_destroy(&timer);

    // Step 5c: Thin SVD on Projected Matrix (D_temp -> U_tilde @ S @ V_R_t)
    cpu_timer_init(&timer, "Step 5c: Thin SVD (Rayleigh-Ritz)", print_time);
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', N, d, D_temp, d, S, U_final, d, Vt_R, d, superb);
    cpu_timer_destroy(&timer);

    // Step 5d: Local Basis Reconstruction & Transpose (Vh_final_local = (V_local @ V_R)^T)
    cpu_timer_init(&timer, "Step 5d: Local Basis Reconstruction & Transpose", print_time);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M_local, d, d, 1.0, V, d, Vt_R, d, 0.0, W, d);
    mkl_domatcopy('R', 'T', M_local, d, 1.0, W, d, Vh_final, M_local);
    cpu_timer_destroy(&timer);

    // Step 6: SVD Sign Flip Alignment
    cpu_timer_init(&timer, "Step 6: SVD Sign Flip Alignment", print_time);
    svd_flip_mpi(N, M_local, d, U_final, Vh_final, W);
    cpu_timer_destroy(&timer);

    // Step 7a: Local Final Score SpMM (U = X_local @ Vh_final_local^T)
    cpu_timer_init(&timer, "Step 7a: Local Final Score SpMM", print_time);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, W, d, d, 0.0, U, d);
    cpu_timer_destroy(&timer);

    // Step 7b: AllReduce Final Score Pass (D_final = sum(U))
    cpu_timer_init(&timer, "Step 7b: AllReduce Final Score Pass", print_time);
    MPI_Allreduce(U, D_final, N * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    cpu_timer_destroy(&timer);

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
 * Computes memory-efficient K-nearest neighbor indices for a row-partitioned slice 
 * [N_start, N_end) of D_final (N x d) across MPI ranks.
 *
 * Parameters:
 *   N          : Total number of global samples (rows in D_final)
 *   N_start    : Starting global sample index for this node's partition
 *   N_end      : Ending global sample index (exclusive) for this node's partition
 *   d          : Feature dimension (cols)
 *   D_final    : Full input matrix of shape (N, d), available on all ranks
 *   K_nbs      : Output matrix of 32-bit neighbor indices of shape (N_local, num_K_nbs) [Pre-allocated]
 *   num_K_nbs  : Number of nearest neighbors (K) to extract per sample
 */
void knn_compute_mpi(
    MKL_INT N, MKL_INT N_start, MKL_INT N_end, int d,
    MKL_INT num_K_nbs, const double *D_final, int32_t *K_nbs
) {
    MKL_INT N_local = N_end - N_start;
    if (N_local <= 0 || N <= 0 || num_K_nbs <= 0) return;

    MKL_INT actual_k = (num_K_nbs < N) ? num_K_nbs : N;

    // 1. Compute squared row norms for all N samples: sq_norms[i] = sum_k (D[i, k]^2)
    double *sq_norms = (double*)aligned_alloc_64(N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < N; ++i) {
        double norm_sq = 0.0;
        const double *row = &D_final[i * d];
        for (int k = 0; k < d; ++k) {
            norm_sq += row[k] * row[k];
        }
        sq_norms[i] = norm_sq;
    }

    // 2. Row Tiling: Memory footprint capped to (TILE_SIZE x N) doubles
    const MKL_INT TILE_SIZE = 1024;
    double *Dist_block = (double*)aligned_alloc_64(TILE_SIZE * N * sizeof(double));

    for (MKL_INT local_offset = 0; local_offset < N_local; local_offset += TILE_SIZE) {
        MKL_INT B_curr = (TILE_SIZE < N_local - local_offset) ? TILE_SIZE : (N_local - local_offset);
        MKL_INT block_global_start = N_start + local_offset;
        const double *D_block = D_final + block_global_start * d;

        // Compute sub-matrix Gram products G_block (B_curr x N) = D_block (B_curr x d) * D_final^T (d x N)
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            B_curr, N, d,
            1.0, D_block, d,
            D_final, d,
            0.0, Dist_block, N
        );

        // 3. Extract Top-K Nearest Neighbors per row using thread-local max-heaps
        #pragma omp parallel
        {
            DistPair *heap = (DistPair*)malloc(actual_k * sizeof(DistPair));

            #pragma omp for schedule(dynamic)
            for (MKL_INT r = 0; r < B_curr; ++r) {
                MKL_INT global_i = block_global_start + r;
                MKL_INT local_i  = local_offset + r;
                const double *G_row = Dist_block + r * N;
                double sq_norm_i = sq_norms[global_i];

                MKL_INT heap_size = 0;

                for (MKL_INT j = 0; j < N; ++j) {
                    double g_ij = G_row[j];
                    double d_sq = sq_norm_i + sq_norms[j] - 2.0 * g_ij;
                    if (d_sq < 0.0) d_sq = 0.0;

                    if (heap_size < actual_k) {
                        heap[heap_size].dist_sq = d_sq;
                        heap[heap_size].index = (int32_t)j;
                        heapify_up(heap, heap_size);
                        heap_size++;
                    } else if (d_sq < heap[0].dist_sq) {
                        heap[0].dist_sq = d_sq;
                        heap[0].index = (int32_t)j;
                        heapify_down(heap, actual_k, 0);
                    }
                }

                // Extract sorted indices from heap into local row of K_nbs (nearest first)
                MKL_INT current_heap_size = actual_k;
                for (MKL_INT k = actual_k - 1; k >= 0; --k) {
                    K_nbs[local_i * num_K_nbs + k] = heap[0].index;
                    heap[0] = heap[current_heap_size - 1];
                    current_heap_size--;
                    if (current_heap_size > 0) {
                        heapify_down(heap, current_heap_size, 0);
                    }
                }

                // Fill padding if num_K_nbs > N
                for (MKL_INT k = actual_k; k < num_K_nbs; ++k) {
                    K_nbs[local_i * num_K_nbs + k] = -1;
                }
            }

            free(heap);
        }
    }

    mkl_free(Dist_block);
    mkl_free(sq_norms);
}

void mpi_baseline(
    const MKL_INT N, const MKL_INT M_local,
    const MKL_INT M_global, const int rank,
    const int world_size, const int d, const int n_iter,
    const MKL_INT num_K_nbs, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    MKL_INT *K_nb, double *block_power_time, double *knn_time,
    bool print_result, const unsigned int seed
) {
    CPUTimer timer;

    printf("Running rank=%d/%d with M_local=%lld/%lld\n", rank, world_size, (long long)M_local, (long long)M_global);

    // Allocate output matrix D_final (N x d)
    double *D_final = (double*)aligned_alloc_64(N * d * sizeof(double));

    // 2. Execute Block Power Iteration
    printf("=== Running Block Power Iteration (%d iterations, d=%d) of rank=%d ===\n", n_iter, d, rank);

    cpu_timer_init(&timer, "Block Power Iteration", print_result);
    block_power_iter_mpi(
        N, M_local, d, n_iter, row_ptr,
        col_ind, values, D_final, seed, print_result
    );
    *block_power_time = cpu_timer_destroy(&timer);

    printf("Block Power Iteration rank=%d completed in: %.4f seconds\n\n", rank, *block_power_time);

    // 3. Compute N x N Distance Matrix
    printf("=== Computing Pairwise Distance Matrix (%lld x %lld) of rank=%d ===\n", (long long)N, (long long)N, rank);

    cpu_timer_init(&timer, "KNN", print_result);
    MKL_INT N_start, N_end;
    get_start_end(N, rank, world_size, &N_start, &N_end);
    knn_compute_mpi(N, N_start, N_end, d, num_K_nbs, D_final, K_nb);
    *knn_time = cpu_timer_destroy(&timer);

    printf("Distance matrix computation of rank %d completed in: %.4f seconds\n\n", rank, *knn_time);

    if (rank == 0) {
        // 4. Print Results Summary
        printf("Top 5x5 Submatrix of K neighbor indices:\n");
        for (int i = 0; i < 5; ++i) {
            printf("Row %d: [", i);
            for (int j = 0; j < 5; ++j) {
                printf("%lld%s", (long long)K_nb[i * num_K_nbs + j], (j == 4) ? "" : ", ");
            }
            printf("]\n");
        }
    }

    mkl_free(D_final);
}
