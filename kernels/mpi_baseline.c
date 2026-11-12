#include "kernels.h"

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
    double *U, unsigned int random_state,
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
    double *Vt_R     = (double*)aligned_alloc_64(d * d * sizeof(double));
    double *tau      = (double*)aligned_alloc_64(d * sizeof(double));

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
        // Step 1a: Forward pass U = X @ V (N x d)
        cpu_timer_init(&timer, "Step 1a: Local Forward SpMM", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, U, d);
        cpu_timer_destroy(&timer);

        // Step 1b: AllReduce Forward Pass (U = sum(D_temp))
        cpu_timer_init(&timer, "Step 1b: AllReduce Forward Pass", print_time);
        MPI_Allreduce(MPI_IN_PLACE, U, N * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cpu_timer_destroy(&timer);

        // Step 2: Backward pass V = X.T @ U (M x d)
        cpu_timer_init(&timer, "Step 2: Local Backward SpMM", print_time);
        mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, U, d, d, 0.0, V, d);
        cpu_timer_destroy(&timer);

        // Step 3 & 4: Orthonormalize V in-place via Cholesky QR
        cpu_timer_init(&timer, "Step 3a: Local Gram Matrix", print_time);
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            d, d, M_local, 1.0, V, d, V, d, 0.0, Vt_R, d
        );
        cpu_timer_destroy(&timer);

        // Step 3b: AllReduce Gram Matrix (G_global = sum(G_local))
        cpu_timer_init(&timer, "Step 3b: AllReduce Gram Matrix", print_time);
        MPI_Allreduce(MPI_IN_PLACE, Vt_R, d * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cpu_timer_destroy(&timer);

        // Step 4: Cholesky factorization: G = R^T @ R
        cpu_timer_init(&timer, "Step 4a: Cholesky Factorization", print_time);
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', d, Vt_R, d);
        cpu_timer_destroy(&timer);

        // V = V @ R^{-1} (in-place solve)
        cpu_timer_init(&timer, "Step 4b: Triangular Solve (V Update)", print_time);
        cblas_dtrsm(
            CblasRowMajor, CblasRight, CblasUpper,
            CblasNoTrans, CblasNonUnit,
            M_local, d, 1.0, Vt_R, d, V, d
        );
        cpu_timer_destroy(&timer);
    }

    // Step 5a: Orthogonal rotation/sign-flip from Rayleigh-Ritz SVD preserves pairwise distances
    cpu_timer_init(&timer, "Step 5a: Local Rayleigh-Ritz SpMM", print_time);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, U, d);

    // Step 5b: AllReduce Rayleigh-Ritz Pass (D_temp = sum(U))
    cpu_timer_init(&timer, "Step 5b: AllReduce Rayleigh-Ritz Pass", print_time);
    MPI_Allreduce(MPI_IN_PLACE, U, N * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    cpu_timer_destroy(&timer);

    // Cleanup temporary buffers
    mkl_sparse_destroy(X_handle);
    mkl_free(V);
    mkl_free(Vt_R);
    mkl_free(tau);
}

/**
 * Computes memory-efficient K-nearest neighbor indices for a row-partitioned slice 
 * [N_start, N_end) of D_final (N x d) across MPI ranks.
 *
 * Parameters:
 *   N          : Total number of global samples (rows in D_final)
 *   N_local    : Number of samples responsible
 *   d          : Feature dimension (cols)
 *   D_final    : Full input matrix of shape (N, d), available on all ranks
 *   K_nbs      : Output matrix of 32-bit neighbor indices of shape (N_local, num_K_nbs) [Pre-allocated]
 *   num_K_nbs  : Number of nearest neighbors (K) to extract per sample
 */
void knn_extract_mpi(
    const MKL_INT N, const MKL_INT N_start,
    const MKL_INT N_end, const int d,
    const MKL_INT num_K_nbs,
    const double *U, MKL_INT *K_nbs
) {
    MKL_INT N_local = N_end - N_start;
    if (N_local <= 0 || N <= 0 || num_K_nbs <= 0 || N_start < 0 || N_end > N) return;

    MKL_INT actual_k = (num_K_nbs < N) ? num_K_nbs : N;

    // 1. Compute squared row norms across all global rows: sq_norms[g] = sum_k (U[g, k]^2)
    double *sq_norms = (double*)aligned_alloc_64(N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (MKL_INT g = 0; g < N; ++g) {
        double norm_sq = 0.0;
        for (int k = 0; k < d; ++k) {
            double val = U[g * d + k];
            norm_sq += val * val;
        }
        sq_norms[g] = norm_sq;
    }

    // 2. Row Tiling over N_local: Memory footprint capped to min(1024, N_local) x N
    const MKL_INT TILE_SIZE = (N_local < 1024) ? N_local : 1024;
    double *Dist_block = (double*)aligned_alloc_64(TILE_SIZE * N * sizeof(double));

    for (MKL_INT local_block_start = 0; local_block_start < N_local; local_block_start += TILE_SIZE) {
        MKL_INT B_curr = (TILE_SIZE < N_local - local_block_start) ? TILE_SIZE : (N_local - local_block_start);
        MKL_INT global_block_start = N_start + local_block_start;
        const double *U_block = U + global_block_start * d;

        // Compute sub-matrix Gram products G_block (B_curr x N) = U_block (B_curr x d) * U^T (d x N)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    B_curr, N, d,
                    1.0, U_block, d,
                    U, d,
                    0.0, Dist_block, N);

        // 3. Extract Top-K Nearest Neighbors per local row using thread-local max-heaps
        #pragma omp parallel
        {
            DistPair *heap = (DistPair*)malloc(actual_k * sizeof(DistPair));

            #pragma omp for schedule(dynamic)
            for (MKL_INT r = 0; r < B_curr; ++r) {
                MKL_INT local_i  = local_block_start + r;   // Index in [0, N_local)
                MKL_INT global_i = global_block_start + r;  // Index in [N_start, N_end)

                const double *G_row = Dist_block + r * N;
                double sq_norm_i = sq_norms[global_i];

                MKL_INT heap_size = 0;

                for (MKL_INT j = 0; j < N; ++j) {
                    double d_sq = (j == global_i) ? 0.0 : (sq_norm_i + sq_norms[j] - 2.0 * G_row[j]);
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

                // Extract sorted global neighbor indices into local output buffer (nearest first)
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
    knn_extract_mpi(N, N_start, N_end, d, num_K_nbs, D_final, K_nb);
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
