#include "main.h"

#include <omp.h>

/**
 * Sign alignment (svd_flip) matching scikit-learn behavior:
 * Forces max absolute entry in each column of U to be positive.
 */
#include <math.h>
#include <stddef.h>
#include <mkl.h>

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
    double *D_final, unsigned int random_state
) {
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
    double *V        = (double*)mkl_malloc(M * d * sizeof(double), 64);
    double *U        = (double*)mkl_malloc(N * d * sizeof(double), 64);
    double *W        = (double*)mkl_malloc(M * d * sizeof(double), 64);
    double *D_temp   = (double*)mkl_malloc(N * d * sizeof(double), 64);
    double *Vt_R     = (double*)mkl_malloc(d * d * sizeof(double), 64);
    double *tau      = (double*)mkl_malloc(d * sizeof(double), 64);
    double *superb   = (double*)mkl_malloc(d * sizeof(double), 64);
    double *U_final  = (double*)mkl_malloc(N * d * sizeof(double), 64);
    double *S        = (double*)mkl_malloc(d * sizeof(double), 64);
    double *Vh_final = (double*)mkl_malloc(d * M * sizeof(double), 64);

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
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, V, d, d, 0.0, U, d);

        // Step 2: Backward pass W = X.T @ U (M x d)
        mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, X_handle, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, U, d, d, 0.0, W, d);

        // Step 3 & 4: Orthonormalize d vectors and update V
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, d, W, d, tau);
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, d, d, W, d, tau);
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
