#include "main.h"

#include <omp.h>

/**
 * Computes the N x N pairwise Euclidean distance matrix for D_final (N x d).
 *
 * Parameters:
 *   N         : Number of samples (rows)
 *   d         : Feature dimension (cols)
 *   D_final   : Input matrix of shape (N, d)
 *   Dist      : Output distance matrix of shape (N, N) [Pre-allocated]
 */
void compute_pairwise_distances(MKL_INT N, int d, const double *D_final, double *Dist) {
    // 1. Compute squared row norms: S[i] = sum_k (D[i, k]^2)
    double *sq_norms = (double*)mkl_malloc(N * sizeof(double), 64);
    
    #pragma omp parallel for
    for (MKL_INT i = 0; i < N; ++i) {
        double norm_sq = 0.0;
        for (int k = 0; k < d; ++k) {
            double val = D_final[i * d + k];
            norm_sq += val * val;
        }
        sq_norms[i] = norm_sq;
    }

    // 2. Gram Matrix G = D_final * D_final^T via Intel MKL cblas_dgemm
    //    Dist matrix temporarily stores the dot products G[i, j]
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, d,
                1.0, D_final, d,
                D_final, d,
                0.0, Dist, N);

    // 3. Convert Gram Matrix to Euclidean Distances: sqrt(max(0, S[i] + S[j] - 2*G[i, j]))
    #pragma omp parallel for collapse(2)
    for (MKL_INT i = 0; i < N; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            double g_ij = Dist[i * N + j];
            double d_sq = sq_norms[i] + sq_norms[j] - 2.0 * g_ij;
            Dist[i * N + j] = (d_sq > 0.0) ? sqrt(d_sq) : 0.0;
        }
    }

    mkl_free(sq_norms);
}