#include "main.h"

int main() {
    // Hyperparameters
    const int d = 5;                // z_pcs
    const int n_iter = 1200;        // iterations
    const unsigned int seed = 42;

    MKL_INT N, M, nnz;
    MKL_INT *row_ptr;
    MKL_INT *col_ind;
    double  *values;

    int err_data = load_data(&N, &M, &nnz, &row_ptr, &col_ind, &values, "data/sparse_matrix.bin");
    if (err_data != 0) {
        return 0;
    } 

    // Allocate output matrix D_final (N x d)
    double *D_final = (double*)mkl_malloc(N * d * sizeof(double), 64);

    // 2. Execute Block Power Iteration
    printf("\n=== Running Block Power Iteration (%d iterations, d=%d) ===\n", n_iter, d);
    double start_time = dsecnd();

    block_power_iter(N, M, d, n_iter, row_ptr, col_ind, values, D_final, seed);

    double end_time = dsecnd();
    printf("Execution completed in: %.4f seconds\n\n", end_time - start_time);

    // 3. Compute N x N Distance Matrix
    printf("=== Computing Pairwise Distance Matrix (%lld x %lld) ===\n", (long long)N, (long long)N);
    double *Dist_matrix = (double*)mkl_malloc(N * N * sizeof(double), 64);

    double dist_start = dsecnd();
    compute_pairwise_distances(N, d, D_final, Dist_matrix);
    double dist_end = dsecnd();

    printf("Distance matrix computation completed in: %.4f seconds\n\n", dist_end - dist_start);

    // 4. Print Results Summary
    printf("Top 5x5 Submatrix of Pairwise Distances:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Row %d: [", i);
        for (int j = 0; j < 5; ++j) {
            printf("%8.8f%s", Dist_matrix[i * N + j], (j == 4) ? "" : ", ");
        }
        printf("]\n");
    }

    // 5. Memory Cleanup
    mkl_free(row_ptr);
    mkl_free(col_ind);
    mkl_free(values);
    mkl_free(D_final);
    mkl_free(Dist_matrix);

    return 0;
}
