#include "main.h"

/*

int compare_mode(
    const MKL_INT N, const MKL_INT M, const int d,
    const int n_iter, const unsigned int seed,
    const MKL_INT *row_ptr, const MKL_INT *col_ind,
    const double *values, algo_func_t func1, algo_func_t func2
) {
    // Allocate output matrix D_final (N x d)
    double *Dist_matrix = (double*)aligned_alloc_64((size_t)N * N * sizeof(double));
    float block_power_time, knn_time;

    func1(
        N, M, d, n_iter, row_ptr, col_ind,
        values, Dist_matrix, &block_power_time,
        &knn_time, true, seed
    );
    printf("Total computation completed in: %.4f seconds\n\n", block_power_time + knn_time);


    double *Dist_matrix_new = (double*)aligned_alloc_64((size_t)N * N * sizeof(double));
    func2(
        N, M, d, n_iter, row_ptr, col_ind,
        values, Dist_matrix_new, &block_power_time,
        &knn_time, true, seed
    );
    printf("Total computation completed in: %.4f seconds\n\n", block_power_time + knn_time);

    // 4. Print Results Summary
    printf("Top 5x5 Submatrix of Pairwise Distances:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Row %d: [", i);
        for (int j = 0; j < 5; ++j) {
            printf("%8.8f%s", Dist_matrix[i * N + j], (j == 4) ? "" : ", ");
        }
        printf("]\n");
    }
    
    bool compare = validate(Dist_matrix, Dist_matrix_new, M, N, 1e-4, 1e-4);
    printf("CPU equals to GPU: %d\n", compare);

    // 5. Memory Cleanup
    mkl_free(Dist_matrix);
    mkl_free(Dist_matrix_new);

    return 0;
}
*/


int run_mode(
    const MKL_INT N, const MKL_INT M_local,
    const MKL_INT M_global, const int rank,
    const int world_size, const int d, const int n_iter,
    const unsigned int seed, const MKL_INT *row_ptr,
    const MKL_INT *col_ind, const double *values,
    algo_func_t func1, const char *save_path
) {
    // Allocate output matrix D_final (N x d)
    double *Dist_matrix = (double*)aligned_alloc_64((size_t)N * N * sizeof(double));
    float block_power_time, knn_time;

    func1(
        N, M_local, M_global, rank, world_size,
        d, n_iter, row_ptr, col_ind, values,
        Dist_matrix, &block_power_time,
        &knn_time, true, seed
    );
    printf("Total computation completed in: %.4f seconds of save_path=%s\n\n",
        block_power_time + knn_time, save_path
    );

    if (rank != 0) {
        mkl_free(Dist_matrix);
        return 0;
    }

    int save_err = save_dist_mat_bin(save_path, N, seed, d, n_iter, Dist_matrix);

    mkl_free(Dist_matrix);

    return save_err;
}
