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
    const MKL_INT num_K_nbs = 100;
    MKL_INT N_start, N_end;
    get_start_end(N, rank, world_size, &N_start, &N_end);
    const MKL_INT N_local = N_end - N_start; 
    // Allocate output matrix D_final (N x d)
    
    MKL_INT *K_nb = (MKL_INT*)aligned_alloc_64((size_t)N_local * num_K_nbs * sizeof(MKL_INT));
    float block_power_time, knn_time;

    func1(
        N, M_local, M_global, rank, world_size,
        d, n_iter, num_K_nbs, row_ptr, col_ind,
        values, K_nb, &block_power_time,
        &knn_time, rank == 0, seed
    );
    printf("Total computation completed in: %.4f seconds of save_path=%s\n\n",
        block_power_time + knn_time, save_path
    );

    int save_err = save_dist_mat_bin_mpi(
        save_path, N, seed, d, n_iter,
        num_K_nbs, K_nb, rank, world_size
    );

    mkl_free(K_nb);

    return save_err;
}
