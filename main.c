#include "main.h"

void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -d <int>          Dimensions / z_pcs (default: 5)\n");
    printf("  -n, --n-iter <int> Number of iterations (default: 10)\n");
    printf("  -s, --seed <int>   Random seed (default: 42)\n");
    printf("  -r, --read <str>   Read prefix path (default: \"data/hgdp.qc.pruned\")\n");
    printf("  -o, --save <str>   Save path (default: \"data/hgdp.qc.pruned.bin\")\n");
    printf("  -h, --help         Display this help menu\n");
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Win win;
    MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);    

    // Default hyperparameters
    int d = 5;
    int n_iter = 1000;
    int num_K_nbs = 100;
    unsigned int seed = 42;
    const char *read_prefix = "data/hgdp.qc.pruned";
    const char *save_path = "data/hgdp.qc.pruned.bin";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            d = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-iter") == 0) && i + 1 < argc) {
            n_iter = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--num-k-nbs") == 0) && i + 1 < argc) {
            num_K_nbs = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--read") == 0) && i + 1 < argc) {
            read_prefix = argv[++i];
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--save") == 0) && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Your main logic continues below:
    if (rank == 0) {
        printf("Configuration loaded:\n");
        printf("  d           : %d\n", d);
        printf("  n_iter      : %d\n", n_iter);
        printf("  num_K_nbs   : %d\n", num_K_nbs);
        printf("  seed        : %u\n", seed);
        printf("  read_prefix : %s\n", read_prefix);
        printf("  save_path   : %s\n\n", save_path);
    }

    MKL_INT N, M_global, M_local, nnz;
    MKL_INT *row_ptr;
    MKL_INT *col_ind;
    double  *values;
    
    double load_data_time, block_power_time, knn_time, save_time;

    int err_data = load_plink_partition_mpi(
        read_prefix, rank, world_size,
        &load_data_time, &N, &M_local,
        &M_global, &nnz, &row_ptr,
        &col_ind, &values, rank == 0
    );

    if (err_data) {
        return err_data;
    }

    int run_mode_err = run_mode(
        N, M_local, M_global, rank,
        world_size, d, n_iter, seed,
        num_K_nbs, row_ptr, col_ind,
        values, mpi_impl, save_path,
        &block_power_time, &knn_time,
        &save_time
    );

    if (run_mode_err) {
        return run_mode_err;
    }

    // 5. Memory Cleanup
    free(row_ptr);
    free(col_ind);
    free(values);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    {
        printf("Timing of rank %d\n", rank);
        printf("Load data time %.4f\n", load_data_time);
        printf("Block power iter computation time: %.4f\n", block_power_time);
        printf("KNN indices computation time: %.4f\n", knn_time);
        printf("Total computation completed in: %.4f seconds\n", block_power_time + knn_time);
        printf("Save data time: %.4f seconds\n", save_time);
        printf("-------------------\n");
    }
    MPI_Win_unlock(0, win);

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
