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
    // Default hyperparameters
    int d = 5;
    int n_iter = 10;
    unsigned int seed = 42;
    const char *read_prefix = "data/hgdp.qc.pruned";
    const char *save_path = "data/hgdp.qc.pruned.bin";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            d = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-iter") == 0) && i + 1 < argc) {
            n_iter = atoi(argv[++i]);
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
    printf("Configuration loaded:\n");
    printf("  d           : %d\n", d);
    printf("  n_iter      : %d\n", n_iter);
    printf("  seed        : %u\n", seed);
    printf("  read_prefix : %s\n", read_prefix);
    printf("  save_path   : %s\n\n", save_path);

    int32_t N, M, nnz;
    int32_t *row_ptr;
    int32_t *col_ind;
    double  *values;

    int err_data = load_data_plink(&N, &M, &nnz, &row_ptr, &col_ind, &values, read_prefix);
    if (err_data) {
        return err_data;
    }

    int run_mode_err = run_mode(
        N, M, d, n_iter, seed,
        row_ptr, col_ind, values,
        naive_seq_lib_baseline,
        "data/hgdp.qc.pruned.cpu_naive.bin"
    );

    if (run_mode_err) {
        return run_mode_err;
    }

    // 5. Memory Cleanup
    free(row_ptr);
    free(col_ind);
    free(values);

    return 0;
}
