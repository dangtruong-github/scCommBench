#include "main.h"

// Function to print a CSR matrix in dense 2D grid format
void print_csr_as_dense(
    MKL_INT N, MKL_INT M, 
    const MKL_INT *row_ptr, const MKL_INT *col_ind, const double *values,
    MKL_INT max_rows, MKL_INT max_cols
) {
    MKL_INT r_limit = (max_rows > 0 && max_rows < N) ? max_rows : N;
    MKL_INT c_limit = (max_cols > 0 && max_cols < M) ? max_cols : M;

    printf("\n=== Sparse Matrix Printed in Dense View (%lld x %lld preview of %lld x %lld) ===\n", 
           (long long)r_limit, (long long)c_limit, (long long)N, (long long)M);

    for (MKL_INT i = 0; i < r_limit; ++i) {
        MKL_INT row_start = row_ptr[i];
        MKL_INT row_end   = row_ptr[i + 1];
        MKL_INT ptr = row_start;

        printf("[");
        for (MKL_INT j = 0; j < c_limit; ++j) {
            double val = 0.0;

            // Check if current column matches the stored non-zero entry
            if (ptr < row_end && col_ind[ptr] == j) {
                val = values[ptr];
                ptr++;
            }

            printf("%6.2f", val);
            if (j < c_limit - 1) printf(", ");
        }
        if (c_limit < M) printf("  ...");
        printf("]");
        if (i < r_limit) printf(",");
        printf("\n");
    }

    if (r_limit < N) {
        printf("...\n");
    }
    printf("\n");
}

int load_data(
    MKL_INT *N, MKL_INT *M, MKL_INT *nnz,
    MKL_INT **row_ptr, MKL_INT **col_ind,
    double **values, const char *file_path
) {
    // 1. Load Matrix from Binary File
    FILE *f = fopen(file_path, "rb");
    if (!f) {
        perror("Error opening sparse_matrix.bin");
        return 1;
    }

    int64_t header[3];
    if (fread(header, sizeof(int64_t), 3, f) != 3) {
        fclose(f);
        return -1;
    }

    *N   = (MKL_INT)header[0];
    *M   = (MKL_INT)header[1];
    *nnz = (MKL_INT)header[2];

    *row_ptr = (MKL_INT*)mkl_malloc(((*N) + 1) * sizeof(MKL_INT), 64);
    *col_ind = (MKL_INT*)mkl_malloc((*nnz) * sizeof(MKL_INT), 64);
    *values  = (double*)mkl_malloc((*nnz) * sizeof(double), 64);

    printf("=== Loaded Sparse CSR Matrix from File (%lld x %lld) ===\n", (long long)(*N), (long long)(*M));
    printf("Loaded %lld non-zero elements (Sparsity: %.2f%%)\n", 
           (long long)(*nnz), (1.0 - (double)(*nnz) / ((*N) * (*M))) * 100.0);

    // Temporary integer buffers for 64-bit binary conversion
    int64_t *tmp_row_ptr = (int64_t*)malloc(((*N) + 1) * sizeof(int64_t));
    int64_t *tmp_col_ind = (int64_t*)malloc((*nnz) * sizeof(int64_t));

    // Allocate 64-byte aligned CSR arrays for MKL

    if (!tmp_row_ptr || !tmp_col_ind || !*row_ptr || !*col_ind || !*values) {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(f);
        return 1;
    }

    // Read binary arrays sequentially
    if (fread(tmp_row_ptr, sizeof(int64_t), (*N) + 1, f) != (size_t)((*N) + 1) ||
        fread(tmp_col_ind, sizeof(int64_t), (*nnz), f)   != (size_t)(*nnz) ||
        fread((*values),      sizeof(double),  (*nnz), f)   != (size_t)(*nnz)) {
        fprintf(stderr, "Error reading CSR array data.\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    // Convert loaded integer indices to MKL_INT type
    for (MKL_INT i = 0; i <= (*N); ++i) (*row_ptr)[i] = (MKL_INT)tmp_row_ptr[i];
    for (MKL_INT i = 0; i < (*nnz); ++i)  (*col_ind)[i] = (MKL_INT)tmp_col_ind[i];

    free(tmp_row_ptr);
    free(tmp_col_ind);

    return 0;
}