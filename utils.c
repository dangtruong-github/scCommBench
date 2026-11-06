#include "utils.h"

// Function to print a CSR matrix in dense 2D grid format
void print_csr_as_dense(
    int32_t N, int32_t M, 
    const int32_t *row_ptr, const int32_t *col_ind, const double *values,
    int32_t max_rows, int32_t max_cols
) {
    int32_t r_limit = (max_rows > 0 && max_rows < N) ? max_rows : N;
    int32_t c_limit = (max_cols > 0 && max_cols < M) ? max_cols : M;

    printf("\n=== Sparse Matrix Printed in Dense View (%lld x %lld preview of %lld x %lld) ===\n", 
           (long long)r_limit, (long long)c_limit, (long long)N, (long long)M);

    for (int32_t i = 0; i < r_limit; ++i) {
        int32_t row_start = row_ptr[i];
        int32_t row_end   = row_ptr[i + 1];
        int32_t ptr = row_start;

        printf("[");
        for (int32_t j = 0; j < c_limit; ++j) {
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
    int32_t *N, int32_t *M, int32_t *nnz,
    int32_t **row_ptr, int32_t **col_ind,
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

    *N   = (int32_t)header[0];
    *M   = (int32_t)header[1];
    *nnz = (int32_t)header[2];

    *row_ptr = (int32_t*)aligned_alloc_64(((*N) + 1) * sizeof(int32_t));
    *col_ind = (int32_t*)aligned_alloc_64((*nnz) * sizeof(int32_t));
    *values  = (double*)aligned_alloc_64((*nnz) * sizeof(double));

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

    // Convert loaded integer indices to int32_t type
    for (int32_t i = 0; i <= (*N); ++i) (*row_ptr)[i] = (int32_t)tmp_row_ptr[i];
    for (int32_t i = 0; i < (*nnz); ++i)  (*col_ind)[i] = (int32_t)tmp_col_ind[i];

    free(tmp_row_ptr);
    free(tmp_col_ind);

    return 0;
}

// Helper to count lines in a text file (.fam / .bim / .bin)
static int32_t count_file_lines(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    int32_t lines = 0;
    int ch;
    int last_ch = '\n';
    
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '\n') {
            lines++;
        }
        last_ch = ch;
    }
    
    // Handle files that don't end with a trailing newline
    if (last_ch != '\n' && lines >= 0) {
        lines++;
    }

    fclose(fp);
    return lines;
}

int load_data_plink(
    int32_t *out_N, int32_t *out_M, int32_t *out_nnz,
    int32_t **out_row_ptr, int32_t **out_col_ind,
    double **out_values, const char *prefix
) {
    char path_buf[512];

    // 1. Get N from .fam
    snprintf(path_buf, sizeof(path_buf), "%s.fam", prefix);
    int32_t N = count_file_lines(path_buf);
    if (N <= 0) {
        fprintf(stderr, "Error reading .fam file: %s\n", path_buf);
        return -1;
    }

    // 2. Get M from .bim (or fallback to .bin)
    snprintf(path_buf, sizeof(path_buf), "%s.bim", prefix);
    int32_t M = count_file_lines(path_buf);
    if (M <= 0) {
        snprintf(path_buf, sizeof(path_buf), "%s.bin", prefix);
        M = count_file_lines(path_buf);
        if (M <= 0) {
            fprintf(stderr, "Error reading .bim/.bin file: %s\n", path_buf);
            return -2;
        }
    }

    // 3. Open .bed binary file and check magic header
    snprintf(path_buf, sizeof(path_buf), "%s.bed", prefix);
    FILE *bed_fp = fopen(path_buf, "rb");
    if (!bed_fp) {
        fprintf(stderr, "Error opening .bed file: %s\n", path_buf);
        return -3;
    }

    unsigned char header[3];
    if (fread(header, 1, 3, bed_fp) != 3 ||
        header[0] != 0x6C || header[1] != 0x1B || header[2] != 0x01) {
        fprintf(stderr, "Error: Invalid PLINK BED header or not in SNP-major mode.\n");
        fclose(bed_fp);
        return -4;
    }

    size_t bytes_per_snp = (N + 3) / 4;
    unsigned char *snp_buf = (unsigned char *)malloc(bytes_per_snp);
    int32_t *row_counts = (int32_t *)calloc(N, sizeof(int32_t));
    
    if (!snp_buf || !row_counts) {
        fclose(bed_fp);
        free(snp_buf);
        free(row_counts);
        return -5;
    }

    // --- PASS 1: Count non-zeros per row ---
    int32_t total_nnz = 0;
    for (int32_t j = 0; j < M; ++j) {
        if (fread(snp_buf, 1, bytes_per_snp, bed_fp) != bytes_per_snp) {
            fprintf(stderr, "Error reading byte stream at SNP %d\n", j);
            free(snp_buf);
            free(row_counts);
            fclose(bed_fp);
            return -6;
        }

        for (int32_t i = 0; i < N; ++i) {
            uint8_t bits = (snp_buf[i / 4] >> ((i % 4) * 2)) & 0x03;
            if (bits == 0b10 || bits == 0b11) {
                row_counts[i]++;
                total_nnz++;
            }
        }
    }

    // Allocate output CSR arrays
    int32_t *row_ptr = (int32_t *)malloc((N + 1) * sizeof(int32_t));
    int32_t *col_ind = (int32_t *)malloc(total_nnz * sizeof(int32_t));
    double  *values  = (double *)malloc(total_nnz * sizeof(double));
    int32_t *curr_pos = (int32_t *)malloc(N * sizeof(int32_t));

    if (!row_ptr || !col_ind || !values || !curr_pos) {
        free(snp_buf); free(row_counts); free(curr_pos);
        free(row_ptr); free(col_ind); free(values);
        fclose(bed_fp);
        return -7;
    }

    // Construct row_ptr offsets
    row_ptr[0] = 0;
    for (int32_t i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        curr_pos[i] = row_ptr[i];
    }
    free(row_counts);

    // --- PASS 2: Populate col_ind and values ---
    fseek(bed_fp, 3, SEEK_SET); // Reset file pointer past the 3-byte header
    for (int32_t j = 0; j < M; ++j) {
        fread(snp_buf, 1, bytes_per_snp, bed_fp);

        for (int32_t i = 0; i < N; ++i) {
            uint8_t bits = (snp_buf[i / 4] >> ((i % 4) * 2)) & 0x03;
            double val = 0.0;
            
            if (bits == 0b10) val = 1.0;      // Heterozygous
            else if (bits == 0b11) val = 2.0; // Homozygous A2

            if (val > 0.0) {
                int32_t idx = curr_pos[i]++;
                col_ind[idx] = j;
                values[idx]  = val;
            }
        }
    }

    // Cleanup local buffers
    free(snp_buf);
    free(curr_pos);
    fclose(bed_fp);

    // Assign back to caller
    *out_N = N;
    *out_M = M;
    *out_nnz = total_nnz;
    *out_row_ptr = row_ptr;
    *out_col_ind = col_ind;
    *out_values = values;

    return 0;
}

int save_dist_mat_bin(
    const char *file_path, const int32_t N,
    const unsigned int seed, const int d,
    const int n_iter, const double *Dist_matrix_new
) {
    FILE *fp = fopen(file_path, "wb");
    if (fp == NULL) {
        perror("Error opening file for writing");
        // Handle error appropriately
        return 1;
    } 
    
    // 1. Write the integer N at the top of the file
    fwrite(&N, sizeof(int32_t), 1, fp);
    fwrite(&seed, sizeof(unsigned int), 1, fp);
    fwrite(&d, sizeof(int), 1, fp);
    fwrite(&n_iter, sizeof(int), 1, fp);

    // 2. Write the N x N array of doubles
    size_t total_elements = (size_t)N * N;
    size_t written = fwrite(Dist_matrix_new, sizeof(double), total_elements, fp);

    if (written != total_elements) {
        fprintf(stderr, "Warning: Expected to write %zu elements, but wrote %zu\n", total_elements, written);
        return 1;
    }

    // Close the file stream
    fclose(fp);
    printf("Successfully saved matrix to %s\n", file_path);
    return 0;
}

bool validate(
    double *dist_mat_truth, double *dist_mat,
    const int32_t M, const int32_t N,
    const double atol, const double rtol
) {
    if (dist_mat_truth == NULL || dist_mat == NULL) {
        return dist_mat_truth == dist_mat;
    }

    bool matches = true;

    for (int32_t r = 0; r < M; ++r) {
        for (int32_t c = 0; c < N; ++c) {
            int64_t i = (int64_t)r * N + c;
            double truth = dist_mat_truth[i];
            double val = dist_mat[i];

            if (truth == val) {
                continue;
            }

            if (isnan(truth) || isnan(val)) {
                if (isnan(truth) && isnan(val)) {
                    continue;
                }
                printf("Mismatch at index %lld (row %d, col %d): truth = %g, val = %g (NaN mismatch)\n",
                       (long long)i, r, c, truth, val);
                matches = false;
                continue;
            }

            double diff = fabs(truth - val);
            double max_val = fmax(fabs(truth), fabs(val));

            if (diff > atol && diff > rtol * max_val) {
                printf("Mismatch at index %lld (row %d, col %d): truth = %g, val = %g, diff = %e (atol = %e, rtol * max_val = %e)\n",
                       (long long)i, r, c, truth, val, diff, atol, rtol * max_val);
                matches = false;
            }
        }
    }

    return matches;
}

// timer
double cpu_timer_get_time(void) {
    struct timespec ts;
    // CLOCK_MONOTONIC ensures time doesn't jump backwards if system time updates
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void cpu_timer_init(CPUTimer *timer, const char *name, bool print_out) {
    if (!timer) return;
    
    timer->function_name = name;
    timer->print_out = print_out;
    timer->start_time = cpu_timer_get_time();
}

double cpu_timer_destroy(CPUTimer *timer) {
    if (!timer) return 0.0;
    
    double elapsed = cpu_timer_get_time() - timer->start_time;
    
    if (timer->print_out) {
        printf("[%s] Elapsed time: %.6f seconds\n", 
               timer->function_name ? timer->function_name : "Timer", 
               elapsed);
    }
    
    return elapsed;
}
