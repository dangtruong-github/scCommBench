#include "utils.h"

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

// Helper to count lines in a text file (.fam / .bim / .bin)
static MKL_INT count_file_lines(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    MKL_INT lines = 0;
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
    MKL_INT *out_N, MKL_INT *out_M, MKL_INT *out_nnz,
    MKL_INT **out_row_ptr, MKL_INT **out_col_ind,
    double **out_values, const char *prefix
) {
    char path_buf[512];

    // 1. Get N from .fam
    snprintf(path_buf, sizeof(path_buf), "%s.fam", prefix);
    MKL_INT N = count_file_lines(path_buf);
    if (N <= 0) {
        fprintf(stderr, "Error reading .fam file: %s\n", path_buf);
        return -1;
    }

    // 2. Get M from .bim (or fallback to .bin)
    snprintf(path_buf, sizeof(path_buf), "%s.bim", prefix);
    MKL_INT M = count_file_lines(path_buf);
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
    MKL_INT *row_counts = (MKL_INT *)calloc(N, sizeof(MKL_INT));
    
    if (!snp_buf || !row_counts) {
        fclose(bed_fp);
        free(snp_buf);
        free(row_counts);
        return -5;
    }

    // --- PASS 1: Count non-zeros per row ---
    MKL_INT total_nnz = 0;
    for (MKL_INT j = 0; j < M; ++j) {
        if (fread(snp_buf, 1, bytes_per_snp, bed_fp) != bytes_per_snp) {
            fprintf(stderr, "Error reading byte stream at SNP %d\n", j);
            free(snp_buf);
            free(row_counts);
            fclose(bed_fp);
            return -6;
        }

        for (MKL_INT i = 0; i < N; ++i) {
            uint8_t bits = (snp_buf[i / 4] >> ((i % 4) * 2)) & 0x03;
            if (bits == 0b10 || bits == 0b11) {
                row_counts[i]++;
                total_nnz++;
            }
        }
    }

    // Allocate output CSR arrays
    MKL_INT *row_ptr = (MKL_INT *)malloc((N + 1) * sizeof(MKL_INT));
    MKL_INT *col_ind = (MKL_INT *)malloc(total_nnz * sizeof(MKL_INT));
    double  *values  = (double *)malloc(total_nnz * sizeof(double));
    MKL_INT *curr_pos = (MKL_INT *)malloc(N * sizeof(MKL_INT));

    if (!row_ptr || !col_ind || !values || !curr_pos) {
        free(snp_buf); free(row_counts); free(curr_pos);
        free(row_ptr); free(col_ind); free(values);
        fclose(bed_fp);
        return -7;
    }

    // Construct row_ptr offsets
    row_ptr[0] = 0;
    for (MKL_INT i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        curr_pos[i] = row_ptr[i];
    }
    free(row_counts);

    // --- PASS 2: Populate col_ind and values ---
    fseek(bed_fp, 3, SEEK_SET); // Reset file pointer past the 3-byte header
    for (MKL_INT j = 0; j < M; ++j) {
        int fread_err = fread(snp_buf, 1, bytes_per_snp, bed_fp);

        if (fread_err) {
            printf("Error %d: Read file error", fread_err);
            return fread_err;
        }

        for (MKL_INT i = 0; i < N; ++i) {
            uint8_t bits = (snp_buf[i / 4] >> ((i % 4) * 2)) & 0x03;
            double val = 0.0;
            
            if (bits == 0b10) val = 1.0;      // Heterozygous
            else if (bits == 0b11) val = 2.0; // Homozygous A2

            if (val > 0.0) {
                MKL_INT idx = curr_pos[i]++;
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

int load_plink_partition_mpi(
    const char *prefix, int rank, int world_size,
    MKL_INT *out_N, MKL_INT *out_M_local, MKL_INT *out_M_glob,
    MKL_INT *out_nnz, MKL_INT **out_row_ptr,
    MKL_INT **out_col_ind, double **out_values
) {
    char path_buf[512];
    MKL_INT N = 0, M_global = 0;

    // --- STEP 1: Parse global N and M on Rank 0 ---
    if (rank == 0) {
        snprintf(path_buf, sizeof(path_buf), "%s.fam", prefix);
        N = count_file_lines(path_buf);

        snprintf(path_buf, sizeof(path_buf), "%s.bim", prefix);
        M_global = count_file_lines(path_buf);
    }

    // Broadcast global dimensions to all ranks
    MPI_Bcast(&N, 1, MPI_MKL_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M_global, 1, MPI_MKL_INT, 0, MPI_COMM_WORLD);

    if (N <= 0 || M_global <= 0) return -1;

    // --- STEP 2: Compute Column Range per Node ---
    MKL_INT base_M = M_global / world_size;
    MKL_INT rem    = M_global % world_size;

    MKL_INT M_local = base_M + (rank < rem ? 1 : 0);
    MKL_INT j_start = rank * base_M + (rank < rem ? rank : rem);

    size_t bytes_per_snp = (N + 3) / 4;
    off_t  byte_offset   = 3 + (off_t)j_start * bytes_per_snp;

    // --- STEP 3: Open .bed and Read Local Buffer ---
    snprintf(path_buf, sizeof(path_buf), "%s.bed", prefix);
    int fd = open(path_buf, O_RDONLY);
    if (fd < 0) return -2;

    // Verify 3-byte magic header
    unsigned char header[3];
    if (pread(fd, header, 3, 0) != 3 ||
        header[0] != 0x6C || header[1] != 0x1B || header[2] != 0x01) {
        close(fd);
        return -3;
    }

    size_t local_bytes = (size_t)M_local * bytes_per_snp;
    unsigned char *raw_buf = (unsigned char *)malloc(local_bytes);
    if (!raw_buf) { close(fd); return -4; }

    // Direct offset read (thread-safe, avoids fseek seek-race)
    if (pread(fd, raw_buf, local_bytes, byte_offset) != (ssize_t)local_bytes) {
        free(raw_buf);
        close(fd);
        return -5;
    }
    close(fd);

    // --- STEP 4: Build Local CSR Matrix (Single-Pass in RAM) ---
    MKL_INT *row_counts = (MKL_INT *)calloc(N, sizeof(MKL_INT));
    MKL_INT total_nnz = 0;

    // Pass 1 (In Memory): Count non-zeros per sample row
    for (MKL_INT j = 0; j < M_local; ++j) {
        const unsigned char *snp_ptr = raw_buf + (j * bytes_per_snp);
        for (MKL_INT i = 0; i < N; ++i) {
            uint8_t bits = (snp_ptr[i / 4] >> ((i % 4) * 2)) & 0x03;
            if (bits == 0b10 || bits == 0b11) {
                row_counts[i]++;
                total_nnz++;
            }
        }
    }

    // Allocate output CSR arrays
    MKL_INT *row_ptr  = (MKL_INT *)malloc((N + 1) * sizeof(MKL_INT));
    MKL_INT *col_ind  = (MKL_INT *)malloc(total_nnz * sizeof(MKL_INT));
    double  *values   = (double *)malloc(total_nnz * sizeof(double));
    MKL_INT *curr_pos = (MKL_INT *)malloc(N * sizeof(MKL_INT));

    row_ptr[0] = 0;
    for (MKL_INT i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        curr_pos[i] = row_ptr[i];
    }
    free(row_counts);

    // Pass 2 (In Memory): Populate local column indices [0 .. M_local - 1]
    for (MKL_INT j = 0; j < M_local; ++j) {
        const unsigned char *snp_ptr = raw_buf + (j * bytes_per_snp);
        for (MKL_INT i = 0; i < N; ++i) {
            uint8_t bits = (snp_ptr[i / 4] >> ((i % 4) * 2)) & 0x03;
            double val = 0.0;
            if (bits == 0b10) val = 1.0;
            else if (bits == 0b11) val = 2.0;

            if (val > 0.0) {
                MKL_INT idx = curr_pos[i]++;
                col_ind[idx] = j; // Local column index [0 .. M_local - 1]
                values[idx]  = val;
            }
        }
    }

    free(raw_buf);
    free(curr_pos);

    *out_N = N;
    *out_M_glob = M_global;
    *out_M_local = M_local;
    *out_nnz = total_nnz;
    *out_row_ptr = row_ptr;
    *out_col_ind = col_ind;
    *out_values = values;

    return 0;
}

int save_dist_mat_bin(
    const char *file_path, const MKL_INT N,
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
    fwrite(&N, sizeof(MKL_INT), 1, fp);
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

int save_dist_mat_bin_mpi(
    const char *file_path, const MKL_INT N,
    const unsigned int seed, const int d,
    const int n_iter, const double *Dist_local,
    const int rank, const int world_size
) {
    MKL_INT N_start, N_end;
    get_start_end(N, rank, world_size, &N_start, &N_end);

    MKL_INT N_local = N_end - N_start;
    size_t header_size = sizeof(MKL_INT) + sizeof(unsigned int) + 2 * sizeof(int);

    // 1. Rank 0 creates/truncates file and writes binary header
    if (rank == 0) {
        int fd = open(file_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            perror("Rank 0 failed to create binary output file");
            return 1;
        }

        char header_buf[128];
        char *ptr = header_buf;
        memcpy(ptr, &N, sizeof(MKL_INT));            ptr += sizeof(MKL_INT);
        memcpy(ptr, &seed, sizeof(unsigned int));    ptr += sizeof(unsigned int);
        memcpy(ptr, &d, sizeof(int));                ptr += sizeof(int);
        memcpy(ptr, &n_iter, sizeof(int));

        if (write(fd, header_buf, header_size) != (ssize_t)header_size) {
            perror("Error writing header");
            close(fd);
            return 1;
        }
        close(fd);
    }

    // Wait until Rank 0 finishes creating the file and writing the header
    MPI_Barrier(MPI_COMM_WORLD);

    // 2. Open file for parallel writing on all ranks
    int fd = open(file_path, O_WRONLY);
    if (fd < 0) {
        fprintf(stderr, "Rank %d failed opening file for pwrite\n", rank);
        return 1;
    }

    // 3. Compute target file offset and perform offset-based write
    off_t target_offset = (off_t)header_size + ((off_t)N_start * N * sizeof(double));
    size_t total_bytes  = (size_t)N_local * N * sizeof(double);

    size_t bytes_written = 0;
    const char *byte_ptr = (const char *)Dist_local;

    while (bytes_written < total_bytes) {
        ssize_t ret = pwrite(
            fd, byte_ptr + bytes_written,
            total_bytes - bytes_written,
            target_offset + (off_t)bytes_written
        );

        if (ret <= 0) {
            fprintf(stderr, "Rank %d error writing payload at offset %ld\n", rank, target_offset);
            close(fd);
            return 1;
        }
        bytes_written += ret;
    }

    close(fd);

    if (rank == 0) {
        printf("Successfully saved distributed matrix via POSIX pwrite to %s\n", file_path);
    }
    return 0;
}

bool validate(
    double *dist_mat_truth, double *dist_mat,
    const MKL_INT M, const MKL_INT N,
    const double atol, const double rtol
) {
    if (dist_mat_truth == NULL || dist_mat == NULL) {
        return dist_mat_truth == dist_mat;
    }

    bool matches = true;

    for (MKL_INT r = 0; r < M; ++r) {
        for (MKL_INT c = 0; c < N; ++c) {
            uint64_t i = (uint64_t)r * N + c;
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

void get_start_end(MKL_INT N, int rank, int world_size, MKL_INT *N_start, MKL_INT *N_end) {
    if (!N_start || !N_end) return;
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        *N_start = 0;
        *N_end = 0;
        return;
    }

    MKL_INT base_N = N / world_size;
    MKL_INT rem    = N % world_size;

    // Distribute remainder 1 by 1 to the first 'rem' ranks
    MKL_INT local_count = base_N + (rank < rem ? 1 : 0);
    MKL_INT start       = rank * base_N + (rank < rem ? rank : rem);

    *N_start = start;
    *N_end   = start + local_count;
}
