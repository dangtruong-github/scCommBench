import argparse
import struct
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import svd_flip

def load_plink_csr(prefix: str) -> sp.csr_matrix:
    """
    Reads PLINK binary files (<prefix>.bed, <prefix>.bim, <prefix>.fam) 
    and returns a scipy.sparse.csr_matrix of shape (N_samples, M_snps).
    """
    fam_path = f"{prefix}.fam"
    bim_path = f"{prefix}.bim"
    bed_path = f"{prefix}.bed"

    # 1. Count sample size N (.fam) and variant count M (.bim)
    with open(fam_path, "r") as f:
        N = sum(1 for _ in f)
    with open(bim_path, "r") as f:
        M = sum(1 for _ in f)

    bytes_per_snp = (N + 3) // 4

    # 2. Read PLINK binary BED file
    with open(bed_path, "rb") as f:
        magic = f.read(3)
        if len(magic) < 3 or magic[:2] != b"\x6c\x1b":
            raise ValueError(f"File '{bed_path}' is not a valid PLINK BED file.")
        if magic[2] != 1:
            raise ValueError("Only SNP-major mode (0x01) PLINK files are supported.")

        raw_bytes = np.fromfile(f, dtype=np.uint8)

    expected_bytes = M * bytes_per_snp
    if raw_bytes.size != expected_bytes:
        raise ValueError(
            f"BED byte count mismatch: expected {expected_bytes} bytes for "
            f"N={N}, M={M}, but read {raw_bytes.size} bytes."
        )

    # 3. Decode 2-bit genotype pairs
    # Bit layout mapping: 00 -> 0.0, 01 -> 0.0 (missing), 10 -> 1.0, 11 -> 2.0
    raw_bytes = raw_bytes.reshape(M, bytes_per_snp)
    lookup = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float64)

    s0 = lookup[raw_bytes & 3]
    s1 = lookup[(raw_bytes >> 2) & 3]
    s2 = lookup[(raw_bytes >> 4) & 3]
    s3 = lookup[(raw_bytes >> 6) & 3]

    genotypes_padded = np.empty((M, bytes_per_snp * 4), dtype=np.float64)
    genotypes_padded[:, 0::4] = s0
    genotypes_padded[:, 1::4] = s1
    genotypes_padded[:, 2::4] = s2
    genotypes_padded[:, 3::4] = s3

    # Slice actual N samples and transpose to shape (N_samples, M_snps)
    genotypes = genotypes_padded[:, :N].T
    return sp.csr_matrix(genotypes)


def read_dist_mat_bin(file_path: str):
    """
    Reads the binary distance matrix saved by C++ save_dist_mat_bin.
    
    Header (16 bytes):
      - N       : int32 (4 bytes)
      - seed    : uint32 (4 bytes)
      - d       : int32 (4 bytes)
      - n_iter  : int32 (4 bytes)
    Data:
      - Matrix  : N x N doubles (float64)
    """
    with open(file_path, "rb") as f:
        header_data = f.read(16)
        if len(header_data) < 16:
            raise ValueError(f"File '{file_path}' is too short to contain a valid header.")
            
        N, seed, d, n_iter = struct.unpack("iIii", header_data)
        
        matrix_flat = np.fromfile(f, dtype=np.float64, count=N * N)
        if matrix_flat.size != N * N:
            raise ValueError(
                f"Data size mismatch: expected {N * N} elements, but read {matrix_flat.size}."
            )
            
        dist_matrix = matrix_flat.reshape((N, N))

    return {
        "N": N,
        "seed": seed,
        "d": d,
        "n_iter": n_iter,
        "dist_matrix": dist_matrix,
    }


def block_power_iteration(X: sp.csr_matrix, z: int, n_iter: int = 20, random_state: int = 42):
    """Block Power Iteration (Subspace Iteration) for Sparse Matrices."""
    rng = np.random.RandomState(random_state)
    N, M = X.shape

    # Step 0: Initialize random V matrix (M x z) with orthonormal columns
    V_init = rng.randn(M, z)
    V, _ = np.linalg.qr(V_init)

    # Power Iteration Loop
    for _ in range(n_iter):
        U = X @ V
        W = X.T @ U
        V, _ = np.linalg.qr(W)

    # Rayleigh-Ritz Step
    Z = X @ V
    U_tilde, S, V_R_t = np.linalg.svd(Z, full_matrices=False)

    U_final = U_tilde
    V_final = V @ V_R_t.T
    Vh_final = V_final.T

    # Align signs and build Z matrix
    U_final, Vh_final = svd_flip(U_final, Vh_final)
    Z_final = U_final * S

    return Z_final


def knn_dist_mat(Z_data: np.ndarray) -> np.ndarray:
    """Computes pair-wise Euclidean distance matrix for embedding Z_data."""
    sq_norms = np.sum(Z_data**2, axis=1)
    D_numpy = np.sqrt(np.maximum(0, sq_norms[:, None] + sq_norms[None, :] - 2 * (Z_data @ Z_data.T)))
    return D_numpy


def compare_c_and_python(read_prefix: str, save_path: str, atol: float = 1e-4):
    print("=== 1. Loading C++ Binary Distance Output ===")
    cpp_output = read_dist_mat_bin(save_path)
    N = cpp_output["N"]
    seed = cpp_output["seed"]
    d = cpp_output["d"]
    n_iter = cpp_output["n_iter"]
    D_cpp = cpp_output["dist_matrix"]

    print(f"  Saved File  : {save_path}")
    print(f"  Matrix N    : {N} x {N}")
    print(f"  Hyperparams : d={d}, n_iter={n_iter}, seed={seed}\n")

    print("=== 2. Loading PLINK Input Data ===")
    print(f"  Prefix      : {read_prefix}")
    X_sparse = load_plink_csr(read_prefix)
    print(f"  Loaded Shape: {X_sparse.shape} (N_samples x M_variants)")
    print(f"  Non-zeros   : {X_sparse.nnz}\n")

    print("=== 3. Executing Python Custom Benchmark ===")
    Z_py = block_power_iteration(X_sparse, z=d, n_iter=n_iter, random_state=seed)
    D_py = knn_dist_mat(Z_py)

    print("=== 4. Verification Results ===")
    diff = np.abs(D_cpp - D_py)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    is_close = np.allclose(D_cpp, D_py, atol=atol)

    print(f"  Max Absolute Error  : {max_err:.6e}")
    print(f"  Mean Absolute Error : {mean_err:.6e}")
    print(f"  Exact Match?        : {'YES' if is_close else 'NO'} (atol={atol})\n")

    print("Top-left 3x3 Distance Matrix (C++):")
    print(np.round(D_cpp[:3, :3], 4))
    print("\nTop-left 3x3 Distance Matrix (Python):")
    print(np.round(D_py[:3, :3], 4))

    return is_close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare C++ distance matrix binary against Python PLINK benchmark.")
    parser.add_argument("-r", "--read", type=str, default="data/hgdp.qc.pruned", help="PLINK prefix (reads .bed, .bim, .fam)")
    parser.add_argument("-o", "--save", type=str, default="data/hgdp.qc.pruned.bin", help="C++ output distance matrix binary path")
    parser.add_argument("-a", "--atol", type=float, default=1e-4, help="Absolute tolerance for numpy.allclose")

    args = parser.parse_args()
    compare_c_and_python(args.read, args.save, args.atol)
