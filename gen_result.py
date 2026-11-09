import argparse
import struct
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import svd_flip


def load_plink_csr(prefix: str, N_limit: int = None) -> sp.csr_matrix:
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

    genotypes = genotypes_padded[:, :N].T

    if N_limit is not None and N_limit < N:
        genotypes = genotypes[:N_limit, :]

    return sp.csr_matrix(genotypes)


def block_power_iteration(X: sp.csr_matrix, z: int, n_iter: int = 20, random_state: int = 42) -> np.ndarray:
    """Block Power Iteration (Subspace Iteration) for Sparse Matrices."""
    rng = np.random.RandomState(random_state)
    N, M = X.shape

    V_init = rng.randn(M, z)
    V, _ = np.linalg.qr(V_init)

    for _ in range(n_iter):
        U = X @ V
        W = X.T @ U
        V, _ = np.linalg.qr(W)

    Z = X @ V
    U_tilde, S, V_R_t = np.linalg.svd(Z, full_matrices=False)

    U_final = U_tilde
    V_final = V @ V_R_t.T
    Vh_final = V_final.T

    U_final, Vh_final = svd_flip(U_final, Vh_final)
    Z_final = U_final * S

    return Z_final


def extract_knn_indices(Z_data: np.ndarray, num_K_nbs: int) -> np.ndarray:
    """
    Computes K-nearest neighbor indices matrix (N x num_K_nbs) for embedding Z_data.
    Indices per row are sorted ascending by distance and cast to int32.
    """
    N = Z_data.shape[0]
    actual_k = min(num_K_nbs, N)

    # 1. Compute pairwise squared Euclidean distances
    sq_norms = np.sum(Z_data**2, axis=1)
    D_sq = np.maximum(0.0, sq_norms[:, None] + sq_norms[None, :] - 2.0 * (Z_data @ Z_data.T))

    # 2. Extract top-K smallest values per row in O(N) using argpartition
    top_k_partition = np.argpartition(D_sq, actual_k - 1, axis=1)[:, :actual_k]

    # 3. Sort partition indices ascending by distance
    row_indices = np.arange(N)[:, None]
    partition_dists = D_sq[row_indices, top_k_partition]
    sorted_order = np.argsort(partition_dists, axis=1)

    K_nbs = top_k_partition[row_indices, sorted_order]

    # 4. Pad with -1 if num_K_nbs exceeds sample size N
    if num_K_nbs > actual_k:
        padding = np.full((N, num_K_nbs - actual_k), -1, dtype=np.int32)
        K_nbs = np.hstack([K_nbs, padding])

    return K_nbs.astype(np.int32)


def save_knn_bin(file_path: str, N: int, seed: int, d: int, n_iter: int, num_K_nbs: int, K_nbs: np.ndarray):
    """
    Saves the K-nearest neighbor indices matrix into binary format with 32-bit integers.
    Header (20 bytes): N (int32), seed (uint32), d (int32), n_iter (int32), num_K_nbs (int32)
    Data: N x num_K_nbs int32 matrix
    """
    header = struct.pack("iIiii", N, seed, d, n_iter, num_K_nbs)
    with open(file_path, "wb") as f:
        f.write(header)
        K_nbs.astype(np.int32).tofile(f)


def generate_and_save_knn_matrix(
    prefix: str, output_path: str, d: int, n_iter: int, seed: int, num_K_nbs: int, N_override: int = None
):
    print("=== 1. Loading PLINK Input Data ===")
    X_sparse = load_plink_csr(prefix, N_limit=N_override)
    N, M = X_sparse.shape
    print(f"  Prefix      : {prefix}")
    print(f"  Shape (N,M) : ({N}, {M})")
    print(f"  Non-zeros   : {X_sparse.nnz}\n")

    print("=== 2. Running Block Power Iteration & Extracting K-Nearest Neighbors ===")
    print(f"  Parameters  : d={d}, n_iter={n_iter}, seed={seed}, K={num_K_nbs}")
    Z_py = block_power_iteration(X_sparse, z=d, n_iter=n_iter, random_state=seed)
    K_nbs_py = extract_knn_indices(Z_py, num_K_nbs=num_K_nbs)

    print("\n=== 3. Saving Binary Output Matrix ===")
    save_knn_bin(output_path, N=N, seed=seed, d=d, n_iter=n_iter, num_K_nbs=num_K_nbs, K_nbs=K_nbs_py)
    print(f"  Successfully saved binary K-nearest neighbor indices (int32) to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save binary KNN index matrix from PLINK files.")
    parser.add_argument("-r", "--read", type=str, default="data/hgdp.qc.pruned", help="PLINK prefix (reads .bed, .bim, .fam)")
    parser.add_argument("-o", "--save", type=str, default="data/knn_out.bin", help="Output path for binary KNN index matrix")
    parser.add_argument("-d", "--dim", type=int, default=5, help="Embedding dimension d")
    parser.add_argument("-n", "--n-iters", type=int, default=1000, help="Number of power iterations")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-k", "--k-neighbors", type=int, default=100, help="Number of nearest neighbors (K) to extract")
    parser.add_argument("-N", "--num-samples", type=int, default=None, help="Optional sample limit N")

    args = parser.parse_args()
    generate_and_save_knn_matrix(
        prefix=args.read,
        output_path=args.save,
        d=args.dim,
        n_iter=args.n_iters,
        seed=args.seed,
        num_K_nbs=args.k_neighbors,
        N_override=args.num_samples,
    )