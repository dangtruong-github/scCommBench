import argparse
import struct
import sys
import numpy as np


def read_dist_mat_bin(file_path: str) -> dict:
    """Reads a binary distance matrix file and extracts header metadata and matrix data."""
    with open(file_path, "rb") as f:
        header_data = f.read(16)
        if len(header_data) < 16:
            raise ValueError(f"File '{file_path}' is too short to contain a valid header.")

        N, seed, d, n_iter = struct.unpack("iIii", header_data)

        matrix_flat = np.fromfile(f, dtype=np.float64, count=N * N)
        if matrix_flat.size != N * N:
            raise ValueError(
                f"Data size mismatch in '{file_path}': expected {N * N} elements, "
                f"but read {matrix_flat.size}."
            )

        dist_matrix = matrix_flat.reshape((N, N))

    return {
        "N": N,
        "seed": seed,
        "d": d,
        "n_iter": n_iter,
        "dist_matrix": dist_matrix,
    }


def compare_bin_files(file1_path: str, file2_path: str, atol: float = 1e-4, rtol: float = 1e-5) -> bool:
    print("=== Comparing Binary Distance Files ===")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}\n")

    meta1 = read_dist_mat_bin(file1_path)
    meta2 = read_dist_mat_bin(file2_path)

    print("=== 1. Header Metadata Comparison ===")
    headers_match = True
    for key in ["N", "seed", "d", "n_iter"]:
        v1, v2 = meta1[key], meta2[key]
        match = (v1 == v2)
        if not match:
            headers_match = False
        print(f"  {key:<8}: File 1 = {v1:<10} | File 2 = {v2:<10} | Match: {'YES' if match else 'NO'}")

    if meta1["N"] != meta2["N"]:
        print(f"\n[ERROR] Matrix dimension mismatch ({meta1['N']} vs {meta2['N']}). Cannot compare matrix entries.")
        return False

    D1 = meta1["dist_matrix"]
    D2 = meta2["dist_matrix"]

    print("\n=== 2. Numerical Values Comparison ===")
    diff = np.abs(D1 - D2)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    is_close = np.allclose(D1, D2, atol=atol, rtol=rtol)

    print(f"  Max Absolute Error  : {max_err:.6e}")
    print(f"  Mean Absolute Error : {mean_err:.6e}")
    print(f"  Matrices Match?     : {'YES' if is_close else 'NO'} (atol={atol}, rtol={rtol})\n")

    k = min(3, meta1["N"])
    print(f"Top-left {k}x{k} Distance Matrix (File 1):")
    print(np.round(D1[:k, :k], 4))
    print(f"\nTop-left {k}x{k} Distance Matrix (File 2):")
    print(np.round(D2[:k, :k], 4))

    return is_close and headers_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two binary distance matrix output (.bin) files.")
    parser.add_argument("-1", "--file1", type=str, required=True, help="Path to first binary file")
    parser.add_argument("-2", "--file2", type=str, required=True, help="Path to second binary file")
    parser.add_argument("-a", "--atol", type=float, default=1e-4, help="Absolute tolerance for numpy.allclose")
    parser.add_argument("-r", "--rtol", type=float, default=1e-5, help="Relative tolerance for numpy.allclose")

    args = parser.parse_args()
    success = compare_bin_files(args.file1, args.file2, atol=args.atol, rtol=args.rtol)
    sys.exit(0 if success else 1)
