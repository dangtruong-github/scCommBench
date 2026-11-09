import argparse
import struct
import sys
import numpy as np


def read_knn_bin(file_path: str) -> dict:
    """Reads a binary KNN index matrix file and extracts header metadata and matrix data."""
    header_fmt = "iIiii"
    header_size = struct.calcsize(header_fmt)  # 28 bytes

    with open(file_path, "rb") as f:
        header_data = f.read(header_size)
        if len(header_data) < header_size:
            raise ValueError(f"File '{file_path}' is too short to contain a valid header.")

        N, seed, d, n_iter, num_K_nbs = struct.unpack(header_fmt, header_data)

        print(N)
        print(num_K_nbs, flush=True)

        matrix_flat = np.fromfile(f, dtype=np.int32, count=N * num_K_nbs)
        if matrix_flat.size != N * num_K_nbs:
            raise ValueError(
                f"Data size mismatch in '{file_path}': expected {N * num_K_nbs} elements, "
                f"but read {matrix_flat.size}."
            )

        knn_matrix = matrix_flat.reshape((N, num_K_nbs))

    return {
        "N": N,
        "seed": seed,
        "d": d,
        "n_iter": n_iter,
        "num_K_nbs": num_K_nbs,
        "knn_matrix": knn_matrix,
    }


def compare_knn_bin_files(file1_path: str, file2_path: str) -> bool:
    print("=== Comparing Binary KNN Index Files ===")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}\n")

    meta1 = read_knn_bin(file1_path)
    meta2 = read_knn_bin(file2_path)

    print("=== 1. Header Metadata Comparison ===")
    headers_match = True
    for key in ["N", "seed", "d", "n_iter", "num_K_nbs"]:
        v1, v2 = meta1[key], meta2[key]
        match = (v1 == v2)
        if not match:
            headers_match = False
        print(f"  {key:<10}: File 1 = {v1:<10} | File 2 = {v2:<10} | Match: {'YES' if match else 'NO'}")

    if meta1["N"] != meta2["N"] or meta1["num_K_nbs"] != meta2["num_K_nbs"]:
        print(
            f"\n[ERROR] Matrix dimension mismatch: "
            f"File 1 ({meta1['N']}x{meta1['num_K_nbs']}) vs File 2 ({meta2['N']}x{meta2['num_K_nbs']}). "
            f"Cannot compare matrix entries."
        )
        return False

    K1 = meta1["knn_matrix"]
    K2 = meta2["knn_matrix"]

    print("\n=== 2. Index Matrix Comparison ===")
    exact_match = np.array_equal(K1, K2)
    elem_matches = np.sum(K1 == K2)
    total_elems = K1.size
    elem_match_pct = (elem_matches / total_elems) * 100.0

    # Calculate row-wise neighborhood Jaccard overlap (set-based intersection regardless of order)
    row_overlaps = []
    for r in range(meta1["N"]):
        set1 = set(K1[r][K1[r] >= 0])  # Exclude padding (-1)
        set2 = set(K2[r][K2[r] >= 0])
        if set1 or set2:
            jaccard = len(set1 & set2) / len(set1 | set2)
            row_overlaps.append(jaccard)
    avg_jaccard = np.mean(row_overlaps) * 100.0 if row_overlaps else 100.0

    print(f"  Exact Index Matrix Match?  : {'YES' if exact_match else 'NO'}")
    print(f"  Matching Index Count       : {elem_matches} / {total_elems} ({elem_match_pct:.2f}%)")
    print(f"  Mean Neighborhood Overlap  : {avg_jaccard:.2f}% (Jaccard similarity)\n")

    k_rows = min(5, meta1["N"])
    k_cols = min(10, meta1["num_K_nbs"])

    print(f"Top {k_rows} rows x {k_cols} neighbors (File 1):")
    print(K1[:k_rows, :k_cols])
    print(f"\nTop {k_rows} rows x {k_cols} neighbors (File 2):")
    print(K2[:k_rows, :k_cols])

    return exact_match and headers_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two binary KNN index matrix output (.bin) files.")
    parser.add_argument("-1", "--file1", type=str, required=True, help="Path to first binary KNN file")
    parser.add_argument("-2", "--file2", type=str, required=True, help="Path to second binary KNN file")

    args = parser.parse_args()
    success = compare_knn_bin_files(args.file1, args.file2)
    sys.exit(0 if success else 1)
