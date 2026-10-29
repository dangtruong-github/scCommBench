import numpy as np
import scipy.sparse as sp

# 1. Setup dimensions & parameters
N, M = 1000, 500
density = 0.10
seed = 42

np.random.seed(seed)

# 2. Generate synthetic sparse matrix X
raw_data = np.random.randn(N, M)
mask = np.random.rand(N, M) <= density
raw_data[~mask] = 0.0

X_sparse = sp.csr_matrix(raw_data)

# 3. Cast CSR components to explicit standard 64-bit C types
N_rows, M_cols = X_sparse.shape
nnz = X_sparse.nnz
row_ptr = X_sparse.indptr.astype(np.int64)
col_ind = X_sparse.indices.astype(np.int64)
values = X_sparse.data.astype(np.float64)

# 4. Save metadata + arrays into a single binary file
filename = "data/sparse_matrix.bin"
with open(filename, "wb") as f:
    # Header: [N, M, nnz] as 64-bit integers
    np.array([N_rows, M_cols, nnz], dtype=np.int64).tofile(f)
    # CSR Arrays
    row_ptr.tofile(f)
    col_ind.tofile(f)
    values.tofile(f)

print(f"Successfully exported sparse matrix to '{filename}'")
print(f"Dimensions: {N_rows} x {M_cols} | NNZ: {nnz}")
