import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd, svd_flip
from sklearn.metrics import pairwise_distances

def block_power_iteration(X: sp.csr_matrix, z: int, n_iter: int = 20, random_state: int = 42):
    """
    Block Power Iteration (Subspace Iteration) for Sparse Matrices.
    
    Parameters:
        X : sp.csr_matrix of shape (N, M)
        z : Number of principal components / singular vectors
        n_iter : Number of power iterations
        random_state : Seed for reproducibility
    """
    rng = np.random.RandomState(random_state)
    N, M = X.shape
    
    # 0. Initialize random matrix V of size M x z with orthonormal columns
    V_init = rng.randn(M, z)
    V, _ = np.linalg.qr(V_init)
    
    # Block Power Iteration Loop
    for _ in range(n_iter):
        # Step 1: Forward pass (M x z -> N x z)
        U = X @ V                  
        
        # Step 2: Backward pass (N x z -> M x z)
        W = X.T @ U                
        
        # Step 3 & 4: Orthonormalize z vectors and update V
        V, _ = np.linalg.qr(W)     
        
    # Rayleigh-Ritz Step: Extract exact singular vectors and values from subspace V
    Z = X @ V                             # Dense matrix of shape (N, z)
    U_tilde, S, V_R_t = np.linalg.svd(Z, full_matrices=False)
    
    # Reconstruct final singular vectors
    U_final = U_tilde                     # Left singular vectors (N x z)
    V_final = V @ V_R_t.T                 # Right singular vectors (M x z)
    Vh_final = V_final.T                  # (z x M)
    
    # Align signs (SVD vectors are unique up to sign flip)
    U_final, Vh_final = svd_flip(U_final, Vh_final)
    Z_final = U_final * S                 # Equivalent to: U_final * S
    
    return Z_final

def knn_dist_mat(Z_data):
    sq_norms = np.sum(Z_data**2, axis=1)
    D_numpy = np.sqrt(np.maximum(0, sq_norms[:, None] + sq_norms[None, :] - 2 * (Z_data @ Z_data.T)))

    return D_numpy

# =====================================================================
# Verification and Comparison against sklearn.utils.extmath.randomized_svd
# =====================================================================

# 1. Generate Synthetic Sparse Data (e.g., Single-Cell Expression Matrix)
np.random.seed(42)
N_cells, M_genes, z_pcs, iters = 1000, 500, 5, 1200

def load_csr_matrix(filename: str = "data/sparse_matrix.bin") -> sp.csr_matrix:
    """Reads the raw binary CSR matrix created for C/MKL back into SciPy."""
    with open(filename, "rb") as f:
        # Read header: [N, M, nnz] as 64-bit integers
        header = np.fromfile(f, dtype=np.int64, count=3)
        N, M, nnz = header[0], header[1], header[2]
        
        # Read CSR arrays sequentially
        row_ptr = np.fromfile(f, dtype=np.int64, count=N + 1)
        col_ind = np.fromfile(f, dtype=np.int64, count=nnz)
        values = np.fromfile(f, dtype=np.float64, count=nnz)
        
    return sp.csr_matrix((values, col_ind, row_ptr), shape=(N, M))

# Synthetic sparse counts (90% sparse)
raw_data = load_csr_matrix()
print(raw_data.shape)
X_sparse = sp.csr_matrix(raw_data)

# 2. Run Block Power Iteration
Z_bpi = block_power_iteration(
    X_sparse, 
    z=z_pcs, 
    n_iter=iters, 
    random_state=42
)
D_bpi = knn_dist_mat(Z_bpi)

# 3. Run scikit-learn's randomized_svd
# Note: n_oversamples=0 and power_iteration_normalizer='QR' match exact BPI steps
U_sk, S_sk, Vh_sk = randomized_svd(
    X_sparse, 
    n_components=z_pcs, 
    n_oversamples=0, 
    n_iter=iters, 
    power_iteration_normalizer='QR', 
    transpose=False,
    random_state=42
)

# Method A: Via component scaling (U * S)
Z_sk = U_sk * S_sk  # Broadcasting N x z matrix by z singular values
D_sk = pairwise_distances(Z_sk, metric='euclidean')

z_close = np.allclose(Z_bpi, Z_sk, atol=1e-4)
d_close = np.allclose(D_bpi, D_sk, atol=1e-4)

print("=== Verification Results ===")
print(f"Right Dense PCA result ?: {z_close}\n")
print(f"Right KNN Matrix result ?: {d_close}\n")

print("Top 5 Singular Values (BPI):    ", np.round(D_bpi, 4))
print("Top 5 Singular Values (scikit): ", np.round(D_sk, 4))
    