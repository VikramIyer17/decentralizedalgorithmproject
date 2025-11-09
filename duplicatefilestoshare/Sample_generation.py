from scipy.linalg import orth

def generate_low_rank_matrix(d, T, r, seed):
    # np.random.seed(seed)
    U_star = orth(np.random.randn(d, r))  # n x r orthonormal
    print(U_star.shape)
    B = np.random.randn(r, T)
    _, sval_B, _ = np.linalg.svd(B)
    maxsval_B = np.max(sval_B)
    print("Singular value of B", maxsval_B)
    print(B.shape)

    theta_star = U_star @ B  # n x T
    return theta_star, U_star



import numpy as np


def generate_xtandyt(n, T, d, theta_star, seed):
    # np.random.seed(seed)
    # Generate X_t as d x n matrix
    Xt = {t: np.random.randn(d, n) for t in range(T)}

    # Compute Y_t = X_t * Theta_star
    # (n x d) @ (d x T) = (n x T)
    Yt = {k: Xt[k].T @ theta_star[:, k] for k in range(T)}

    return Xt, Yt


def projected_columns(A_list, theta_star):
    return [A @ theta_star[:, k] for k, A in enumerate(A_list)]