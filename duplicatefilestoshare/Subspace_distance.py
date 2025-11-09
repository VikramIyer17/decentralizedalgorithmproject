import numpy as np
def subspace_distance(U_star, U):
    P = np.identity(U_star.shape[0])
    P_star = U @ U.T
    temp = (P - P_star) @ U_star
    return np.linalg.norm(temp, 2)