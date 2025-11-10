import numpy as np
import matplotlib.pyplot as plt
import time
import Subspace_distance
import Average_consensus
import matplotlib.ticker as ticker

def decentralized_altgdmin(X_k_dict, y_k_dict, S_g_dict, G, U, params, Tgd, max_diogonalvalue, U_star):
    n, r, L, T, T_con, Tgd, d = params['n'], params['r'], params['L'], params['T'], params['T_con'], \
    params['Tgd'], params['d']

    eta = []
    for g in range(L):
        eta.append(0.4 / (n * max_diogonalvalue[g - 1] ** 2))

    errors = []
    times = []
    subspace_dists = []
    t0 = time.time()
    B = np.zeros((r, T))
    c= 0

    # initial subspace distance (before any GD)
    # if 'U_star' in params:
    #     U_any = next(iter(U.values()))
    #     subspace_dists.append(subspace_distance(U_any, params['U_star']))

    for it in range(1, Tgd + 1):
        c+=1
        grad = {g: np.zeros((d, r)) for g in range(L)}
        for g in range(L):
            Ug = U[g]
            for k in S_g_dict[g]:
                X_k = X_k_dict[k]
                yk = y_k_dict[k]
                AUg = X_k.T@ Ug

                b = np.linalg.pinv(AUg) @ yk
                B[:, k] = b
                xk = Ug @ b

                resid = X_k.T @ xk - yk
                temp = (resid[:, None] @ b[None, :] )
                grad[g] = grad[g] + X_k @ temp


        # compute relative error and time


        # consensus on gradients (average)
        avg_grad = Average_consensus.avg_consensus(G, grad, T_con)

        # projected gradient update
        for g in range(L):
            ut = U[g] - eta[g-1] * avg_grad[g]
            # protect against degenerate ut shapes
            if ut.size == 0:
                continue
            U[g], _ = np.linalg.qr(ut, mode='reduced')

        err = Subspace_distance.subspace_distance(U_star, U[0])
        # if(c <= 400):
        errors.append((err))
        times.append(time.time() - t0)

    print("error at the end", errors[len(errors)-1])
    # ---- Plot (log scale y-axis for better visibility) ----
    plt.figure(figsize=(6, 4))
    plt.semilogy(errors, 'k-o', linewidth=2, markersize=3)
    plt.title(
        "Subspace Distance vs Iteration count\n 'p': .5, 'n': 50, 'r': 2, 'T': 600, 'L': 20, 'T_pm': 10, 'T_con': 100, \n 'd': 600, 'Tgd' : 1400")
    plt.xlabel("Iteration count")
    plt.ylabel("Subspace Distance")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.savefig('subspace_distance.png')
    # Second plot: Subspace Distance vs Execution Time (log scale)
    plt.figure(figsize=(6, 4))
    plt.semilogy(times, errors, 'k-o', linewidth=2, markersize=3)
    plt.title(
        "Subspace Distance vs Execution Time\n p: .5 'n': 50, 'r': 2, 'T': 600, 'L': 20, 'T_pm': 10, 'T_con' : 100, \n 'd': 600, 'Tgd': 1400")
    plt.xlabel("Execution Time (s)")
    plt.ylabel("Subspace Distance")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.savefig('subspace_distance_execution_time.png')

    return U, errors, times, subspace_dists, B

