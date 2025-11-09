import Average_consensus
import numpy as np
import matplotlib.pyplot as plt
import math
import Subspace_distance
def initialization(X_k_dict, y_k_dict, S_g_dict, G, params, U_star):
    n, T, r, L = params['n'], params['T'], params['r'], params['L']
    T_pm, T_con = params['T_pm'], params['T_con']
    C_tilde = params.get('C_tilde', 9.0)
    max_diagonalvalue = 0
    alpha = {}
    total1 = 0
    for g in range(L):
        total = 0.0
        for k in S_g_dict[g]:
            total += float(np.sum(y_k_dict[k] ** 2))
        alpha[g] = C_tilde * total / (n * T)
        total1 += alpha[g]
    print("total", total1)
    # 2) consensus on alpha
    alpha_cons = Average_consensus.avg_consensus(G, alpha, T_con)
    print("alpha_cons", alpha_cons)
    # alpha = {g: float(alpha_cons[g]) for g in G.nodes()}

    # 3) truncate y
    y_trunc = {}
    total_truncated = 0
    for g in range(L):
        thr = math.sqrt(alpha_cons[g])
        for k in S_g_dict[g]:
            y = y_k_dict[k]
            # Create truncated array: values > thr replaced with 0.0
            truncated_values = np.abs(y) > thr
            y_trunc[k] = np.where(truncated_values, 0.0, y)
            # Count how many values were truncated for this k and accumulate
            total_truncated += np.sum(truncated_values)
    print("Total truncated values:", total_truncated)

    # 4) build local partial sum
    X0 = {}

    for g in range(L):
        mats = []
        for k in S_g_dict[g]:
            temp = X_k_dict[k] @ y_k_dict[k]
            mats.append((1.0 / params['n']) * temp)
            print("temp", temp.shape)
        X0[g] = np.column_stack(mats) if mats else np.zeros((n, 0))
    # 5) common random init
    # np.random.seed(params.get('seed_init', 1))
    # U_init_tilde = np.random.randn(n, r)
    U_int = np.random.randn(params['d'], params['r'])
    U_init, _ = np.linalg.qr(U_int, mode='reduced')

    U = {g: U_init.copy() for g in range(L)}

    subspace_errors = []  # store subspace distances over iterations
    i = 0
    for t in range(T_pm):
        # Step 1: local input
        updates = {g: X0[g] @ (X0[g].T @ U[g]) for g in range(L)}
        # rnkg = np.linalg.matrix_rank(updates[0])
        # print("rank", rnkg)
        # Step 2: decentralized averaging
        cons = Average_consensus.avg_consensus(G, updates, T_con)
        if (i > 0):
            print("1st average consensus", np.linalg.norm(cons[0] - cons[1], 'fro'))
        # Step 3: Node 1 does QR, others prepare zeros
        U_new = {}
        max_diagonalvalue = []
        for g in range(L):
            U_new[g], R = np.linalg.qr(cons[g], mode='reduced')
            s_val_Bg = np.sqrt(np.max(np.linalg.diagonal(R)))
            max_diagonalvalue.append(s_val_Bg)
            # print("s_val_Bg of node", g, s_val_Bg)
            if g != 0:
                U_new[g] = np.zeros_like(U[0])


        U_cons = Average_consensus.avg_consensus(G, U_new, T_con)
        Ucons = U_cons[0]


        dist = Subspace_distance.subspace_distance(U_star, U_cons[0])
        dist2 = Subspace_distance.subspace_distance(U_star, U_cons[1])
        dist3 = Subspace_distance.subspace_distance(U_star, U_cons[2])
        dist4 = Subspace_distance.subspace_distance(U_star, U_cons[3])

        subspace_errors.append(dist)

        print("After - ", np.linalg.norm(U_new[1] - U_cons[1], 'fro'))

        print(f"Iter {t + 1}/{T_pm}: subspace distance = {dist:.6f}")
        print(f"Iter {t + 1}/{T_pm}: subspace distance = {dist2:.6f}")
        print(f"Iter {t + 1}/{T_pm}: subspace distance = {dist3:.6f}")
        print(f"Iter {t + 1}/{T_pm}: subspace distance = {dist4:.6f}")

        i = i + 1
        # update U for next iteration
        for g in range(L):
            U[g] = U_cons[g]
    print("Subspace errors:", subspace_errors)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, T_pm + 1), subspace_errors, marker='o')
    plt.title("Subspace Distance vs TPM Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Subspace Distance")
    plt.grid(True)
    plt.show()


    # # print("initializtion difference", subspace_distance(U[0], U_cons[0]))
    # for g in range(L):
    #     U[g] = U_cons[g]
    print(updates == X0[0] @ (X0[0].T @ U[0]))
    return U, max_diagonalvalue

