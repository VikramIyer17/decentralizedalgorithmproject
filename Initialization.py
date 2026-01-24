import Average_consensus
import numpy as np
import matplotlib.pyplot as plt
import math
import Subspace_distance
import dbhelper

def initialization_distributed(node_id, X_k_dict, y_k_dict, S_g_dict, G, params, U_star, neighbors):
    """
    Distributed initialization - each node runs this independently

    Args:
        node_id: Current node's ID
        neighbors: List of neighbor node IDs from the graph
        (other params same as before)
    """
    n, T, r, L = params['n'], params['T'], params['r'], params['L']
    T_pm, T_con = params['T_pm'], params['T_con']
    C_tilde = params.get('C_tilde', 9.0)

    print(f"Node {node_id}: Starting initialization...", flush=True)

    # ===== STEP 1: Calculate local alpha and run consensus =====
    total = 0.0
    for k in S_g_dict[node_id]:  # Only this node's data
        total += float(np.sum(y_k_dict[k] ** 2))
    alpha_local = C_tilde * total / (n * T)

    print(f"Node {node_id}: Local alpha = {alpha_local}", flush=True)

    # Store local alpha
    dbhelper.store_u_matrix(node_id, iteration=-1, U_matrix=np.array([alpha_local]))

    # Collect all nodes' alphas for consensus
    all_nodes = list(G.nodes())
    alpha_dict = {}
    for other_node in all_nodes:
        if other_node == node_id:
            alpha_dict[node_id] = alpha_local
        else:
            alpha_array = dbhelper.load_u_matrix(other_node, iteration=-1)
            alpha_dict[other_node] = alpha_array[0]  # Extract scalar from array

    # Run average consensus
    alpha_consensus_dict = Average_consensus.avg_consensus(G, alpha_dict, T_con)
    alpha_cons = alpha_consensus_dict[node_id] # Divide by L since avg_consensus returns L * average

    print(f"Node {node_id}: Consensus alpha = {alpha_cons}", flush=True)

    # ===== STEP 2: Truncate y (local operation) =====
    y_trunc = {}
    total_truncated = 0
    thr = math.sqrt(alpha_cons)

    for k in S_g_dict[node_id]:
        y = y_k_dict[k]
        truncated_values = np.abs(y) > thr
        y_trunc[k] = np.where(truncated_values, 0.0, y)
        total_truncated += np.sum(truncated_values)

    print(f"Node {node_id}: Truncated {total_truncated} values", flush=True)

    # ===== STEP 3: Build local partial sum =====
    mats = []
    for k in S_g_dict[node_id]:
        temp = X_k_dict[k] @ y_trunc[k]  # Use y_trunc instead of y_k_dict
        mats.append((1.0 / params['n']) * temp)

    X0_local = np.column_stack(mats) if mats else np.zeros((params['d'], 0))

    # ===== STEP 4: Initialize U (same seed = same initial U) =====
    np.random.seed(params.get('seed_init', 1))
    U_int = np.random.randn(params['d'], params['r'])
    U_init, _ = np.linalg.qr(U_int, mode='reduced')
    U_local = U_init.copy()

    print(f"Node {node_id}: Starting power method iterations...", flush=True)

    # ===== STEP 5: Distributed Power Method =====
    subspace_errors = []

    for t in range(T_pm):
        print(f"Node {node_id}: Iteration {t + 1}/{T_pm}", flush=True)

        # LOCAL UPDATE
        local_update = X0_local @ (X0_local.T @ U_local)

        # STORE LOCAL UPDATE
        dbhelper.store_u_matrix(node_id, iteration=t, U_matrix=local_update)

        # COLLECT ALL NODES' UPDATES for consensus
        updates_dict = {}
        for other_node in all_nodes:
            if other_node == node_id:
                updates_dict[node_id] = local_update
            else:
                updates_dict[other_node] = dbhelper.load_u_matrix(other_node, iteration=t)

        # RUN AVERAGE CONSENSUS on updates
        consensus_dict = Average_consensus.avg_consensus(G, updates_dict, T_con)
        averaged_update = consensus_dict[node_id]  # Divide by L

        # QR DECOMPOSITION
        U_new, R = np.linalg.qr(averaged_update, mode='reduced')

        if node_id == 0:  # Only node 0 keeps the result
            U_local = U_new
            dbhelper.store_u_matrix(node_id, iteration=f"{t}_final", U_matrix=U_local)
        else:
            U_local = np.zeros_like(U_init)
            dbhelper.store_u_matrix(node_id, iteration=f"{t}_final", U_matrix=U_local)

        # SECOND CONSENSUS to broadcast node 0's U
        U_dict = {}
        for other_node in all_nodes:
            if other_node == node_id:
                U_dict[node_id] = U_local
            else:
                U_dict[other_node] = dbhelper.load_u_matrix(other_node, iteration=f"{t}_final")

        # RUN AVERAGE CONSENSUS on final U
        U_consensus_dict = Average_consensus.avg_consensus(G, U_dict, T_con)
        U_consensus = U_consensus_dict[node_id] / L  # Divide by L

        # Update local U
        U_local = U_consensus

        # Calculate subspace distance (for monitoring)
        if U_star is not None:
            dist = Subspace_distance.subspace_distance(U_star, U_consensus)
            subspace_errors.append(dist)
            print(f"Node {node_id} - Iter {t + 1}: subspace distance = {dist:.6f}", flush=True)

    print(f"Node {node_id}: Initialization complete", flush=True)

    # Return final U and max diagonal value
    max_diagonalvalue = [1.0]  # Placeholder - calculate if needed

    return {node_id: U_local}, max_diagonalvalue