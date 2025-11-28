<<<<<<< HEAD
=======
import numpy as np


>>>>>>> 5071e97 (loadingec2instances)
def avg_consensus(G, Z_in, T_con):
    """
    Z_in: dict node -> numpy array (or scalar)
    G: networkx graph (nodes assumed 0..L-1)
    T_con: number of consensus rounds
    returns dict node -> approx L * average(Z_in) (i.e., scaled to sum)
    """
    nodes = list(G.nodes())
    L = len(nodes)

    # degrees
    deg = {g: G.degree(g) for g in nodes}
    # Metropolis weights: W[g][j] for j neighbor of g
<<<<<<< HEAD
    W = {g: {} for g in nodes}
    for g in nodes:
        for j in G.neighbors(g):
            W[g][j] = 1.0 / (max(deg[g], deg[j]))
        # self-weight to ensure row sums to 1
        W[g][g] = 1.0 - sum(W[g].get(j, 0.0) for j in W[g] if j != g)

    # initialize values (safe copy)
    Z = {}
    for g in nodes:
        val = Z_in[g]
        try:
            Z[g] = np.array(val, copy=True)
        except Exception:
            Z[g] = val
=======
    W = {0: {}}
    # for g in nodes:
    for j in G.neighbors(0):
        W[j] = 1.0 / (max(deg[0], deg[j]))
        # self-weight to ensure row sums to 1
    G[0][0] = 1.0 - sum(W.get(j, 0.0) for j in W if j != 0)

    # initialize values (safe copy)
    Z = {}
    # for g in nodes:
    val = Z_in[0]
    try:
        Z[0] = np.array(val, copy=True)
    except Exception:
        Z[0] = val
>>>>>>> 5071e97 (loadingec2instances)

    # iterative mixing: Z <- sum_j W_gj * Zj
    for _ in range(T_con):
        Z_new = {}
<<<<<<< HEAD
        for g in nodes:
            cur = None
            for j, w in W[g].items():
                if cur is None:
                    cur = w * Z[j]
                else:
                    cur = cur + w * Z[j]
            Z_new[g] = cur
=======
        # for g in nodes:
        cur = None
        for j, w, in W[0].items():
            if cur is None:
                cur = w * Z[j]
            else:
                cur = cur + w * Z[j]
            Z_new[0] = cur
>>>>>>> 5071e97 (loadingec2instances)
        Z = Z_new

    # return scaled-by-L values (paper expects L * average)
    return {g: Z[g] * L for g in nodes}
