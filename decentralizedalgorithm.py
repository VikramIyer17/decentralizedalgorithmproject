import numpy as np
import networkx as nx
from scipy.linalg import orth
import math
import pickle
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import Sample_generation
import Subspace_distance
import Alternating_gradientdescent
import Initialization

import loadingec2instances
import Average_consensus





# ---------------------------
#  Main demo driver (improved plotting)
# ---------------------------
def main():
    node = 0
    np.random.seed(0)
    quick_run = True  # set True to test quickly on small sizes
    if quick_run:
        params = {'n': 50,'r': 2, 'T': 30, 'L': 4, 'T_pm': 50, 'T_con': 100, 'seed_init': 1, 'print_every': 40,
                  'd': 600, 'Tgd':  1400}

    else:
        params = {'n': 600, 'm': 200, 'r': 2, 'T': 600, 'L': 20, 'T_pm': 200, 'T_con': 2000, 'seed_init': 1,
                  'print_every': 100, 'd': 50, 'Tgd': 500}

    # data
    theta_star, U_star = Sample_generation.generate_low_rank_matrix(params['d'], params['T'], params['r'], seed=0)
    Xt, Yt = Sample_generation.generate_xtandyt(params['n'], params['T'], params['d'], theta_star, seed=0)


    idx = np.arange(0, params['T'])
    G = nx.path_graph(params['L'])
    np.random.shuffle(idx)
    parts = np.array_split(idx, params['L'])
    S_g = {g: list(parts[g]) for g in range(params['L'])}
    print("sg", S_g[0])

    # np.random.shuffle(idx)
    # parts = np.array_split(idx, params['L'])
    S_g = idx
    # print("sg", S_g[0])
    print("before")

    G = nx.erdos_renyi_graph(params['L'], .5, seed=0)
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(f"Path Graph with {params['L']} Nodes")
    plt.show()
    neighbors = list(G.neighbors(0))
    loadingec2instances.load_ec2_node_ips()
    loadingec2instances.load_neighbor_ips(neighbors)

    if not nx.is_connected(G):
        print(f"Graph: {params['L']} nodes, edges = {G.number_of_edges()}")

    U_init_dict, max_diagonalvalue = Initialization.initialization(Xt, Yt, S_g, G, params, U_star)

    # run
    print("Starting AltGDmin ...")
    U_final, errors, times, subspace_dists, Bk = Alternating_gradientdescent.decentralized_altgdmin(Xt, Yt, S_g, G, U_init_dict, params, params['Tgd'], max_diagonalvalue, U_star)



if __name__ == "__main__":
    main()
