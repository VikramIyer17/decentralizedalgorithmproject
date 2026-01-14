import numpy as np
import networkx as nx
from flask import Flask, request, jsonify
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
import Average_consensus

# Import only the functions, not the app
from loadingec2instances import load_neighbor_ips

# Define Flask app HERE (not in loadingec2instances)
app = Flask(__name__)


# ---------------------------
#  Main demo driver (improved plotting)
# ---------------------------
@app.route("/main", methods=["POST"])
def main():
    # Node 0 always uses ID 0, ignores request data
    NODE_ID = 0

    # Log if this was triggered by another node (optional)
    try:
        data = request.json or {}
        sender_id = data.get("from", "manual_trigger")
        print(f"Node {NODE_ID} received trigger from {sender_id}", flush=True)
    except:
        print(f"Node {NODE_ID} starting computation", flush=True)

    np.random.seed(0)
    quick_run = True  # set True to test quickly on small sizes
    if quick_run:
        params = {'n': 50, 'r': 2, 'T': 30, 'L': 4, 'T_pm': 50, 'T_con': 100, 'seed_init': 1, 'print_every': 40,
                  'd': 600, 'Tgd': 1400}
    else:
        params = {'n': 600, 'm': 200, 'r': 2, 'T': 600, 'L': 20, 'T_pm': 200, 'T_con': 2000, 'seed_init': 1,
                  'print_every': 100, 'd': 50, 'Tgd': 500}

    # data
    print(f"Node {NODE_ID}: Generating data...", flush=True)
    theta_star, U_star = Sample_generation.generate_low_rank_matrix(params['d'], params['T'], params['r'], seed=0)
    Xt, Yt = Sample_generation.generate_xtandyt(params['n'], params['T'], params['d'], theta_star, seed=0)

    idx = np.arange(0, params['T'])

    # Create graph structure
    print(f"Node {NODE_ID}: Creating graph structure...", flush=True)
    G = nx.erdos_renyi_graph(params['L'], .5, seed=0)

    # Ensure graph is connected
    if not nx.is_connected(G):
        print(f"Warning: Graph is not connected. {params['L']} nodes, edges = {G.number_of_edges()}", flush=True)

    # Partition data across nodes
    np.random.shuffle(idx)
    parts = np.array_split(idx, params['L'])
    S_g = {g: list(parts[g]) for g in range(params['L'])}
    print(f"Node {NODE_ID} data partition: {S_g[NODE_ID]}", flush=True)

    # Get neighbors for this node
    neighbors = list(G.neighbors(NODE_ID))
    print(f"Node {NODE_ID} neighbors: {neighbors}", flush=True)

    # Load neighbor IPs and propagate to them
    print(f"Node {NODE_ID}: Loading neighbor IPs and propagating...", flush=True)
    neighbor_ips = load_neighbor_ips(G, NODE_ID, neighbors)

    # Initialize
    print(f"Node {NODE_ID}: Initializing...", flush=True)
    U_init_dict, max_diagonalvalue = Initialization.initialization(Xt, Yt, S_g, G, params, U_star)

    # Run decentralized algorithm
    print(f"Node {NODE_ID}: Starting AltGDmin ...", flush=True)
    U_final, errors, times, subspace_dists, Bk = Alternating_gradientdescent.decentralized_altgdmin(
        Xt, Yt, S_g, G, U_init_dict, params, params['Tgd'], max_diagonalvalue, U_star
    )

    print(f"Node {NODE_ID}: Completed computation", flush=True)

    return jsonify({"status": "success", "node_id": NODE_ID})


if __name__ == "__main__":
    print("Starting Node 0 Flask server...", flush=True)
    app.run(host="0.0.0.0", port=5000)