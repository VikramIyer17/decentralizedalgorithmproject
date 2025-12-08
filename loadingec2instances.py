import boto3
import json
import requests
from flask import  request, jsonify
from decentralizedalgorithm import app


ssm = boto3.client("ssm", region_name="us-east-1")
ec2 = boto3.client("ec2", region_name="us-east-1")

# Read my node name from a file
NODE_ID = 0


def load_ec2_node_ips():
    """Fetch EC2 instances in the cluster using a tag"""
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:cluster", "Values": ["decentralizedalgorithmcluster"]},
            # {"Name": "instance-state-name", "Values": ["running"]}
        ]
    )

    nodes = {}
    count = 1

    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            ip = instance["PrivateIpAddress"]
            node_name = count
            nodes[node_name] = ip
            count += 1
    print("nodes ",nodes, flush=True)
    return nodes


def load_neighbor_ips(G):
    """
    Loads neighbors from /cluster/graph_neighbors
    Converts neighbor node names → IP addresses from EC2
    """
    # 1. load node→IP mapping from running EC2 instances
    node_ips = load_ec2_node_ips()

    # 2. load graph structure from SSM
    # res = ssm.get_parameter(Name="/cluster/graph_neighbors")
    # graph = json.loads(res["Parameter"]["Value"])

    # 3. find neighbors of this node (node names)
    neighbor_nodes = G
    print("G ", G, flush=True)
    # 4. convert neighbor names → IP addresses
    neighbor_ips = [node_ips[n] for n in neighbor_nodes]
    print("neighbor ips  ",neighbor_ips, flush=True)
    visit(neighbor_ips)
    return neighbor_ips


visited = set()


# @app.route("/visit", methods=["POST"])
def visit(neighbor_ips):
    global visited

    data = request.json
    sender = data["from"]

    print(f"{NODE_ID} visited from {sender}")

    # Already visited? Skip
    if NODE_ID in visited:
        return jsonify({"status": "already_visited"})

    visited.add(NODE_ID)

    # BFS propagate
    neighbors = neighbor_ips

    for neighbor_ip in neighbors:
        try:
            requests.post(
                f"http://{neighbor_ip}:5000/main",
                json={"to": neighbor_ip},
                timeout=1
            )
        except Exception as e:
            print(f"Failed to contact {neighbor_ip}: {e}")

    return jsonify({"status": "ok"})


# @app.route("/start_bfs", methods=["POST"])
def start_bfs():
    return visit()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
