import boto3
import json
import requests

# DO NOT import app from anywhere - this causes circular import!
# The Flask app is defined in decentralizedalgorithm.py

ssm = boto3.client("ssm", region_name="us-east-1")
ec2 = boto3.client("ec2", region_name="us-east-1")

# Global visited set to track which nodes have been processed
visited = set()


def load_ec2_node_ips():
    """Fetch EC2 instances in the cluster using a tag"""
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:cluster", "Values": ["decentralizedalgorithmcluster"]},
            {"Name": "instance-state-name", "Values": ["running"]}
        ]
    )

    nodes = {}
    count = 0  # Start from 0 to match NODE_ID

    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            ip = instance["PrivateIpAddress"]
            nodes[count] = ip
            count += 1

    print(f"Loaded {len(nodes)} EC2 nodes: {nodes}", flush=True)
    return nodes


def load_neighbor_ips(G, node_id, neighbors):
    """
    Loads neighbors for a specific node
    Converts neighbor node IDs → IP addresses from EC2
    Propagates computation to neighbors

    Args:
        G: NetworkX graph object
        node_id: Current node ID
        neighbors: List of neighbor node IDs
    """
    global visited

    # Mark current node as visited
    if node_id in visited:
        print(f"Node {node_id} already visited → stopping propagation.", flush=True)
        return []
    visited.add(node_id)

    # Load node→IP mapping from running EC2 instances
    node_ips = load_ec2_node_ips()

    # Convert neighbor node IDs → IP addresses
    neighbor_ips = []

    for neighbor_id in neighbors:
        print(f"neighbor_id {neighbor_id}", flush=True)
        if neighbor_id in node_ips:
            neighbor_ips.append(node_ips[neighbor_id])
        else:
            print(f"Warning: Neighbor {neighbor_id} not found in EC2 nodes", flush=True)

    print(f"Node {node_id} neighbor IPs: {neighbor_ips}", flush=True)

    # Propagate to neighbors
    propagate_to_neighbors(neighbors, neighbor_ips, G)

    return neighbor_ips


def propagate_to_neighbors(neighbor_ids, neighbor_ips, G):
    """
    Send computation requests to neighbor nodes

    Args:
        neighbor_ids: List of neighbor node IDs
        neighbor_ips: List of neighbor IP addresses
        G: NetworkX graph (to pass graph structure)
    """
    for neighbor_id, neighbor_ip in zip(neighbor_ids, neighbor_ips):
        if neighbor_id in visited:
            print(f"Skipping neighbor {neighbor_id} - already visited", flush=True)
            continue

        try:
            print(f"Propagating to neighbor {neighbor_id} at {neighbor_ip}", flush=True)
            response = requests.post(
                f"http://{neighbor_ip}:5000/main",
                json={
                    "to": neighbor_id,
                    "from": 0  # First node ID
                },
                timeout=5
            )
            print(f"Successfully contacted neighbor {neighbor_id}: {response.status_code}", flush=True)
        except requests.exceptions.Timeout:
            print(f"Timeout contacting neighbor {neighbor_id} at {neighbor_ip}", flush=True)
        except requests.exceptions.ConnectionError:
            print(f"Connection error contacting neighbor {neighbor_id} at {neighbor_ip}", flush=True)
        except Exception as e:
            print(f"Failed to contact neighbor {neighbor_id} at {neighbor_ip}: {e}", flush=True)
