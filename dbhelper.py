import boto3
import json
import numpy as np
import time

# Use S3 for simple key-value storage
s3 = boto3.client('s3', region_name='us-east-1')
BUCKET_NAME = 'decentralized-algorithm-data'  # Create this bucket in AWS


def store_u_matrix(node_id, iteration, U_matrix):
    """Store U matrix for a node at specific iteration"""
    key = f"U/node_{node_id}/iter_{iteration}.npy"

    # Convert numpy array to bytes
    import io
    buffer = io.BytesIO()
    np.save(buffer, U_matrix)
    buffer.seek(0)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=buffer.getvalue()
    )
    print(f"Stored U for node {node_id} at iteration {iteration}", flush=True)


def load_u_matrix(node_id, iteration, max_retries=30, retry_delay=2):
    """Load U matrix for a node at specific iteration (with retries)"""
    key = f"U/node_{node_id}/iter_{iteration}.npy"

    for attempt in range(max_retries):
        try:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)

            # Convert bytes back to numpy array
            import io
            buffer = io.BytesIO(response['Body'].read())
            U_matrix = np.load(buffer)

            print(f"Loaded U for node {node_id} at iteration {iteration}", flush=True)
            return U_matrix

        except s3.exceptions.NoSuchKey:
            if attempt < max_retries - 1:
                print(f"Waiting for node {node_id} data (attempt {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(retry_delay)
            else:
                raise Exception(f"Node {node_id} data not available after {max_retries} attempts")


def fetch_neighbor_u_matrices(node_id, neighbors, iteration):
    """Fetch U matrices from all neighbors at specific iteration"""
    neighbor_Us = {}

    for neighbor_id in neighbors:
        U = load_u_matrix(neighbor_id, iteration)
        neighbor_Us[neighbor_id] = U

    return neighbor_Us


def clear_iteration_data(iteration):
    """Clear all node data for a specific iteration (cleanup)"""
    # List and delete all objects with prefix
    prefix = f"U/iter_{iteration}/"
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    if 'Contents' in response:
        objects = [{'Key': obj['Key']} for obj in response['Contents']]
        s3.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': objects})