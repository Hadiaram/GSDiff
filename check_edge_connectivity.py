"""
Check edge connectivity for files 3 and 4 to diagnose why they show 0 cycles.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/user/GSDiff')

from datasets.path_utils import get_data_path

for file_num in [3, 4]:
    print(f"\n{'='*60}")
    print(f"Checking file {file_num}.npy")
    print(f"{'='*60}")

    test_file = get_data_path('rplang-v3-withsemantics', 'test', f'{file_num}.npy')
    data = np.load(test_file, allow_pickle=True)
    graph = data.item()

    padding_mask = graph['padding_mask']
    adjacency_matrix = graph['adjacency_matrix_np_padding']

    num_real_corners = int(padding_mask.sum())
    print(f"Number of real corners: {num_real_corners}")

    # Get adjacency for real corners only
    adj_real = adjacency_matrix[:num_real_corners, :num_real_corners]

    print(f"Total edges: {adj_real.sum()}")
    print(f"Connections per corner: {adj_real.sum(axis=1)}")

    # Check for disconnected corners
    disconnected = np.where(adj_real.sum(axis=1) == 0)[0]
    if len(disconnected) > 0:
        print(f"WARNING: {len(disconnected)} disconnected corners: {disconnected}")

    # Check for corners with only 1 connection
    single_connection = np.where(adj_real.sum(axis=1) == 1)[0]
    if len(single_connection) > 0:
        print(f"WARNING: {len(single_connection)} corners with only 1 connection: {single_connection}")

    # Show edge list
    print("\nEdge list:")
    edges = []
    for i in range(num_real_corners):
        for j in range(i+1, num_real_corners):
            if adj_real[i, j] == 1:
                edges.append((i, j))

    print(f"Total unique edges: {len(edges)}")
    if len(edges) <= 20:
        for edge in edges:
            print(f"  {edge[0]} -- {edge[1]}")
    else:
        print(f"  (showing first 20)")
        for edge in edges[:20]:
            print(f"  {edge[0]} -- {edge[1]}")

    # Check if graph is connected
    # Simple connectivity check using BFS
    visited = set()
    queue = [0]
    visited.add(0)

    while queue:
        node = queue.pop(0)
        neighbors = np.where(adj_real[node] == 1)[0]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    if len(visited) == num_real_corners:
        print(f"✓ Graph is fully connected")
    else:
        print(f"✗ Graph is NOT connected: {len(visited)}/{num_real_corners} nodes reachable")
        unreachable = set(range(num_real_corners)) - visited
        print(f"  Unreachable nodes: {sorted(unreachable)}")
