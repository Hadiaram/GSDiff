"""
Inspect the test data structure to understand edge connectivity.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/user/GSDiff')

from datasets.path_utils import get_data_path

# Load test file 0
test_file = get_data_path('rplang-v3-withsemantics', 'test', '0.npy')
print(f"Loading: {test_file}\n")

data = np.load(test_file, allow_pickle=True)
graph = data.item()

print("Keys in graph:", list(graph.keys()))
print()

# Check corner data
corners_with_semantics = graph['corner_list_np_normalized_padding_withsemantics']
padding_mask = graph['padding_mask']
edges = graph['edges']
global_matrix = graph['global_matrix_np_padding']
adjacency_matrix = graph.get('adjacency_matrix_np_padding', None)

num_real_corners = int(padding_mask.sum())
print(f"Number of real corners (non-padded): {num_real_corners}")
print(f"Corners shape: {corners_with_semantics.shape}")
print(f"Padding mask shape: {padding_mask.shape}")
print(f"Edges shape: {edges.shape}")
print(f"Global matrix shape: {global_matrix.shape}")
if adjacency_matrix is not None:
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
print()

# Show first few real corners
print("First 5 corners (coords + semantics):")
for i in range(min(5, num_real_corners)):
    coords = corners_with_semantics[i, :2]
    semantics = corners_with_semantics[i, 2:]
    print(f"  Corner {i}: coords={coords}, semantics={semantics}")
print()

# Check edge structure
edges_flat = edges.squeeze()
print(f"Edges (flattened): shape={edges_flat.shape}")
print(f"Number of 1s in edges: {np.sum(edges_flat == 1)}")
print(f"Number of 0s in edges: {np.sum(edges_flat == 0)}")
print()

# Reshape edges to matrix form if possible
if edges_flat.shape[0] == 53 * 53:
    edges_matrix = edges_flat.reshape(53, 53)
    print("Edges as matrix (53x53):")
    print(f"  Real corner connections (first {num_real_corners}x{num_real_corners}):")
    edges_real = edges_matrix[:num_real_corners, :num_real_corners]
    print(f"    Sum: {edges_real.sum()}")
    print(f"    Connections per corner: {edges_real.sum(axis=1)[:10]}")
    print()

# Check adjacency matrix if it exists
if adjacency_matrix is not None:
    adj_real = adjacency_matrix[:num_real_corners, :num_real_corners]
    print(f"Adjacency matrix (first {num_real_corners}x{num_real_corners}):")
    print(f"  Sum: {adj_real.sum()}")
    print(f"  Connections per corner: {adj_real.sum(axis=1)[:10]}")
    print()

    # Show which corners are connected
    print("Corner connections (from adjacency matrix):")
    for i in range(min(10, num_real_corners)):
        connected_to = np.where(adj_real[i] == 1)[0]
        if len(connected_to) > 0:
            print(f"  Corner {i} connected to: {connected_to}")
    print()

# Check global attention matrix
global_real = global_matrix[:num_real_corners, :num_real_corners]
print(f"Global attention matrix (first {num_real_corners}x{num_real_corners}):")
print(f"  Sum: {global_real.sum()}")
print(f"  All-to-all? {np.all(global_real == 1)}")
