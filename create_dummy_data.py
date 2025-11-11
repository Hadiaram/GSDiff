"""
DEPRECATED: This script is deprecated and should not be used.

Create minimal dummy data for testing GSDiff without the full dataset.
This creates synthetic data that matches the expected format.

WARNING: This file is deprecated and may be removed in future versions.
Use the actual dataset instead of dummy data.
"""
import os
import numpy as np
from tqdm import tqdm
import warnings

# Issue deprecation warning
warnings.warn(
    "create_dummy_data.py is deprecated and will be removed in a future version. "
    "Please use the actual dataset instead of dummy data.",
    DeprecationWarning,
    stacklevel=2
)

# Create directories
os.makedirs('datasets/rplang-v3-withsemantics/test', exist_ok=True)
os.makedirs('datasets/rplang-v3-withsemantics-withboundary/test', exist_ok=True)
os.makedirs('datasets/prerunning_cnn_featuremaps', exist_ok=True)

print("Creating dummy test data...")

# Create a small number of dummy samples (e.g., 10 samples for quick testing)
num_samples = 10

for i in tqdm(range(num_samples)):
    # Create dummy graph data
    # Based on the code, the graph should have:
    # - corner_list_np_normalized_padding_withsemantics: (53, 16) normalized coordinates and semantics
    # - padding_mask: (53, 1) uint8
    # - global_matrix_np_padding: (53, 53) bool adjacency matrix
    # - edges: (2809, 1) edge list (53*53 = 2809)
    
    num_real_corners = np.random.randint(10, 40)  # Random number of real corners
    
    # Create global attention matrix (True for valid corners, False for padding)
    global_matrix = np.zeros((53, 53), dtype=bool)
    global_matrix[:num_real_corners, :num_real_corners] = True  # Only valid corners have attention
    
    # Create edges array - should have num_real_corners^2 elements when filtered by global_matrix
    edges_full = np.random.randint(0, 2, (2809, 1)).astype(np.float32)
    
    # Create graph for rplang-v3-withsemantics (used by RPlanGEdgeSemanSimplified)
    graph_withsemantics = {
        'corner_list_np_normalized_padding_withsemantics': np.random.randn(53, 16).astype(np.float32),
        'padding_mask': np.ones((53, 1), dtype=np.uint8),
        'global_matrix_np_padding': global_matrix,
        'edges': edges_full
    }
    graph_withsemantics['padding_mask'][num_real_corners:] = 0
    
    # Create graph for rplang-v3-withsemantics-withboundary (used by RPlanGEdgeSemanSimplified_81)
    graph_withboundary = {
        'corner_list_np_normalized_padding_withsemantics': np.random.randn(53, 16).astype(np.float32),
        'padding_mask': np.ones((53, 1), dtype=np.uint8)
    }
    graph_withboundary['padding_mask'][num_real_corners:] = 0
    
    # Save graph data
    filename = f"{i}.npy"
    np.save(f'datasets/rplang-v3-withsemantics/test/{filename}', graph_withsemantics)
    np.save(f'datasets/rplang-v3-withsemantics-withboundary/test/{filename}', graph_withboundary)
    
    # Create dummy CNN feature map
    # Based on the code: featmap_16 should be (1024, 16, 16)
    feat = {
        16: [np.random.randn(1024, 16, 16).astype(np.float32)]
    }
    np.save(f'datasets/prerunning_cnn_featuremaps/{filename}', feat)

print(f"\n✓ Dummy data created successfully!")
print(f"  - Created {num_samples} samples in 'datasets/rplang-v3-withsemantics/test/'")
print(f"  - Created {num_samples} samples in 'datasets/rplang-v3-withsemantics-withboundary/test/'")
print(f"  - Created {num_samples} feature maps in 'datasets/prerunning_cnn_featuremaps/'")
print("\n⚠️  IMPORTANT: This is synthetic data for testing code functionality only.")
print("   Results will be meaningless. Use real dataset for actual experiments.")
print("\nYou can now run the test script with this dummy data.")
