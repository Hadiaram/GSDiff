"""
Fix CNN feature map files by creating placeholder features.
This creates the expected structure for prerunning_cnn_featuremaps files.
"""
import os
import numpy as np
from tqdm import tqdm

# Path to the feature maps directory
feature_dir = 'datasets/prerunning_cnn_featuremaps'

# Get all .npy files
files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]

print(f"Found {len(files)} files to fix...")

for filename in tqdm(files):
    filepath = os.path.join(feature_dir, filename)
    
    # Create placeholder CNN feature with expected structure
    # Shape (1024, 16, 16) as expected by the model
    feat = {
        16: [np.random.randn(1024, 16, 16).astype(np.float32)]
    }
    
    # Save with the correct structure
    np.save(filepath, feat)

print(f"\n✓ Fixed {len(files)} CNN feature map files")
print("⚠️  WARNING: These are random placeholder features for testing only.")
print("   For actual results, generate real CNN features using prerunningCNN.py")
