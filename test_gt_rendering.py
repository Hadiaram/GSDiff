"""
Simple script to test ground truth PNG rendering from your NPY files.
This verifies that the data is correct and PNGs are not empty.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

# Add project to path
sys.path.insert(0, '/home/user/GSDiff')

from datasets.rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from gsdiff.utils import *

# Settings
resolution = 512
aa_scale = 1
batch_size_test = 10

# Output directory
output_dir = 'test_gt_outputs'
os.makedirs(output_dir, exist_ok=True)

# Load test dataset
print("Loading test dataset...")
dataset_test = RPlanGEdgeSemanSimplified('test')
print(f"Found {len(dataset_test)} test files")

dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0)

# Room colors
colors = {
    6: (0, 0, 0),         # Black (walls/boundaries)
    0: (244, 241, 222),   # Beige
    1: (234, 182, 159),   # Peach
    2: (107, 112, 92),    # Olive
    3: (224, 122, 95),    # Coral
    4: (95, 121, 123),    # Teal
    5: (242, 204, 143)    # Yellow
}

# Process each file
test_count = 0
for batch_idx, (corners_withsemantics_batch, global_attn_matrix_batch, corners_padding_mask_batch, edges_batch) in enumerate(dataloader_test):
    print(f"\nProcessing batch {batch_idx + 1}/{len(dataloader_test)}")

    for i in range(corners_withsemantics_batch.shape[0]):
        corners_withsemantics = corners_withsemantics_batch[i][None, :, :]
        global_attn_matrix = global_attn_matrix_batch[i][None, :, :]
        corners_padding_mask = corners_padding_mask_batch[i][None, :, :]
        edges = edges_batch[i][None, :, :]

        # Convert to numpy
        corners_withsemantics = corners_withsemantics.clamp(-1, 1).cpu().numpy()
        corners = (corners_withsemantics[0, :, :2] * (resolution // 2) + (resolution // 2)).astype(int)
        semantics = corners_withsemantics[0, :, 2:].astype(int)
        global_attn_matrix = global_attn_matrix.cpu().numpy()
        corners_padding_mask = corners_padding_mask.cpu().numpy()
        edges = edges.cpu().numpy()

        # Remove padding
        corners_depadded = corners[corners_padding_mask.squeeze() == 1][None, :, :]
        semantics_depadded = semantics[corners_padding_mask.squeeze() == 1][None, :, :]
        edges_depadded = edges[global_attn_matrix.reshape(1, -1, 1)][None, :, None]
        edges_depadded = np.concatenate((1 - edges_depadded, edges_depadded), axis=2)

        print(f"  File {test_count}: {len(corners_depadded[0])} corners after depadding")
        print(f"    Semantics shape: {semantics_depadded.shape}")
        print(f"    Semantics unique values: {np.unique(semantics_depadded)}")
        print(f"    Binary semantics: {np.all(np.isin(semantics_depadded, [0, 1]))}")

        # Transform semantics for cycle detection
        semantics_transform = semantics_depadded
        semantics_transform_indices = np.indices(semantics_transform.shape)[-1]
        semantics_transform = np.where(semantics_transform == 1, semantics_transform_indices, 99999)

        # Create points list with semantics
        points = [tuple(corner_with_seman) for corner_with_seman in
                 np.concatenate((corners_depadded, semantics_transform), axis=-1).tolist()[0]]

        # Get edges coordinates
        edges_coords = edges_to_coordinates(
            np.triu(edges_depadded[0, :, 1].reshape(len(points), len(points))).reshape(-1),
            points)

        print(f"    Points: {len(points)}, Edges: {len(edges_coords)}")

        # Get planar cycles
        try:
            d_rev, simple_cycles, simple_cycles_semantics = get_cycle_basis_and_semantic_3_semansimplified(
                points, edges_coords)

            print(f"    Found {len(simple_cycles)} cycles/rooms")

            # Scale cycles if needed
            simple_cycles_scaled = []
            for polygon in simple_cycles:
                polygon_scaled = [(p[0] * aa_scale, p[1] * aa_scale) for p in polygon]
                simple_cycles_scaled.append(polygon_scaled)

            # Draw floor plan
            img = Image.new('RGB', (resolution * aa_scale, resolution * aa_scale), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Draw room polygons
            for polygon_i, polygon in enumerate(simple_cycles_scaled):
                room_color = colors.get(simple_cycles_semantics[polygon_i], (200, 200, 200))
                draw.polygon(polygon, fill=room_color, outline=None)

            # Draw walls and corners
            for polygon_i, polygon in enumerate(simple_cycles_scaled):
                for point_i, point in enumerate(polygon):
                    if point_i < len(polygon) - 1:
                        p1 = (point[0], point[1])
                        # Draw corner point
                        draw.rectangle((p1[0] - 3 * aa_scale, p1[1] - 3 * aa_scale,
                                      p1[0] + 3 * aa_scale, p1[1] + 3 * aa_scale),
                                     fill=(150, 150, 150), outline=None)
                        # Draw wall line
                        p2 = (polygon[point_i + 1][0], polygon[point_i + 1][1])
                        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(150, 150, 150), width=7 * aa_scale)

            # Save
            output_path = os.path.join(output_dir, f"test_gt_{test_count}.png")
            img.save(output_path)
            print(f"    ✓ Saved: {output_path}")

        except Exception as e:
            print(f"    ✗ Error generating PNG: {e}")

        test_count += 1

print(f"\n{'='*60}")
print(f"✓ Generated {test_count} ground truth PNGs in {output_dir}/")
print(f"{'='*60}")
print("\nCheck the PNGs to verify they are not empty!")
