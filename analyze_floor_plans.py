#!/usr/bin/env python3
"""
Floor Plan Corner Analysis Tool
================================

This script analyzes GSDiff floor plan data files (.npy format) to:
1. Count actual corners (excluding padding)
2. Display corner statistics across dataset
3. Visualize walls/edges for inspection

Usage:
    python analyze_floor_plans.py <data_path> [--num-samples N] [--visualize]

Example:
    python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train --num-samples 10 --visualize

Author: Claude
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


def load_floor_plan(file_path):
    """Load a floor plan .npy file and extract corner/edge information."""
    try:
        graph = np.load(file_path, allow_pickle=True).item()
        return graph
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def count_actual_corners(graph):
    """Count non-padded corners in a floor plan."""
    if 'padding_mask' in graph:
        # padding_mask: 1 = real corner, 0 = padding
        padding_mask = graph['padding_mask']
        return int(padding_mask.sum())
    elif 'corner_list_np_normalized_padding_withsemantics' in graph:
        # Fallback: count non-zero coordinate rows
        corners = graph['corner_list_np_normalized_padding_withsemantics']
        # Assume corners are (x, y, ...) and padding has x=y=0
        non_padding = (corners[:, 0] != 0) | (corners[:, 1] != 0)
        return int(non_padding.sum())
    else:
        return 0


def visualize_floor_plan(graph, title="Floor Plan"):
    """Visualize floor plan with corners and edges."""
    corners = graph.get('corner_list_np_normalized_padding_withsemantics', None)
    if corners is None:
        print("No corner data found for visualization")
        return

    padding_mask = graph.get('padding_mask', np.ones(len(corners)))
    edges = graph.get('edges', None)

    # Filter out padding corners
    valid_indices = np.where(padding_mask == 1)[0]
    valid_corners = corners[valid_indices, :2]  # Extract x, y coordinates

    # Convert from normalized [-1, 1] to pixel coordinates for visualization
    # Assuming original range was [0, 256]
    coords = ((valid_corners + 1) / 2) * 256

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match image coordinates (y increases downward)

    # Draw edges if available
    if edges is not None and len(edges) > 0:
        max_corners = len(corners)

        # Reshape edges to adjacency matrix
        if edges.shape[0] == max_corners * max_corners:
            edge_matrix = edges.reshape(max_corners, max_corners)
        else:
            # Fallback: try to interpret as edge list
            edge_matrix = None

        if edge_matrix is not None:
            # Extract valid edges
            for i in valid_indices:
                for j in valid_indices:
                    if i < j and edge_matrix[i, j] > 0.5:  # Edge exists
                        x1, y1 = coords[np.where(valid_indices == i)[0][0]]
                        x2, y2 = coords[np.where(valid_indices == j)[0][0]]
                        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6)

    # Draw corners
    for idx, (x, y) in enumerate(coords):
        # Different colors for different semantic types (if available)
        color = 'red'
        if corners.shape[1] > 2:
            # Check semantic flags (columns 2+)
            semantics = corners[valid_indices[idx], 2:]
            if np.any(semantics > 0):
                color = 'green'  # Has semantic information

        ax.add_patch(Circle((x, y), 3, color=color, alpha=0.8))
        ax.text(x + 5, y - 5, str(idx), fontsize=8, color='black')

    ax.set_title(f"{title}\nTotal Corners: {len(coords)}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Legend
    red_patch = mpatches.Patch(color='red', label='Corner (no semantics)')
    green_patch = mpatches.Patch(color='green', label='Corner (with semantics)')
    blue_line = mpatches.Patch(color='blue', label='Edge/Wall', alpha=0.6)
    ax.legend(handles=[red_patch, green_patch, blue_line], loc='upper right')

    plt.tight_layout()
    plt.show()


def analyze_dataset(data_path, num_samples=None, visualize=False):
    """Analyze floor plan dataset and report statistics."""
    if not os.path.exists(data_path):
        print(f"Error: Path '{data_path}' does not exist")
        return

    # Get all .npy files
    if os.path.isfile(data_path):
        files = [data_path]
        data_dir = os.path.dirname(data_path)
    else:
        files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
        data_dir = data_path

    if not files:
        print(f"No .npy files found in {data_path}")
        return

    # Sort files
    try:
        files = sorted(files, key=lambda x: int(os.path.basename(x)[:-4]))
    except ValueError:
        files = sorted(files)

    # Limit number of samples
    if num_samples is not None:
        files = files[:num_samples]

    print(f"\nAnalyzing {len(files)} floor plan files from: {data_path}")
    print("=" * 70)

    corner_counts = []
    file_details = []

    for i, file_path in enumerate(files):
        graph = load_floor_plan(file_path)
        if graph is None:
            continue

        num_corners = count_actual_corners(graph)
        corner_counts.append(num_corners)

        file_details.append({
            'file': os.path.basename(file_path),
            'corners': num_corners,
            'graph': graph
        })

        # Print progress every 1000 files
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} files...")

    # Statistics
    if corner_counts:
        counts_array = np.array(corner_counts)
        count_distribution = Counter(corner_counts)

        print(f"\nCorner Statistics:")
        print(f"  Total files analyzed: {len(corner_counts)}")
        print(f"  Min corners: {counts_array.min()}")
        print(f"  Max corners: {counts_array.max()}")
        print(f"  Mean corners: {counts_array.mean():.2f}")
        print(f"  Median corners: {np.median(counts_array):.2f}")
        print(f"  Std deviation: {counts_array.std():.2f}")

        print(f"\nCorner Count Distribution (top 10 most common):")
        for count, freq in count_distribution.most_common(10):
            percentage = (freq / len(corner_counts)) * 100
            print(f"  {count} corners: {freq} files ({percentage:.2f}%)")

        # Find examples with different corner counts
        print(f"\nExample files by corner count:")
        counts_to_show = sorted(set([counts_array.min(), counts_array.max(),
                                      int(np.median(counts_array))]))
        for target_count in counts_to_show[:5]:
            matching = [d for d in file_details if d['corners'] == target_count]
            if matching:
                example = matching[0]
                print(f"  {target_count} corners: {example['file']}")

        # Visualize samples if requested
        if visualize:
            print(f"\nVisualizing sample floor plans...")

            # Show 3-5 samples with different corner counts
            samples_to_visualize = []
            for target_count in counts_to_show[:3]:
                matching = [d for d in file_details if d['corners'] == target_count]
                if matching:
                    samples_to_visualize.append(matching[0])

            for sample in samples_to_visualize:
                visualize_floor_plan(
                    sample['graph'],
                    title=f"{sample['file']} - {sample['corners']} corners"
                )
    else:
        print("No valid floor plans found")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GSDiff floor plan corner counts and visualize walls',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all files in training set
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train

  # Analyze first 100 files
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train --num-samples 100

  # Analyze and visualize 10 samples
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/val --num-samples 10 --visualize

  # Analyze single file with visualization
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train/0.npy --visualize
        """
    )

    parser.add_argument(
        'data_path',
        help='Path to dataset directory or single .npy file'
    )

    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=None,
        help='Number of samples to analyze (default: all)'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Visualize sample floor plans with corners and edges'
    )

    args = parser.parse_args()

    analyze_dataset(args.data_path, args.num_samples, args.visualize)


if __name__ == '__main__':
    main()
