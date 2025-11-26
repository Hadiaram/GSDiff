#!/usr/bin/env python3
"""
Floor Plan Corner Analysis Tool
================================

This script analyzes GSDiff floor plan data files (.npy or .json format) to:
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
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


def load_json_floor_plan(file_path):
    """Load a floor plan from JSON file and convert to graph format."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Convert JSON to graph format
        graph = {}

        # Handle different JSON formats
        if isinstance(data, dict):
            # Format 1: Direct graph structure (already has corners, edges, etc.)
            if 'corners' in data or 'vertices' in data or 'junctions' in data:
                corners_key = 'corners' if 'corners' in data else ('vertices' if 'vertices' in data else 'junctions')
                corners = data[corners_key]

                # Convert corners to numpy array
                if isinstance(corners, list):
                    corners_array = np.array(corners)

                    # Ensure shape is (N, 2) at minimum
                    if corners_array.ndim == 1:
                        corners_array = corners_array.reshape(-1, 2)

                    # Add dummy semantics if not present (9 dimensions total: x, y, + 7 semantics)
                    if corners_array.shape[1] < 9:
                        num_corners = corners_array.shape[0]
                        full_array = np.zeros((num_corners, 9))
                        full_array[:, :corners_array.shape[1]] = corners_array
                        corners_array = full_array

                    graph['corner_list_np_normalized_padding_withsemantics'] = corners_array
                    graph['padding_mask'] = np.ones(len(corners_array))  # All real corners

                # Handle edges/walls
                if 'edges' in data:
                    edges = data['edges']
                    if isinstance(edges, list):
                        # Convert edge list to adjacency matrix
                        num_corners = len(corners)
                        edge_matrix = np.zeros((num_corners * num_corners, 1))
                        for edge in edges:
                            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                                i, j = edge[0], edge[1]
                                edge_matrix[i * num_corners + j] = 1
                                edge_matrix[j * num_corners + i] = 1  # Symmetric
                        graph['edges'] = edge_matrix

            # Format 2: RPLAN-style annotation
            elif 'boxes' in data or 'lines' in data:
                # Extract corners from boxes or lines
                corners_set = set()

                if 'boxes' in data:
                    for box in data['boxes']:
                        if isinstance(box, dict):
                            # Extract corners from bounding box
                            x, y, w, h = box.get('x', 0), box.get('y', 0), box.get('width', 0), box.get('height', 0)
                            corners_set.add((x, y))
                            corners_set.add((x + w, y))
                            corners_set.add((x, y + h))
                            corners_set.add((x + w, y + h))

                if 'lines' in data or 'walls' in data:
                    lines_key = 'lines' if 'lines' in data else 'walls'
                    for line in data[lines_key]:
                        if isinstance(line, (list, tuple)) and len(line) >= 4:
                            corners_set.add((line[0], line[1]))
                            corners_set.add((line[2], line[3]))
                        elif isinstance(line, dict):
                            x1, y1 = line.get('x1', 0), line.get('y1', 0)
                            x2, y2 = line.get('x2', 0), line.get('y2', 0)
                            corners_set.add((x1, y1))
                            corners_set.add((x2, y2))

                if corners_set:
                    corners_list = sorted(list(corners_set))
                    corners_array = np.zeros((len(corners_list), 9))
                    corners_array[:, :2] = np.array(corners_list)
                    graph['corner_list_np_normalized_padding_withsemantics'] = corners_array
                    graph['padding_mask'] = np.ones(len(corners_array))

        elif isinstance(data, list):
            # Format 3: Direct list of corners
            corners_array = np.array(data)
            if corners_array.ndim == 1:
                corners_array = corners_array.reshape(-1, 2)

            if corners_array.shape[1] < 9:
                num_corners = corners_array.shape[0]
                full_array = np.zeros((num_corners, 9))
                full_array[:, :corners_array.shape[1]] = corners_array
                corners_array = full_array

            graph['corner_list_np_normalized_padding_withsemantics'] = corners_array
            graph['padding_mask'] = np.ones(len(corners_array))

        return graph if graph else None

    except Exception as e:
        print(f"Error loading JSON {file_path}: {e}")
        return None


def load_floor_plan(file_path):
    """Load a floor plan from .npy or .json file and extract corner/edge information."""
    try:
        # Detect file type
        if file_path.endswith('.json'):
            return load_json_floor_plan(file_path)
        elif file_path.endswith('.npy'):
            graph = np.load(file_path, allow_pickle=True).item()
            return graph
        else:
            print(f"Unsupported file format: {file_path}")
            return None
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


def visualize_floor_plan(graph, title="Floor Plan", return_count=True):
    """Visualize floor plan with corners and edges.

    Returns:
        int: Number of corners actually displayed (if return_count=True)
        None: If return_count=False
    """
    corners = graph.get('corner_list_np_normalized_padding_withsemantics', None)
    if corners is None:
        print("No corner data found for visualization")
        return 0 if return_count else None

    padding_mask = graph.get('padding_mask', np.ones(len(corners)))
    edges = graph.get('edges', None)

    # Filter out padding corners
    valid_indices = np.where(padding_mask == 1)[0]
    valid_corners = corners[valid_indices, :2]  # Extract x, y coordinates

    # THIS IS THE ACTUAL COUNT - based on what will be displayed
    displayed_corner_count = len(valid_corners)

    # Convert from normalized [-1, 1] to pixel coordinates for visualization
    # Assuming original range was [0, 256]
    coords = ((valid_corners + 1) / 2) * 256

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match image coordinates (y increases downward)

    # Count edges for display
    edge_count = 0

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
                        edge_count += 1

    # Draw corners
    corners_with_semantics = 0
    for idx, (x, y) in enumerate(coords):
        # Different colors for different semantic types (if available)
        color = 'red'
        if corners.shape[1] > 2:
            # Check semantic flags (columns 2+)
            semantics = corners[valid_indices[idx], 2:]
            if np.any(semantics > 0):
                color = 'green'  # Has semantic information
                corners_with_semantics += 1

        ax.add_patch(Circle((x, y), 3, color=color, alpha=0.8))
        ax.text(x + 5, y - 5, str(idx), fontsize=8, color='black')

    # Enhanced title with detailed counts
    title_text = f"{title}\n"
    title_text += f"Displayed Corners: {displayed_corner_count}"
    if corners_with_semantics > 0:
        title_text += f" ({corners_with_semantics} with semantics)"
    if edge_count > 0:
        title_text += f"\nEdges/Walls: {edge_count}"

    ax.set_title(title_text, fontsize=12, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Legend
    red_patch = mpatches.Patch(color='red', label='Corner (no semantics)')
    green_patch = mpatches.Patch(color='green', label='Corner (with semantics)')
    blue_line = mpatches.Patch(color='blue', label='Edge/Wall', alpha=0.6)
    ax.legend(handles=[red_patch, green_patch, blue_line], loc='upper right')

    plt.tight_layout()
    plt.show()

    # Return the count of corners that were actually displayed
    return displayed_corner_count if return_count else None


def analyze_dataset(data_path, num_samples=None, visualize=False):
    """Analyze floor plan dataset and report statistics."""
    if not os.path.exists(data_path):
        print(f"Error: Path '{data_path}' does not exist")
        return

    # Get all .npy and .json files
    if os.path.isfile(data_path):
        files = [data_path]
        data_dir = os.path.dirname(data_path)
    else:
        files = [os.path.join(data_path, f) for f in os.listdir(data_path)
                 if f.endswith('.npy') or f.endswith('.json')]
        data_dir = data_path

    if not files:
        print(f"No .npy or .json files found in {data_path}")
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

        # When visualizing, get corner count from visualization
        # Otherwise use standard counting
        if visualize and (num_samples is None or i < num_samples):
            # Visualize first, get the displayed count
            print(f"\n{'='*70}")
            print(f"Visualizing: {os.path.basename(file_path)}")
            print(f"{'='*70}")
            num_corners = visualize_floor_plan(
                graph,
                title=f"{os.path.basename(file_path)}",
                return_count=True
            )
            # Also get raw count for comparison
            raw_count = count_actual_corners(graph)
            if raw_count != num_corners:
                print(f"\nNote: Raw count ({raw_count}) differs from displayed count ({num_corners})")
        else:
            # Just count without visualizing
            num_corners = count_actual_corners(graph)

        corner_counts.append(num_corners)

        file_details.append({
            'file': os.path.basename(file_path),
            'corners': num_corners,
            'graph': graph
        })

        # Print progress every 1000 files (only when not visualizing each one)
        if not visualize and (i + 1) % 1000 == 0:
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

        # Note about visualization
        if visualize:
            print(f"\n✓ All files were visualized during analysis")
            print(f"  Corner counts shown above are based on displayed corners")
        else:
            print(f"\nTip: Use --visualize to see floor plans and verify corner counts")
    else:
        print("No valid floor plans found")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze floor plan corner counts and visualize walls (supports .npy and .json)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all .npy files in training set
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train

  # Analyze first 100 files (both .npy and .json)
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train --num-samples 100

  # Analyze and visualize 10 samples
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/val --num-samples 10 --visualize

  # Analyze single .npy file with visualization
  python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train/0.npy --visualize

  # Analyze JSON files
  python analyze_floor_plans.py ./my_floor_plans/ --visualize

  # Analyze single JSON file
  python analyze_floor_plans.py ./my_floor_plan.json --visualize

Supported JSON formats:
  1. {"corners": [[x1,y1], [x2,y2], ...], "edges": [[i,j], ...]}
  2. {"vertices": [[x1,y1], ...], "edges": [...]}
  3. {"boxes": [...], "lines": [...]} (RPLAN-style)
  4. [[x1,y1], [x2,y2], ...] (direct corner list)
        """
    )

    parser.add_argument(
        'data_path',
        help='Path to dataset directory or single .npy/.json file'
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
        help='Visualize floor plans FIRST, then count displayed corners (visualization-driven counting)'
    )

    args = parser.parse_args()

    analyze_dataset(args.data_path, args.num_samples, args.visualize)


if __name__ == '__main__':
    main()
