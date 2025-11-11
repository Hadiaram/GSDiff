#!/usr/bin/env python3
"""
gpickle_to_npy_converter.py

Converts NetworkX graphs (saved as .gpickle files) to GSDiff-compatible .npy format.

This script adapts the preprocessing pipeline from rplan-process4.py to work with
custom NetworkX graphs instead of RPLAN PNG images.

Author: Generated for GSDiff custom dataset support
Date: 2025-11-10

Usage:
    python gpickle_to_npy_converter.py \
        --input_dir path/to/gpickle/files \
        --output_dir path/to/output \
        --max_corners 53

Requirements:
    - NetworkX graphs must have node attributes: 'x', 'y' (coordinates)
    - Optional node attributes: 'semantic' (room type label 0-13)
    - Edges represent walls/connections between corners
"""

import os
import sys
import argparse
import copy
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import warnings

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MAX_CORNERS = 53
DEFAULT_SEMANTIC_DIM = 14

# Semantic label mappings (same as RPLAN)
SEMANTIC_LABELS = {
    0: 'Living room / Dining / Entrance',
    1: 'Master bedroom',
    2: 'Kitchen',
    3: 'Bathroom',
    4: 'Dining room',
    5: 'Child room',
    6: 'Study room',
    7: 'Second bedroom',
    8: 'Guest room',
    9: 'Balcony',
    10: 'Entrance',
    11: 'Storage',
    12: 'Walk-in closet',
    13: 'External area'
}


# ============================================================================
# Helper Functions (adapted from rplan-process4.py)
# ============================================================================

def get_label_from_semantics(semantic_values):
    """
    Determine the most common semantic label.

    Args:
        semantic_values: List of semantic labels

    Returns:
        Most common label (int)
    """
    if not semantic_values:
        return 0  # Default to living room if no semantics

    label_counts = Counter(semantic_values)
    max_label = max(label_counts, key=label_counts.get)
    return max_label


def extract_cycles_and_semantics(G, corners_dict, node_semantics):
    """
    Extract room polygons (cycles) from graph and assign semantic labels.

    Args:
        G: NetworkX graph
        corners_dict: Dictionary mapping node_id → (x, y) coordinates
        node_semantics: Dictionary mapping node_id → semantic label

    Returns:
        List of (polygon_vertices, semantic_label) tuples
    """
    polygon_semantics = []

    # Remove bridges (edges that disconnect the graph)
    bridges = list(nx.bridges(G))
    G_copy = G.copy()
    for bridge in bridges:
        if G_copy.has_edge(bridge[0], bridge[1]):
            G_copy.remove_edge(bridge[0], bridge[1])

    # Find connected components (after removing bridges, these are rooms)
    connected_components = list(nx.connected_components(G_copy))

    for component in connected_components:
        if len(component) <= 2:
            # Skip isolated nodes or simple edges
            continue

        # Get subgraph for this component
        subgraph = G_copy.subgraph(component)

        # For cyclic components, extract the cycle
        if len(component) >= 3:
            # Get nodes in cycle
            cycle_nodes = list(component)

            # Get semantic labels for nodes in this cycle
            semantics_in_cycle = [node_semantics.get(node, 0) for node in cycle_nodes]
            cycle_semantic = get_label_from_semantics(semantics_in_cycle)

            # Get coordinates
            polygon_coords = [corners_dict[node] for node in cycle_nodes]

            polygon_semantics.append((polygon_coords, cycle_semantic))

    return polygon_semantics


def create_semantic_vectors(corners_list, polygon_semantics, semantic_dim=14):
    """
    Create semantic vectors for each corner based on adjacent rooms.

    Args:
        corners_list: List of (x, y) corner coordinates
        polygon_semantics: List of (polygon, semantic_label) tuples
        semantic_dim: Number of semantic dimensions (default 14)

    Returns:
        Dictionary mapping corner coordinates → semantic vector
    """
    semantic_dict = {}

    # Initialize all corners with zero vectors
    for corner in corners_list:
        corner_tuple = tuple(corner)
        semantic_dict[corner_tuple] = [0] * semantic_dim

    # For each room polygon, mark its corners with that semantic type
    for polygon, semantic_label in polygon_semantics:
        for coord in polygon:
            corner_tuple = tuple(coord)
            if corner_tuple in semantic_dict:
                if semantic_label < semantic_dim:
                    semantic_dict[corner_tuple][semantic_label] += 1

    return semantic_dict


# ============================================================================
# Main Conversion Functions
# ============================================================================

def load_networkx_graph(gpickle_path):
    """
    Load NetworkX graph from gpickle file.

    Args:
        gpickle_path: Path to .gpickle file

    Returns:
        NetworkX graph

    Raises:
        ValueError: If graph format is invalid
    """
    try:
        G = nx.read_gpickle(gpickle_path)
    except Exception as e:
        raise ValueError(f"Failed to load gpickle file: {e}")

    # Validate graph has required attributes
    if len(G.nodes()) == 0:
        raise ValueError("Graph has no nodes")

    # Check if nodes have position attributes
    sample_node = list(G.nodes())[0]
    node_data = G.nodes[sample_node]

    if 'x' not in node_data or 'y' not in node_data:
        if 'pos' in node_data:
            # Try extracting from 'pos' attribute
            warnings.warn("Nodes have 'pos' attribute instead of 'x', 'y'. Extracting coordinates...")
        else:
            raise ValueError(
                "Nodes must have 'x' and 'y' attributes (or 'pos' tuple). "
                f"Found attributes: {list(node_data.keys())}"
            )

    return G


def extract_corners_and_edges(G):
    """
    Extract corner coordinates and adjacency information from NetworkX graph.

    Args:
        G: NetworkX graph with node attributes 'x', 'y' (or 'pos')

    Returns:
        Tuple of (corners_dict, corners_list, adjacency_list, adjacency_matrix, node_semantics)
    """
    # Extract corner coordinates
    corners_dict = {}  # node_id → (x, y)
    node_to_idx = {}   # node_id → index
    node_semantics = {}  # node_id → semantic label

    for idx, node in enumerate(G.nodes()):
        node_data = G.nodes[node]

        # Extract coordinates
        if 'x' in node_data and 'y' in node_data:
            x, y = node_data['x'], node_data['y']
        elif 'pos' in node_data:
            x, y = node_data['pos']
        else:
            raise ValueError(f"Node {node} has no position information")

        corners_dict[node] = (x, y)
        node_to_idx[node] = idx

        # Extract semantic label if available
        if 'semantic' in node_data:
            node_semantics[node] = node_data['semantic']
        else:
            node_semantics[node] = 0  # Default to living room

    # Create corners list (ordered by node index)
    corners_list = [corners_dict[node] for node in sorted(node_to_idx.keys(), key=lambda n: node_to_idx[n])]

    # Create adjacency list
    adjacency_list = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        adjacency_list[node] = neighbors

    # Create adjacency matrix
    n = len(G.nodes())
    adjacency_matrix = [[0] * n for _ in range(n)]

    for node in G.nodes():
        i = node_to_idx[node]
        for neighbor in G.neighbors(node):
            j = node_to_idx[neighbor]
            adjacency_matrix[i][j] = 1

    return corners_dict, corners_list, adjacency_list, adjacency_matrix, node_semantics, node_to_idx


def normalize_coordinates(corners_list, coord_range=(0, 256)):
    """
    Normalize coordinates to [-1, 1] range.

    Args:
        corners_list: List of (x, y) tuples
        coord_range: Original coordinate range (min, max)

    Returns:
        Normalized corners array (n, 2)
    """
    corners_np = np.array(corners_list, dtype=np.float64)

    # Determine normalization parameters
    if coord_range is not None:
        min_coord, max_coord = coord_range
        center = (min_coord + max_coord) / 2
        scale = (max_coord - min_coord) / 2
    else:
        # Auto-detect from data
        min_coord = corners_np.min()
        max_coord = corners_np.max()
        center = (min_coord + max_coord) / 2
        scale = (max_coord - min_coord) / 2

    # Normalize: (coord - center) / scale
    corner_list_np_normalized = (corners_np - center) / scale

    return corner_list_np_normalized


def create_padded_arrays(corners_list_normalized, adjacency_matrix, max_corners=53):
    """
    Create padded arrays with fixed size.

    Args:
        corners_list_normalized: Array of normalized coordinates (n, 2)
        adjacency_matrix: Adjacency matrix (n, n)
        max_corners: Maximum number of corners to pad to

    Returns:
        Dictionary with padded arrays
    """
    n = len(corners_list_normalized)

    if n > max_corners:
        warnings.warn(
            f"Graph has {n} corners, which exceeds max_corners={max_corners}. "
            f"Truncating to first {max_corners} corners."
        )
        n = max_corners
        corners_list_normalized = corners_list_normalized[:max_corners]
        adjacency_matrix = [row[:max_corners] for row in adjacency_matrix[:max_corners]]

    # Pad corner coordinates
    corner_list_np_normalized_padding = np.zeros((max_corners, 2), dtype=np.float64)
    corner_list_np_normalized_padding[:n, :] = corners_list_normalized[:n]

    # Padding mask (1 = real corner, 0 = padding)
    padding_mask = np.zeros((max_corners, 1), dtype=np.uint8)
    padding_mask[:n, :] = 1

    # Global attention matrix (all real corners attend to each other)
    global_matrix_np_padding = np.zeros((max_corners, max_corners), dtype=np.uint8)
    global_matrix_np_padding[:n, :n] = 1

    # Adjacency matrix padding
    adjacency_matrix_np = np.array(adjacency_matrix, dtype=np.uint8)
    adjacency_matrix_np_padding = np.zeros((max_corners, max_corners), dtype=np.uint8)
    adjacency_matrix_np_padding[:n, :n] = adjacency_matrix_np[:n, :n]

    # Edge coordinates: all pairwise combinations
    edge_coord1 = np.repeat(corner_list_np_normalized_padding[:, None, :], max_corners, axis=1)
    edge_coord2 = np.repeat(corner_list_np_normalized_padding[None, :, :], max_corners, axis=0)
    edge_coords = np.concatenate((edge_coord1, edge_coord2), axis=2).reshape(-1, 4)

    # Edges: flattened adjacency matrix
    edges = adjacency_matrix_np_padding.reshape(-1, 1)

    return {
        'corner_list_np_normalized_padding': corner_list_np_normalized_padding,
        'padding_mask': padding_mask,
        'global_matrix_np_padding': global_matrix_np_padding,
        'adjacency_matrix_np_padding': adjacency_matrix_np_padding,
        'edge_coords': edge_coords,
        'edges': edges,
        'n_real_corners': n
    }


def create_corners_with_semantics(corner_list_normalized_padding, semantic_dict,
                                    corners_list, max_corners=53, semantic_dim=14):
    """
    Combine corner coordinates with semantic labels.

    Args:
        corner_list_normalized_padding: Padded corner coordinates (max_corners, 2)
        semantic_dict: Dictionary mapping coords → semantic vector
        corners_list: Original list of corner coordinates
        max_corners: Maximum number of corners
        semantic_dim: Number of semantic dimensions

    Returns:
        Array of shape (max_corners, 2 + semantic_dim)
    """
    result = np.zeros((max_corners, 2 + semantic_dim), dtype=np.float64)

    for idx in range(len(corners_list)):
        if idx >= max_corners:
            break

        # Get coordinates
        coord = corner_list_normalized_padding[idx]
        coord_tuple = tuple(coord)

        # Get semantic vector
        if coord_tuple in semantic_dict:
            semantic_vector = semantic_dict[coord_tuple]
        else:
            # Try finding by original coordinates
            original_coord = tuple(corners_list[idx])
            if original_coord in semantic_dict:
                semantic_vector = semantic_dict[original_coord]
            else:
                semantic_vector = [0] * semantic_dim

        # Combine
        result[idx] = np.concatenate((coord, semantic_vector))

    return result


def convert_gpickle_to_npy(gpickle_path, output_path, max_corners=53,
                           semantic_dim=14, coord_range=None, file_id=None):
    """
    Convert a single gpickle file to GSDiff .npy format.

    Args:
        gpickle_path: Path to input .gpickle file
        output_path: Path to output .npy file
        max_corners: Maximum number of corners (default 53)
        semantic_dim: Number of semantic dimensions (default 14)
        coord_range: Tuple (min, max) for coordinate normalization
        file_id: Optional file ID (default: extract from filename)

    Returns:
        Dictionary that was saved to .npy file

    Raises:
        ValueError: If graph format is invalid
    """
    # Load NetworkX graph
    G = load_networkx_graph(gpickle_path)

    # Extract file ID
    if file_id is None:
        file_id = int(Path(gpickle_path).stem)

    # Extract corners and edges
    corners_dict, corners_list, adjacency_list, adjacency_matrix, node_semantics, node_to_idx = \
        extract_corners_and_edges(G)

    # Normalize coordinates
    corners_np = np.array(corners_list, dtype=np.float64)
    corner_list_np_normalized = normalize_coordinates(corners_list, coord_range)

    # Create semantic vectors
    try:
        polygon_semantics = extract_cycles_and_semantics(G, corners_dict, node_semantics)
        semantic_dict = create_semantic_vectors(corners_list, polygon_semantics, semantic_dim)
    except Exception as e:
        warnings.warn(f"Failed to extract room semantics: {e}. Using default semantics.")
        # Create default semantic vectors (all zeros)
        semantic_dict = {tuple(corner): [0] * semantic_dim for corner in corners_list}

    # Create normalized semantic dict (with normalized coordinates as keys)
    normalized_semantic_dict = {}
    for idx, corner_normalized in enumerate(corner_list_np_normalized):
        coord_tuple = tuple(corner_normalized)
        original_corner = tuple(corners_list[idx])
        if original_corner in semantic_dict:
            normalized_semantic_dict[coord_tuple] = semantic_dict[original_corner]
        else:
            normalized_semantic_dict[coord_tuple] = [0] * semantic_dim

    # Create padded arrays
    padded_data = create_padded_arrays(corner_list_np_normalized, adjacency_matrix, max_corners)

    # Create corners with semantics
    corners_with_semantics = create_corners_with_semantics(
        padded_data['corner_list_np_normalized_padding'],
        normalized_semantic_dict,
        corners_list,
        max_corners,
        semantic_dim
    )

    # Build final dictionary (same structure as rplan-process4.py output)
    graph_dict = {
        # Identification
        'file_id': file_id,

        # Original data
        'corners': corners_list,
        'corners_np': corners_np,
        'adjacency_matrix': adjacency_matrix,
        'adjacency_list': adjacency_list,
        'adjacency_list_np': np.array(list(adjacency_list.values()), dtype=object),

        # Normalized
        'corner_list_np_normalized': corner_list_np_normalized,

        # Padded
        'corner_list_np_normalized_padding': padded_data['corner_list_np_normalized_padding'],
        'padding_mask': padded_data['padding_mask'],
        'global_matrix_np_padding': padded_data['global_matrix_np_padding'],
        'adjacency_matrix_np_padding': padded_data['adjacency_matrix_np_padding'],

        # Edges
        'edge_coords': padded_data['edge_coords'],
        'edges': padded_data['edges'],

        # Semantics
        'semantics': normalized_semantic_dict,
        'corner_list_np_normalized_padding_withsemantics': corners_with_semantics,
    }

    # Save to .npy file
    np.save(output_path, graph_dict)

    return graph_dict


# ============================================================================
# Batch Conversion
# ============================================================================

def convert_directory(input_dir, output_dir, max_corners=53, semantic_dim=14,
                      coord_range=None, train_val_test_split=None):
    """
    Convert all .gpickle files in a directory to .npy format.

    Args:
        input_dir: Directory containing .gpickle files
        output_dir: Directory to save .npy files
        max_corners: Maximum number of corners
        semantic_dim: Number of semantic dimensions
        coord_range: Tuple (min, max) for coordinate normalization
        train_val_test_split: Optional dict with 'train', 'val', 'test' lists of file IDs

    Returns:
        Dictionary with conversion statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    if train_val_test_split:
        (output_dir / 'train').mkdir(parents=True, exist_ok=True)
        (output_dir / 'val').mkdir(parents=True, exist_ok=True)
        (output_dir / 'test').mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .gpickle files
    gpickle_files = sorted(input_dir.glob('*.gpickle'))

    if not gpickle_files:
        print(f"No .gpickle files found in {input_dir}")
        return {'converted': 0, 'failed': 0}

    print(f"Found {len(gpickle_files)} .gpickle files")

    # Convert each file
    stats = {'converted': 0, 'failed': 0, 'errors': []}

    for gpickle_path in tqdm(gpickle_files, desc="Converting files"):
        try:
            # Determine output path
            file_id = int(gpickle_path.stem)
            output_name = f"{file_id}.npy"

            if train_val_test_split:
                if file_id in train_val_test_split.get('train', []):
                    output_path = output_dir / 'train' / output_name
                elif file_id in train_val_test_split.get('val', []):
                    output_path = output_dir / 'val' / output_name
                elif file_id in train_val_test_split.get('test', []):
                    output_path = output_dir / 'test' / output_name
                else:
                    output_path = output_dir / output_name
            else:
                output_path = output_dir / output_name

            # Convert
            convert_gpickle_to_npy(
                gpickle_path,
                output_path,
                max_corners=max_corners,
                semantic_dim=semantic_dim,
                coord_range=coord_range,
                file_id=file_id
            )

            stats['converted'] += 1

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append((gpickle_path.name, str(e)))
            print(f"\nError converting {gpickle_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {stats['converted']}")
    print(f"Failed: {stats['failed']}")

    if stats['errors']:
        print(f"\nErrors:")
        for filename, error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {filename}: {error}")

    return stats


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert NetworkX graphs (.gpickle) to GSDiff .npy format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all files in a directory
    python gpickle_to_npy_converter.py --input_dir data/graphs --output_dir data/npy

    # Specify coordinate range for normalization
    python gpickle_to_npy_converter.py --input_dir data/graphs --output_dir data/npy --coord_range 0 1000

    # Convert single file
    python gpickle_to_npy_converter.py --input data/graphs/0.gpickle --output data/npy/0.npy

    # Change maximum corners
    python gpickle_to_npy_converter.py --input_dir data/graphs --output_dir data/npy --max_corners 100

NetworkX Graph Requirements:
    - Nodes must have 'x' and 'y' attributes (or 'pos' tuple)
    - Optional: 'semantic' attribute (0-13) for room type
    - Edges represent connections/walls between corners
        """
    )

    parser.add_argument('--input', type=str, help='Input .gpickle file')
    parser.add_argument('--input_dir', type=str, help='Input directory containing .gpickle files')
    parser.add_argument('--output', type=str, help='Output .npy file (for single file conversion)')
    parser.add_argument('--output_dir', type=str, help='Output directory for .npy files')
    parser.add_argument('--max_corners', type=int, default=53,
                       help='Maximum number of corners (default: 53)')
    parser.add_argument('--semantic_dim', type=int, default=14,
                       help='Number of semantic dimensions (default: 14)')
    parser.add_argument('--coord_range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='Original coordinate range for normalization (e.g., 0 256)')
    parser.add_argument('--split_file', type=str,
                       help='JSON file with train/val/test split (keys: train, val, test with file ID lists)')

    args = parser.parse_args()

    # Validate arguments
    if args.input and args.input_dir:
        parser.error("Specify either --input or --input_dir, not both")

    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input_dir")

    if args.input and not args.output:
        parser.error("Must specify --output when using --input")

    if args.input_dir and not args.output_dir:
        parser.error("Must specify --output_dir when using --input_dir")

    # Load split file if provided
    split = None
    if args.split_file:
        import json
        with open(args.split_file, 'r') as f:
            split = json.load(f)

    # Convert
    coord_range = tuple(args.coord_range) if args.coord_range else None

    if args.input:
        # Single file conversion
        print(f"Converting {args.input} → {args.output}")
        result = convert_gpickle_to_npy(
            args.input,
            args.output,
            max_corners=args.max_corners,
            semantic_dim=args.semantic_dim,
            coord_range=coord_range
        )
        print(f"Conversion successful!")
        print(f"  - Corners: {result['padding_mask'].sum()}")
        print(f"  - Edges: {result['adjacency_matrix_np_padding'][:int(result['padding_mask'].sum()), :int(result['padding_mask'].sum())].sum() // 2}")
    else:
        # Directory conversion
        stats = convert_directory(
            args.input_dir,
            args.output_dir,
            max_corners=args.max_corners,
            semantic_dim=args.semantic_dim,
            coord_range=coord_range,
            train_val_test_split=split
        )


if __name__ == '__main__':
    main()
