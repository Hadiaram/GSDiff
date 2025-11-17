"""
Raster-to-Graph Converter for GSDiff

Converts flood-filled raster NPY files (2D arrays with room IDs) to GSDiff's
graph format (dictionaries with corners, edges, and semantics).

Usage:
    python raster_to_graph_converter.py --input_dir path/to/raster_npy_files --output_dir path/to/output

Input format:  NPY files containing 2D arrays where:
    - 0 = walls/boundaries
    - 1, 2, 3, ... = room IDs

Output format: NPY files containing dictionaries with all GSDiff-required fields
"""

import numpy as np
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy import ndimage
from skimage import measure
from collections import defaultdict
import copy


# Room type mapping (adjust based on your data)
ROOM_TYPE_MAPPING = {
    1: 'LivingRoom',      # Living room
    2: 'MasterRoom',      # Master bedroom
    3: 'Kitchen',         # Kitchen
    4: 'Bathroom',        # Bathroom
    5: 'DiningRoom',      # Dining room
    6: 'ChildRoom',       # Child room
    7: 'StudyRoom',       # Study room
    8: 'SecondRoom',      # Second bedroom
    9: 'GuestRoom',       # Guest room
    10: 'Balcony',        # Balcony
    11: 'Entrance',       # Entrance
    12: 'Storage',        # Storage
    13: 'WallIn',         # Walk-in closet
    14: 'External',       # External area
    15: 'ExteriorWall',   # Exterior wall
    16: 'FrontDoor',      # Front door
    17: 'InteriorWall',   # Interior wall
    18: 'InteriorDoor',   # Interior door
}

# Semantic vector indices (14 dimensions as per GSDiff)
# These are based on the RPLAN dataset structure
SEMANTIC_INDICES = {
    'LivingRoom': 0,
    'MasterRoom': 1,
    'Kitchen': 2,
    'Bathroom': 3,
    'DiningRoom': 4,
    'ChildRoom': 5,
    'StudyRoom': 6,
    'SecondRoom': 7,
    'GuestRoom': 8,
    'Balcony': 9,
    'Entrance': 10,
    'Storage': 11,
    'WallIn': 12,
    'External': 13,
}


def extract_room_boundaries(raster_array, room_id, simplify_tolerance=2.0):
    """
    Extract boundary polygon for a specific room from raster array.

    Args:
        raster_array: 2D numpy array with room IDs
        room_id: ID of the room to extract
        simplify_tolerance: Douglas-Peucker simplification tolerance

    Returns:
        corners: List of (x, y) tuples representing polygon corners
    """
    # Create binary mask for this room
    room_mask = (raster_array == room_id).astype(np.uint8)

    if room_mask.sum() == 0:
        return []

    # Find contours using OpenCV (more robust than skimage for this use case)
    contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return []

    # Take the largest contour (in case there are multiple disconnected regions)
    contour = max(contours, key=cv2.contourArea)

    # Simplify polygon using Douglas-Peucker algorithm
    epsilon = simplify_tolerance
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Convert to list of tuples
    corners = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]

    return corners


def extract_all_rooms(raster_array, max_corners_per_room=20):
    """
    Extract all rooms from raster array.

    Returns:
        room_polygons: Dict mapping room_id -> list of corner coordinates
    """
    # Get unique room IDs (excluding 0 which is walls)
    room_ids = np.unique(raster_array)
    room_ids = room_ids[room_ids > 0]  # Remove walls (0)

    room_polygons = {}
    for room_id in room_ids:
        corners = extract_room_boundaries(raster_array, room_id, simplify_tolerance=2.0)

        # If too many corners, increase simplification
        if len(corners) > max_corners_per_room:
            corners = extract_room_boundaries(raster_array, room_id, simplify_tolerance=4.0)

        if len(corners) > 0:
            room_polygons[int(room_id)] = corners

    return room_polygons


def build_graph_from_rooms(room_polygons, image_size=256):
    """
    Build graph structure from room polygons.

    Returns:
        corners: List of unique corner coordinates
        adjacency_matrix: 2D adjacency matrix
        adjacency_list: Adjacency list representation
        corner_to_rooms: Mapping of corner index to room IDs that share it
    """
    # Collect all unique corners with room information
    corner_to_rooms = defaultdict(set)
    all_corners_with_rooms = []

    for room_id, corners_list in room_polygons.items():
        for i, corner in enumerate(corners_list):
            # Round to avoid floating point issues
            rounded = (round(corner[0], 2), round(corner[1], 2))
            corner_to_rooms[rounded].add(room_id)

    # Create unique corner list
    unique_corners = list(corner_to_rooms.keys())
    corner_index = {c: i for i, c in enumerate(unique_corners)}

    # Build adjacency from rooms
    n = len(unique_corners)
    adjacency_matrix = [[0] * n for _ in range(n)]
    adjacency_list = [[] for _ in range(n)]

    # For each room, connect consecutive corners
    for room_id, corners_list in room_polygons.items():
        # Round corners
        rounded_corners = [(round(c[0], 2), round(c[1], 2)) for c in corners_list]

        # Connect consecutive corners (including closing edge)
        for i in range(len(rounded_corners)):
            c1 = rounded_corners[i]
            c2 = rounded_corners[(i + 1) % len(rounded_corners)]

            if c1 in corner_index and c2 in corner_index:
                idx1 = corner_index[c1]
                idx2 = corner_index[c2]

                # Undirected edge
                adjacency_matrix[idx1][idx2] = 1
                adjacency_matrix[idx2][idx1] = 1

                if idx2 not in adjacency_list[idx1]:
                    adjacency_list[idx1].append(idx2)
                if idx1 not in adjacency_list[idx2]:
                    adjacency_list[idx2].append(idx1)

    return unique_corners, adjacency_matrix, adjacency_list, dict(corner_to_rooms)


def create_semantic_vector(corner, corner_to_rooms, room_polygons):
    """
    Create 14-dimensional semantic vector for a corner.

    Based on which room types this corner belongs to.
    """
    vector = [0] * 14

    # Get rooms that share this corner
    rooms = corner_to_rooms.get(corner, set())

    for room_id in rooms:
        # Map room_id to room type
        room_type = ROOM_TYPE_MAPPING.get(room_id, 'LivingRoom')  # Default to LivingRoom

        # Get semantic index
        if room_type in SEMANTIC_INDICES:
            idx = SEMANTIC_INDICES[room_type]
            if idx < 14:  # Safety check
                vector[idx] = 1

    return vector


def normalize_coordinates(corners, image_size=256):
    """
    Normalize corner coordinates from [0, image_size] to [-1, 1].
    """
    corners_array = np.array(corners, dtype=np.float64)
    normalized = (corners_array - image_size / 2) / (image_size / 2)
    return normalized


def convert_raster_to_graph(raster_file, output_file, image_size=256):
    """
    Convert a single raster NPY file to GSDiff graph format.

    Args:
        raster_file: Path to input raster NPY file
        output_file: Path to output graph NPY file
        image_size: Size of the raster image (default: 256)
    """
    # Load raster data
    raster_data = np.load(raster_file, allow_pickle=True)

    # Handle different input formats
    if isinstance(raster_data, np.ndarray):
        if raster_data.ndim == 2:
            # Direct 2D array - this is what we expect
            raster_array = raster_data
        elif raster_data.ndim == 0:
            # 0-d array containing the actual data
            raster_array = raster_data.item()
        else:
            raise ValueError(f"Unexpected raster format: shape={raster_data.shape}")
    else:
        raise ValueError(f"Unexpected data type: {type(raster_data)}")

    # Extract file ID from filename (keep as string to support descriptive names)
    file_id = Path(raster_file).stem

    # Extract room polygons
    room_polygons = extract_all_rooms(raster_array, max_corners_per_room=15)

    if len(room_polygons) == 0:
        print(f"Warning: No rooms found in {raster_file}")
        return False

    # Build graph structure
    corners, adjacency_matrix, adjacency_list, corner_to_rooms = build_graph_from_rooms(
        room_polygons, image_size
    )

    if len(corners) == 0:
        print(f"Warning: No corners found in {raster_file}")
        return False

    if len(corners) > 53:
        print(f"Warning: Too many corners ({len(corners)}) in {raster_file}, truncating to 53")
        # Simplify more aggressively
        room_polygons = extract_all_rooms(raster_array, max_corners_per_room=8)
        corners, adjacency_matrix, adjacency_list, corner_to_rooms = build_graph_from_rooms(
            room_polygons, image_size
        )

        if len(corners) > 53:
            # Still too many, truncate
            corners = corners[:53]
            adjacency_matrix = [row[:53] for row in adjacency_matrix[:53]]
            adjacency_list = adjacency_list[:53]

    # Create graph dictionary
    g = {}
    g['file_id'] = file_id

    # Original data
    g['corners'] = corners
    g['adjacency_matrix'] = adjacency_matrix
    g['adjacency_list'] = adjacency_list

    # Convert to numpy arrays
    corners_np = np.array([list(c) for c in corners], dtype=np.float64)
    g['corners_np'] = corners_np

    adjacency_matrix_np = np.array(adjacency_matrix, dtype=np.uint8)
    g['adjacency_matrix_np'] = adjacency_matrix_np

    adjacency_list_np = np.array([len(adj) for adj in adjacency_list], dtype=np.uint8)
    g['adjacency_list_np'] = adjacency_list_np

    # Normalized coordinates [-1, 1]
    corner_list_np_normalized = normalize_coordinates(corners, image_size)
    g['corner_list_np_normalized'] = corner_list_np_normalized

    # Padding to 53 corners
    padding_to_number = 53

    # Padded corner list
    corner_list_np_normalized_padding = np.zeros((padding_to_number, 2), dtype=np.float64)
    corner_list_np_normalized_padding[:len(corner_list_np_normalized), :] = corner_list_np_normalized
    g['corner_list_np_normalized_padding'] = corner_list_np_normalized_padding

    # Padding mask (1 = real corner, 0 = padding)
    padding_mask = np.zeros((padding_to_number, 1), dtype=np.uint8)
    padding_mask[:len(corner_list_np_normalized), :] = 1
    g['padding_mask'] = padding_mask

    # Global attention matrix (all real corners attend to each other)
    global_matrix_np_padding = np.zeros((padding_to_number, padding_to_number), dtype=np.uint8)
    global_matrix_np_padding[:len(corner_list_np_normalized), :len(corner_list_np_normalized)] = 1
    g['global_matrix_np_padding'] = global_matrix_np_padding

    # Adjacency matrix (padded)
    adjacency_matrix_np_padding = np.zeros((padding_to_number, padding_to_number), dtype=np.uint8)
    adjacency_matrix_np_padding[:len(adjacency_matrix_np), :len(adjacency_matrix_np)] = adjacency_matrix_np
    g['adjacency_matrix_np_padding'] = adjacency_matrix_np_padding

    # Edge coordinates and labels
    edge_coord1 = np.repeat(corner_list_np_normalized_padding[:, None, :], padding_to_number, axis=1)
    edge_coord2 = np.repeat(corner_list_np_normalized_padding[None, :, :], padding_to_number, axis=0)
    edge_coords = np.concatenate((edge_coord1, edge_coord2), axis=2).reshape(-1, 4)
    g['edge_coords'] = edge_coords

    edges = adjacency_matrix_np_padding[:, :, None].reshape(-1, 1)
    g['edges'] = edges

    # Create semantics dictionary and corner_list_with_semantics
    semantics = {}
    rounded_corners = [(round(c[0], 2), round(c[1], 2)) for c in corners]

    for corner, rounded in zip(corners, rounded_corners):
        semantic_vector = create_semantic_vector(rounded, corner_to_rooms, room_polygons)
        semantics[corner] = semantic_vector

    g['semantics'] = semantics

    # Create corner_list_np_normalized_padding_withsemantics (53, 16)
    # First 2 columns: normalized coordinates
    # Next 14 columns: semantic vectors
    result = np.zeros((53, 16), dtype=np.float64)

    for idx, (coord, rounded) in enumerate(zip(corner_list_np_normalized, rounded_corners)):
        if rounded in corner_to_rooms:
            vector = create_semantic_vector(rounded, corner_to_rooms, room_polygons)
        else:
            vector = [0] * 14

        result[idx] = np.concatenate((coord, vector))

    g['corner_list_np_normalized_padding_withsemantics'] = result

    # Save as numpy file
    np.save(output_file, g)

    return True


def batch_convert(input_dir, output_dir, image_size=256):
    """
    Convert all raster NPY files in input_dir to graph format in output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all NPY files
    npy_files = list(input_path.glob('*.npy'))

    if len(npy_files) == 0:
        print(f"No NPY files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} NPY files to convert")

    # Convert each file
    success_count = 0
    for npy_file in tqdm(npy_files, desc="Converting"):
        output_file = output_path / npy_file.name

        try:
            if convert_raster_to_graph(npy_file, output_file, image_size):
                success_count += 1
        except Exception as e:
            print(f"\nError processing {npy_file}: {e}")

    print(f"\nSuccessfully converted {success_count}/{len(npy_files)} files")


def validate_converted_file(filepath):
    """Validate that a converted file has the correct GSDiff format."""
    try:
        data = np.load(filepath, allow_pickle=True)

        if data.ndim == 0:
            graph = data.item()
        elif data.size == 1:
            graph = data.flatten()[0]
        else:
            print(f"❌ FAIL: {filepath} - Wrong array format")
            return False

        if not isinstance(graph, dict):
            print(f"❌ FAIL: {filepath} - Not a dictionary")
            return False

        # Check required keys
        required_keys = [
            'corner_list_np_normalized_padding_withsemantics',
            'padding_mask',
            'global_matrix_np_padding',
            'edges'
        ]

        for key in required_keys:
            if key not in graph:
                print(f"❌ FAIL: {filepath} - Missing key: {key}")
                return False

        # Check shapes
        corners = graph['corner_list_np_normalized_padding_withsemantics']
        if corners.shape != (53, 16):
            print(f"❌ FAIL: {filepath} - Wrong corners shape: {corners.shape}")
            return False

        padding_mask = graph['padding_mask']
        if padding_mask.shape != (53, 1):
            print(f"❌ FAIL: {filepath} - Wrong padding_mask shape: {padding_mask.shape}")
            return False

        print(f"✅ PASS: {filepath}")
        return True

    except Exception as e:
        print(f"❌ ERROR: {filepath} - {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert raster NPY files to GSDiff graph format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raster NPY files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save converted graph NPY files')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of the raster images (default: 256)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate converted files after conversion')

    args = parser.parse_args()

    # Convert files
    batch_convert(args.input_dir, args.output_dir, args.image_size)

    # Validate if requested
    if args.validate:
        print("\nValidating converted files...")
        output_path = Path(args.output_dir)
        npy_files = list(output_path.glob('*.npy'))
        passed = sum(1 for f in npy_files if validate_converted_file(f))
        print(f"\n{passed}/{len(npy_files)} files passed validation")
