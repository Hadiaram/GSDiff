#!/usr/bin/env python3
"""
json_to_npy_converter.py

Converts JSON floor plan files to GSDiff-compatible .npy format.

Supports multiple JSON formats:
1. Node-edge format: {"nodes": [...], "edges": [...]}
2. Corners-adjacency format: {"corners": [...], "adjacency": [...]}
3. Room-based format: {"rooms": [{corners: [...], type: ...}]}
4. BIM format: {"rooms": [...], "walls": [...]} - Revit/BIM exports

Usage:
    python json_to_npy_converter.py --input_dir data/json_files --output_dir data/npy_files
    python json_to_npy_converter.py --input data/floor_plan.json --output data/floor_plan.npy
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


# ============================================================================
# JSON Format Detection and Parsing
# ============================================================================

def detect_json_format(data: Dict) -> str:
    """
    Detect the format of the JSON file.

    Args:
        data: Parsed JSON dictionary

    Returns:
        Format type: 'node-edge', 'corners-adjacency', 'rooms', 'bim', or 'unknown'
    """
    if 'nodes' in data and 'edges' in data:
        return 'node-edge'
    elif 'corners' in data and ('adjacency' in data or 'adjacency_list' in data or 'adjacency_matrix' in data):
        return 'corners-adjacency'
    elif 'rooms' in data and 'walls' in data:
        # BIM format: rooms reference walls by ID, walls have coordinates
        return 'bim'
    elif 'rooms' in data:
        return 'rooms'
    else:
        return 'unknown'


def parse_bim_format(data: Dict) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Parse BIM format JSON (Revit, ArchiCAD, etc.).

    Format:
    {
        "rooms": [
            {
                "bounding_box": {"min": [x, y, z], "max": [x, y, z]},
                "walls": [wall_id1, wall_id2, ...],
                "name": "LIVING ROOM",
                "id": 12345
            },
            ...
        ],
        "walls": [
            {
                "start": [x, y, z],
                "end": [x, y, z],
                "id": wall_id
            },
            ...
        ]
    }

    Returns:
        corners_list: List of (x, y) tuples
        edges_list: List of (node1, node2) tuples
        semantics: Dict mapping node index to semantic label
    """
    
    # Room type mapping
    room_type_map = {
        'living': 0, 'living_room': 0, 'dining': 0, 'entrance': 0, 'foyer': 0,
        'master_bedroom': 1, 'bedroom': 1,
        'kitchen': 2,
        'bathroom': 3, 'bath': 3, 'powder': 3,
        'dining_room': 4,
        'child_room': 5,
        'study_room': 6, 'study': 6,
        'second_bedroom': 7,
        'guest_room': 8,
        'balcony': 9,
        'entrance_hall': 10,
        'storage': 11, 'storeroom': 11,
        'closet': 12, 'walk_in': 12, 'dressing': 12,
        'external': 13, 'outside': 13,
    }
    
    # Build wall ID to geometry mapping
    wall_dict = {}
    for wall in data['walls']:
        wall_id = wall['id']
        # Extract 2D coordinates (ignore Z)
        start = (wall['start'][0], wall['start'][1])
        end = (wall['end'][0], wall['end'][1])
        wall_dict[wall_id] = {'start': start, 'end': end}
    
    # Process each room
    corners_list = []
    edges_list = []
    semantics = {}
    corner_to_idx = {}
    
    def get_or_create_corner(coord: Tuple[float, float], room_type: int) -> int:
        """Get existing corner index or create new one"""
        # Round to avoid floating point issues
        rounded = (round(coord[0], 6), round(coord[1], 6))
        if rounded in corner_to_idx:
            return corner_to_idx[rounded]
        else:
            idx = len(corners_list)
            corners_list.append(coord)
            corner_to_idx[rounded] = idx
            semantics[idx] = room_type
            return idx
    
    for room in data['rooms']:
        # Get room type
        room_name = room.get('name', '').lower()
        room_type = 0  # default to living room
        
        for keyword, type_id in room_type_map.items():
            if keyword in room_name:
                room_type = type_id
                break
        
        # Get walls for this room
        room_wall_ids = room.get('walls', [])
        if not room_wall_ids:
            # Skip rooms with no walls (like balconies)
            continue
        
        # Collect all endpoints from room's walls
        room_corners = []
        room_wall_segments = []
        
        for wall_id in room_wall_ids:
            if wall_id in wall_dict:
                wall_geom = wall_dict[wall_id]
                room_wall_segments.append((wall_geom['start'], wall_geom['end']))
        
        if not room_wall_segments:
            continue
        
        # Build a graph of wall connections to find the room boundary
        endpoint_connections = defaultdict(list)
        for start, end in room_wall_segments:
            # Round coordinates for matching
            start_rounded = (round(start[0], 4), round(start[1], 4))
            end_rounded = (round(end[0], 4), round(end[1], 4))
            endpoint_connections[start_rounded].append(end_rounded)
            endpoint_connections[end_rounded].append(start_rounded)
        
        # Find a closed loop of corners
        if endpoint_connections:
            # Start from any point
            start_point = list(endpoint_connections.keys())[0]
            ordered_corners = [start_point]
            current = start_point
            visited = {start_point}
            
            # Trace the boundary
            while True:
                neighbors = [n for n in endpoint_connections[current] if n not in visited]
                if not neighbors:
                    break
                next_point = neighbors[0]
                ordered_corners.append(next_point)
                visited.add(next_point)
                current = next_point
                
                # Stop if we've closed the loop
                if len(ordered_corners) > 2 and current in endpoint_connections[start_point]:
                    break
                
                # Prevent infinite loops
                if len(ordered_corners) > 100:
                    break
            
            # Create corners and edges
            room_corner_indices = []
            for corner in ordered_corners:
                # Convert back to original precision
                corner_coord = corner
                idx = get_or_create_corner(corner_coord, room_type)
                room_corner_indices.append(idx)
            
            # Add edges forming the room boundary
            for i in range(len(room_corner_indices)):
                edge = (room_corner_indices[i], room_corner_indices[(i + 1) % len(room_corner_indices)])
                edges_list.append(edge)
    
    return corners_list, edges_list, semantics


def parse_node_edge_format(data: Dict) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Parse node-edge format JSON.

    Format:
    {
        "nodes": [
            {"id": 0, "x": 50, "y": 50, "semantic": 0},
            {"id": 1, "x": 200, "y": 50, "semantic": 0},
            ...
        ],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            ...
        ]
    }

    Returns:
        corners_list: List of (x, y) tuples
        edges_list: List of (node1, node2) tuples
        semantics: Dict mapping node index to semantic label
    """
    nodes = data['nodes']
    edges = data['edges']

    # Build node ID to index mapping
    node_id_to_idx = {}
    corners_list = []
    semantics = {}

    for idx, node in enumerate(nodes):
        # Handle different coordinate formats
        if 'x' in node and 'y' in node:
            x, y = node['x'], node['y']
        elif 'pos' in node:
            x, y = node['pos']
        elif 'position' in node:
            x, y = node['position']
        elif 'coordinates' in node:
            x, y = node['coordinates']
        else:
            raise ValueError(f"Node {node.get('id', idx)} missing coordinate information")

        corners_list.append((float(x), float(y)))

        # Get node ID
        node_id = node.get('id', idx)
        node_id_to_idx[node_id] = idx

        # Get semantic label (default to 0)
        semantic = node.get('semantic', node.get('type', node.get('label', 0)))
        semantics[idx] = int(semantic)

    # Parse edges
    edges_list = []
    for edge in edges:
        # Handle different edge formats
        if 'source' in edge and 'target' in edge:
            src, tgt = edge['source'], edge['target']
        elif 'from' in edge and 'to' in edge:
            src, tgt = edge['from'], edge['to']
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            src, tgt = edge
        else:
            raise ValueError(f"Edge format not recognized: {edge}")

        # Convert node IDs to indices
        src_idx = node_id_to_idx.get(src, src)
        tgt_idx = node_id_to_idx.get(tgt, tgt)
        edges_list.append((src_idx, tgt_idx))

    return corners_list, edges_list, semantics


def parse_corners_adjacency_format(data: Dict) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Parse corners-adjacency format JSON.

    Format:
    {
        "corners": [
            [50, 50],
            [200, 50],
            [200, 150],
            ...
        ],
        "adjacency": [[0, 1], [1, 2], [2, 3], ...],
        "semantics": [0, 0, 2, 2, ...]  // optional
    }

    Or with adjacency_list:
    {
        "corners": [[50, 50], [200, 50], ...],
        "adjacency_list": {
            "0": [1, 3],
            "1": [0, 2],
            ...
        }
    }

    Or with adjacency_matrix:
    {
        "corners": [[50, 50], [200, 50], ...],
        "adjacency_matrix": [[0, 1, 0, 1], [1, 0, 1, 0], ...]
    }
    """
    corners = data['corners']

    # Parse corners
    corners_list = []
    for corner in corners:
        if isinstance(corner, (list, tuple)):
            x, y = corner[0], corner[1]
        elif isinstance(corner, dict):
            x = corner.get('x', corner.get('X', corner.get('0', 0)))
            y = corner.get('y', corner.get('Y', corner.get('1', 0)))
        else:
            raise ValueError(f"Corner format not recognized: {corner}")
        corners_list.append((float(x), float(y)))

    # Parse semantics
    semantics = {}
    if 'semantics' in data:
        sem_list = data['semantics']
        for idx, sem in enumerate(sem_list):
            semantics[idx] = int(sem)
    else:
        # Default to 0
        for idx in range(len(corners_list)):
            semantics[idx] = 0

    # Parse adjacency
    edges_list = []

    if 'adjacency' in data:
        # List of edges
        for edge in data['adjacency']:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                edges_list.append((int(edge[0]), int(edge[1])))
            else:
                raise ValueError(f"Edge format not recognized: {edge}")

    elif 'adjacency_list' in data:
        # Adjacency list format
        adj_list = data['adjacency_list']
        added = set()
        for node_str, neighbors in adj_list.items():
            node = int(node_str)
            for neighbor in neighbors:
                neighbor = int(neighbor)
                edge = tuple(sorted([node, neighbor]))
                if edge not in added:
                    edges_list.append(edge)
                    added.add(edge)

    elif 'adjacency_matrix' in data:
        # Adjacency matrix format
        adj_matrix = data['adjacency_matrix']
        n = len(adj_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 1:
                    edges_list.append((i, j))

    else:
        raise ValueError("No adjacency information found in JSON")

    return corners_list, edges_list, semantics


def parse_rooms_format(data: Dict) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Parse room-based format JSON.

    Format:
    {
        "rooms": [
            {
                "type": 0,  // or "living_room"
                "corners": [[0, 0], [100, 0], [100, 100], [0, 100]]
            },
            {
                "type": 2,  // or "kitchen"
                "corners": [[100, 0], [200, 0], [200, 100], [100, 100]]
            },
            ...
        ]
    }
    """
    rooms = data['rooms']

    # Map room type strings to integers
    room_type_map = {
        'living_room': 0, 'living': 0, 'entrance': 0, 'dining': 0,
        'master_bedroom': 1, 'bedroom': 1,
        'kitchen': 2,
        'bathroom': 3, 'bath': 3,
        'dining_room': 4,
        'child_room': 5,
        'study_room': 6, 'study': 6,
        'second_bedroom': 7,
        'guest_room': 8,
        'balcony': 9,
        'entrance_hall': 10,
        'storage': 11, 'storeroom': 11,
        'closet': 12, 'walk_in': 12,
        'external': 13, 'outside': 13,
    }

    corners_list = []
    edges_list = []
    semantics = {}
    corner_to_idx = {}

    for room in rooms:
        room_corners = room['corners']
        room_type = room.get('type', room.get('semantic', room.get('label', 0)))

        # Convert string type to integer
        if isinstance(room_type, str):
            room_type = room_type_map.get(room_type.lower().replace(' ', '_'), 0)
        room_type = int(room_type)

        # Process corners
        room_indices = []
        for corner in room_corners:
            if isinstance(corner, (list, tuple)):
                x, y = float(corner[0]), float(corner[1])
            elif isinstance(corner, dict):
                x = float(corner.get('x', corner.get('X', 0)))
                y = float(corner.get('y', corner.get('Y', 0)))
            else:
                raise ValueError(f"Corner format not recognized: {corner}")

            # Check if corner already exists (shared corners between rooms)
            corner_key = (x, y)
            if corner_key in corner_to_idx:
                idx = corner_to_idx[corner_key]
            else:
                idx = len(corners_list)
                corners_list.append((x, y))
                corner_to_idx[corner_key] = idx

            room_indices.append(idx)
            semantics[idx] = room_type

        # Add edges connecting corners in order (forming room boundary)
        for i in range(len(room_indices)):
            edge = (room_indices[i], room_indices[(i + 1) % len(room_indices)])
            edges_list.append(edge)

    return corners_list, edges_list, semantics


def load_json_graph(json_path: str) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Load a JSON file and parse it into corners, edges, and semantics.

    Args:
        json_path: Path to JSON file

    Returns:
        corners_list: List of (x, y) tuples
        edges_list: List of (node1, node2) tuples
        semantics: Dict mapping node index to semantic label
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Detect format
    format_type = detect_json_format(data)

    if format_type == 'node-edge':
        return parse_node_edge_format(data)
    elif format_type == 'corners-adjacency':
        return parse_corners_adjacency_format(data)
    elif format_type == 'bim':
        return parse_bim_format(data)
    elif format_type == 'rooms':
        return parse_rooms_format(data)
    else:
        raise ValueError(
            f"Unrecognized JSON format. Supported formats:\n"
            f"1. Node-edge: {{'nodes': [...], 'edges': [...]}}\n"
            f"2. Corners-adjacency: {{'corners': [...], 'adjacency': [...]}}\n"
            f"3. Rooms: {{'rooms': [...]}}\\n"
            f"4. BIM: {{'rooms': [...], 'walls': [...]}}\n"
            f"See documentation for details."
        )


# ============================================================================
# Coordinate Normalization
# ============================================================================

def normalize_coordinates(corners_list: List[Tuple[float, float]],
                         coord_range: Tuple[float, float] = (0, 256)) -> np.ndarray:
    """
    Normalize coordinates to [-1, 1] range.

    Args:
        corners_list: List of (x, y) coordinate tuples
        coord_range: Expected range of input coordinates (min, max) or 'auto'

    Returns:
        Normalized coordinates as numpy array (n, 2)
    """
    corners_np = np.array(corners_list, dtype=np.float64)

    if coord_range == 'auto' or coord_range is None:
        # Auto-detect range from data
        min_coords = corners_np.min(axis=0)
        max_coords = corners_np.max(axis=0)
        center = (min_coords + max_coords) / 2.0
        scale = (max_coords - min_coords).max() / 2.0
    else:
        min_val, max_val = coord_range
        center = (min_val + max_val) / 2.0
        scale = (max_val - min_val) / 2.0

    if scale == 0:
        scale = 1.0  # Avoid division by zero

    normalized = (corners_np - center) / scale

    return normalized


# ============================================================================
# Graph Processing (from gpickle_to_npy_converter.py)
# ============================================================================

def create_adjacency_structures(n_corners: int, edges_list: List[Tuple[int, int]]) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    """
    Create adjacency matrix and adjacency list from edges.

    Args:
        n_corners: Number of corners
        edges_list: List of (node1, node2) tuples

    Returns:
        adjacency_matrix: 2D list
        adjacency_list: Dict mapping node to list of neighbors
    """
    adjacency_matrix = [[0] * n_corners for _ in range(n_corners)]
    adjacency_list = {i: [] for i in range(n_corners)}

    for node1, node2 in edges_list:
        adjacency_matrix[node1][node2] = 1
        adjacency_matrix[node2][node1] = 1
        if node2 not in adjacency_list[node1]:
            adjacency_list[node1].append(node2)
        if node1 not in adjacency_list[node2]:
            adjacency_list[node2].append(node1)

    return adjacency_matrix, adjacency_list


def create_padded_arrays(corner_list_np_normalized: np.ndarray,
                        adjacency_matrix: List[List[int]],
                        max_corners: int = 53) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create padded arrays with fixed dimensions.

    Args:
        corner_list_np_normalized: Normalized corner coordinates (n, 2)
        adjacency_matrix: Adjacency matrix (n, n)
        max_corners: Maximum number of corners

    Returns:
        corner_list_np_normalized_padding: Padded corners (max_corners, 2)
        padding_mask: Mask indicating real corners (max_corners, 1)
        global_matrix_np_padding: Padded global matrix (max_corners, max_corners)
        adjacency_matrix_np_padding: Padded adjacency matrix (max_corners, max_corners)
    """
    n = len(corner_list_np_normalized)

    if n > max_corners:
        raise ValueError(f"Graph has {n} corners, which exceeds max_corners={max_corners}")

    # Padded corners
    corner_list_np_normalized_padding = np.zeros((max_corners, 2), dtype=np.float64)
    corner_list_np_normalized_padding[:n] = corner_list_np_normalized

    # Padding mask
    padding_mask = np.zeros((max_corners, 1), dtype=np.int32)
    padding_mask[:n] = 1

    # Global matrix (all ones for real corners)
    global_matrix_np_padding = np.zeros((max_corners, max_corners), dtype=np.int32)
    global_matrix_np_padding[:n, :n] = 1

    # Adjacency matrix
    adjacency_matrix_np = np.array(adjacency_matrix, dtype=np.int32)
    adjacency_matrix_np_padding = np.zeros((max_corners, max_corners), dtype=np.int32)
    adjacency_matrix_np_padding[:n, :n] = adjacency_matrix_np

    return (corner_list_np_normalized_padding, padding_mask,
            global_matrix_np_padding, adjacency_matrix_np_padding)


def create_edge_arrays(corner_list_np_normalized_padding: np.ndarray,
                      adjacency_matrix_np_padding: np.ndarray,
                      max_corners: int = 53) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create edge coordinate and edge label arrays.

    Args:
        corner_list_np_normalized_padding: Padded corner coordinates
        adjacency_matrix_np_padding: Padded adjacency matrix
        max_corners: Maximum number of corners

    Returns:
        edge_coords: Edge coordinates (max_corners^2, 4)
        edges: Edge labels (max_corners^2, 1)
    """
    max_edges = max_corners * max_corners
    edge_coords = np.zeros((max_edges, 4), dtype=np.float64)
    edges = np.zeros((max_edges, 1), dtype=np.int32)

    idx = 0
    for i in range(max_corners):
        for j in range(max_corners):
            x1, y1 = corner_list_np_normalized_padding[i]
            x2, y2 = corner_list_np_normalized_padding[j]
            edge_coords[idx] = [x1, y1, x2, y2]
            edges[idx] = adjacency_matrix_np_padding[i, j]
            idx += 1

    return edge_coords, edges


def create_semantic_vectors(corners_list: List[Tuple[float, float]],
                           semantics: Dict[int, int],
                           semantic_dim: int = 14) -> Dict[Tuple[float, float], List[int]]:
    """
    Create one-hot semantic vectors for each corner.

    Args:
        corners_list: List of (x, y) tuples
        semantics: Dict mapping corner index to semantic label
        semantic_dim: Number of semantic categories

    Returns:
        Dict mapping corner coordinates to one-hot semantic vector
    """
    semantic_dict = {}

    for idx, corner in enumerate(corners_list):
        semantic_label = semantics.get(idx, 0)
        one_hot = [0] * semantic_dim
        if 0 <= semantic_label < semantic_dim:
            one_hot[semantic_label] = 1
        else:
            one_hot[0] = 1  # Default to living room
        semantic_dict[corner] = one_hot

    return semantic_dict


def create_padded_semantics(corner_list_np_normalized: np.ndarray,
                            corner_list_np_normalized_padding: np.ndarray,
                            semantic_dict: Dict[Tuple[float, float], List[int]],
                            corners_list: List[Tuple[float, float]],
                            max_corners: int = 53,
                            semantic_dim: int = 14) -> np.ndarray:
    """
    Create padded array with coordinates and semantic labels.

    Args:
        corner_list_np_normalized: Normalized corners (n, 2)
        corner_list_np_normalized_padding: Padded corners (max_corners, 2)
        semantic_dict: Dict mapping original corners to semantic vectors
        corners_list: Original corner list
        max_corners: Maximum number of corners
        semantic_dim: Number of semantic categories

    Returns:
        Padded array with coordinates and semantics (max_corners, 2+semantic_dim)
    """
    n = len(corner_list_np_normalized)
    corner_list_np_normalized_padding_withsemantics = np.zeros(
        (max_corners, 2 + semantic_dim), dtype=np.float64
    )

    # Add coordinates
    corner_list_np_normalized_padding_withsemantics[:, :2] = corner_list_np_normalized_padding

    # Add semantics
    for idx in range(n):
        original_corner = corners_list[idx]
        semantic_vector = semantic_dict.get(original_corner, [1] + [0] * (semantic_dim - 1))
        corner_list_np_normalized_padding_withsemantics[idx, 2:] = semantic_vector

    return corner_list_np_normalized_padding_withsemantics


# ============================================================================
# Main Conversion Function
# ============================================================================

def convert_json_to_npy(json_path: str,
                       output_path: str,
                       max_corners: int = 53,
                       semantic_dim: int = 14,
                       coord_range: Tuple[float, float] = (0, 256),
                       file_id: Optional[int] = None) -> Dict:
    """
    Convert a single JSON file to GSDiff .npy format.

    Args:
        json_path: Path to input JSON file
        output_path: Path to output .npy file
        max_corners: Maximum number of corners (default: 53)
        semantic_dim: Number of semantic dimensions (default: 14)
        coord_range: Expected coordinate range (min, max) or 'auto'
        file_id: Optional file ID (extracted from filename if None)

    Returns:
        Dictionary with converted data

    Raises:
        ValueError: If JSON format is invalid
    """
    # Load JSON
    corners_list, edges_list, semantics = load_json_graph(json_path)

    # Extract file ID
    if file_id is None:
        try:
            file_id = int(Path(json_path).stem.split('_')[-1])
        except (ValueError, IndexError):
            file_id = abs(hash(Path(json_path).stem)) % (10**8)

    # Normalize coordinates
    corners_np = np.array(corners_list, dtype=np.float64)
    corner_list_np_normalized = normalize_coordinates(corners_list, coord_range)

    # Create adjacency structures
    n_corners = len(corners_list)
    adjacency_matrix, adjacency_list = create_adjacency_structures(n_corners, edges_list)

    # Create semantic vectors
    semantic_dict = create_semantic_vectors(corners_list, semantics, semantic_dim)

    # Create padded arrays
    (corner_list_np_normalized_padding, padding_mask,
     global_matrix_np_padding, adjacency_matrix_np_padding) = create_padded_arrays(
        corner_list_np_normalized, adjacency_matrix, max_corners
    )

    # Create edge arrays
    edge_coords, edges = create_edge_arrays(
        corner_list_np_normalized_padding, adjacency_matrix_np_padding, max_corners
    )

    # Create normalized semantic dict
    normalized_semantic_dict = {}
    for idx, corner in enumerate(corners_list):
        normalized_corner = tuple(corner_list_np_normalized[idx])
        normalized_semantic_dict[normalized_corner] = semantic_dict[corner]

    # Create padded semantics
    corner_list_np_normalized_padding_withsemantics = create_padded_semantics(
        corner_list_np_normalized, corner_list_np_normalized_padding,
        semantic_dict, corners_list, max_corners, semantic_dim
    )

    # Create output dictionary
    result = {
        'file_id': file_id,
        'corners': corners_list,
        'corners_np': corners_np,
        'adjacency_matrix': adjacency_matrix,
        'adjacency_list': adjacency_list,
        'corner_list_np_normalized': corner_list_np_normalized,
        'corner_list_np_normalized_padding': corner_list_np_normalized_padding,
        'padding_mask': padding_mask,
        'global_matrix_np_padding': global_matrix_np_padding,
        'adjacency_matrix_np_padding': adjacency_matrix_np_padding,
        'edge_coords': edge_coords,
        'edges': edges,
        'semantics': normalized_semantic_dict,
        'corner_list_np_normalized_padding_withsemantics': corner_list_np_normalized_padding_withsemantics,
    }

    # Save to .npy file
    np.save(output_path, result)

    return result


def convert_directory(input_dir: str,
                     output_dir: str,
                     max_corners: int = 53,
                     semantic_dim: int = 14,
                     coord_range: Tuple[float, float] = (0, 256),
                     train_val_test_split: Optional[Dict] = None) -> Dict:
    """
    Convert all JSON files in a directory.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Output directory for .npy files
        max_corners: Maximum number of corners
        semantic_dim: Number of semantic dimensions
        coord_range: Expected coordinate range or 'auto'
        train_val_test_split: Optional dict with 'train', 'val', 'test' lists

    Returns:
        Statistics dictionary
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_val_test_split:
        (output_dir / 'train').mkdir(exist_ok=True)
        (output_dir / 'val').mkdir(exist_ok=True)
        (output_dir / 'test').mkdir(exist_ok=True)

    # Find all JSON files
    json_files = sorted(input_dir.glob('*.json'))

    if not json_files:
        print(f"No .json files found in {input_dir}")
        return {'converted': 0, 'failed': 0}

    print(f"Found {len(json_files)} .json files")

    # Convert each file
    stats = {'converted': 0, 'failed': 0, 'errors': [], 'filename_mapping': {}}

    for idx, json_path in enumerate(tqdm(json_files, desc="Converting files")):
        try:
            # Use sequential numbering for file_id
            file_id = idx
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
            convert_json_to_npy(
                json_path,
                output_path,
                max_corners=max_corners,
                semantic_dim=semantic_dim,
                coord_range=coord_range,
                file_id=file_id
            )

            # Track filename mapping
            stats['filename_mapping'][str(json_path.name)] = file_id
            stats['converted'] += 1

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append((json_path.name, str(e)))
            print(f"\nError converting {json_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {stats['converted']}")
    print(f"Failed: {stats['failed']}")

    if stats['errors']:
        print(f"\nErrors:")
        for filename, error in stats['errors'][:10]:
            print(f"  - {filename}: {error}")

    if stats['filename_mapping'] and stats['converted'] > 0:
        print(f"\nFilename mapping (original → output):")
        for orig_name, file_id in sorted(stats['filename_mapping'].items(), key=lambda x: x[1])[:10]:
            print(f"  {orig_name} → {file_id}.npy")
        if len(stats['filename_mapping']) > 10:
            print(f"  ... and {len(stats['filename_mapping']) - 10} more")

    return stats


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON floor plan files to GSDiff .npy format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported JSON Formats:

1. Node-Edge Format:
   {
       "nodes": [
           {"id": 0, "x": 50, "y": 50, "semantic": 0},
           {"id": 1, "x": 200, "y": 50, "semantic": 0},
           ...
       ],
       "edges": [
           {"source": 0, "target": 1},
           {"source": 1, "target": 2},
           ...
       ]
   }

2. Corners-Adjacency Format:
   {
       "corners": [[50, 50], [200, 50], [200, 150], [50, 150]],
       "adjacency": [[0, 1], [1, 2], [2, 3], [3, 0]],
       "semantics": [0, 0, 2, 2]
   }

3. Rooms Format:
   {
       "rooms": [
           {
               "type": "living_room",
               "corners": [[0, 0], [100, 0], [100, 100], [0, 100]]
           },
           {
               "type": "kitchen",
               "corners": [[100, 0], [200, 0], [200, 100], [100, 100]]
           }
       ]
   }

4. BIM Format (Revit/ArchiCAD):
   {
       "rooms": [...],
       "walls": [{"id": 123, "start": [x,y,z], "end": [x,y,z]}, ...]
   }

Examples:
    # Convert directory (auto-detect coordinate range)
    python json_to_npy_converter.py --input_dir data/json_files --output_dir data/npy_files --coord_range auto

    # Convert single file
    python json_to_npy_converter.py --input data/floor_plan.json --output data/floor_plan.npy --coord_range auto

    # Specify coordinate range
    python json_to_npy_converter.py --input_dir data/json --output_dir data/npy --coord_range 0 1000
        """
    )

    # Input/output
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='Input JSON file')
    group.add_argument('--input_dir', type=str, help='Input directory containing JSON files')

    parser.add_argument('--output', type=str, help='Output .npy file (for single file conversion)')
    parser.add_argument('--output_dir', type=str, help='Output directory for .npy files')

    # Parameters
    parser.add_argument('--max_corners', type=int, default=53,
                       help='Maximum number of corners (default: 53)')
    parser.add_argument('--semantic_dim', type=int, default=14,
                       help='Number of semantic dimensions (default: 14)')
    parser.add_argument('--coord_range', nargs='*', default=['auto'],
                       metavar='MIN MAX',
                       help='Expected coordinate range: "auto" or "MIN MAX" (default: auto)')
    parser.add_argument('--split_file', type=str,
                       help='JSON file with train/val/test split')

    args = parser.parse_args()

    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")

    # Parse coord_range
    if args.coord_range == ['auto'] or args.coord_range[0] == 'auto':
        coord_range = 'auto'
    else:
        try:
            coord_range = tuple(float(x) for x in args.coord_range)
            if len(coord_range) != 2:
                parser.error("--coord_range must be 'auto' or two numbers")
        except ValueError:
            parser.error("--coord_range values must be numbers or 'auto'")

    # Load split file if provided
    train_val_test_split = None
    if args.split_file:
        with open(args.split_file, 'r') as f:
            train_val_test_split = json.load(f)

    # Convert
    if args.input:
        # Single file conversion
        print(f"Converting {args.input} to {args.output}")
        result = convert_json_to_npy(
            args.input,
            args.output,
            max_corners=args.max_corners,
            semantic_dim=args.semantic_dim,
            coord_range=coord_range
        )
        print(f"✓ Conversion successful!")
        print(f"  Corners: {int(result['padding_mask'].sum())}")
        print(f"  Edges: {int(result['adjacency_matrix_np_padding'][:int(result['padding_mask'].sum()), :int(result['padding_mask'].sum())].sum() // 2)}")

    else:
        # Directory conversion
        stats = convert_directory(
            args.input_dir,
            args.output_dir,
            max_corners=args.max_corners,
            semantic_dim=args.semantic_dim,
            coord_range=coord_range,
            train_val_test_split=train_val_test_split
        )


if __name__ == '__main__':
    main()