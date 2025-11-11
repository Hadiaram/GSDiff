#!/usr/bin/env python3
"""
example_create_test_json.py

Creates example JSON files in all supported formats to demonstrate the conversion process.

This script generates simple floor plan JSON files that can be converted
to GSDiff .npy format using json_to_npy_converter.py.

Usage:
    python example_create_test_json.py --output_dir data/example_json --num_files 10
"""

import argparse
import json
import numpy as np
from pathlib import Path


def create_node_edge_json(file_id=0, width=200, height=150):
    """
    Create a simple rectangular floor plan in node-edge format.

    Args:
        file_id: Unique identifier
        width: Width of the room
        height: Height of the room

    Returns:
        Dictionary in node-edge format
    """
    floor_plan = {
        "nodes": [
            {"id": 0, "x": 0, "y": 0, "semantic": 0},
            {"id": 1, "x": width, "y": 0, "semantic": 0},
            {"id": 2, "x": width, "y": height, "semantic": 2},
            {"id": 3, "x": 0, "y": height, "semantic": 2}
        ],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
            {"source": 3, "target": 0}
        ]
    }
    return floor_plan


def create_corners_adjacency_json(file_id=0):
    """
    Create a two-room floor plan in corners-adjacency format.

    Returns:
        Dictionary in corners-adjacency format
    """
    floor_plan = {
        "corners": [
            # Living room
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
            # Kitchen
            [100, 0],
            [180, 0],
            [180, 100],
            [100, 100]
        ],
        "adjacency": [
            # Living room edges
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            # Kitchen edges
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4]
        ],
        "semantics": [0, 0, 0, 0, 2, 2, 2, 2]
    }
    return floor_plan


def create_corners_adjacency_list_json(file_id=0):
    """
    Create a three-room floor plan in corners-adjacency_list format.

    Returns:
        Dictionary with adjacency_list instead of adjacency array
    """
    floor_plan = {
        "corners": [
            # Living room
            [0, 0],
            [120, 0],
            [120, 100],
            [0, 100],
            # Bedroom
            [120, 0],
            [200, 0],
            [200, 70],
            [120, 70],
            # Bathroom
            [120, 70],
            [200, 70],
            [200, 100],
            [120, 100]
        ],
        "adjacency_list": {
            "0": [1, 3],
            "1": [0, 2, 4],
            "2": [1, 3, 11],
            "3": [0, 2],
            "4": [1, 5, 7],
            "5": [4, 6],
            "6": [5, 7],
            "7": [4, 6, 8],
            "8": [7, 9, 11],
            "9": [8, 10],
            "10": [9, 11],
            "11": [2, 8, 10]
        },
        "semantics": [0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3]
    }
    return floor_plan


def create_rooms_json(file_id=0):
    """
    Create a three-room floor plan in rooms format.

    Returns:
        Dictionary in rooms format
    """
    floor_plan = {
        "rooms": [
            {
                "type": "living_room",
                "corners": [
                    [0, 0],
                    [120, 0],
                    [120, 100],
                    [0, 100]
                ]
            },
            {
                "type": "bedroom",
                "corners": [
                    [120, 0],
                    [200, 0],
                    [200, 70],
                    [120, 70]
                ]
            },
            {
                "type": "bathroom",
                "corners": [
                    [120, 70],
                    [200, 70],
                    [200, 100],
                    [120, 100]
                ]
            }
        ]
    }
    return floor_plan


def create_complex_rooms_json(file_id=0):
    """
    Create a complex multi-room floor plan in rooms format.

    Returns:
        Dictionary in rooms format
    """
    floor_plan = {
        "rooms": [
            {
                "type": 0,  # Living room
                "corners": [[0, 0], [150, 0], [150, 120], [0, 120]]
            },
            {
                "type": 2,  # Kitchen (L-shaped)
                "corners": [[150, 0], [220, 0], [220, 80], [180, 80], [180, 120], [150, 120]]
            },
            {
                "type": 1,  # Bedroom 1
                "corners": [[0, 120], [100, 120], [100, 200], [0, 200]]
            },
            {
                "type": 3,  # Bathroom
                "corners": [[100, 120], [150, 120], [150, 160], [100, 160]]
            },
            {
                "type": 7,  # Bedroom 2
                "corners": [[150, 120], [220, 120], [220, 200], [150, 200]]
            }
        ]
    }
    return floor_plan


def create_random_node_edge_json(file_id=0, n_rooms=3, seed=None):
    """
    Create a random floor plan in node-edge format.

    Args:
        file_id: Unique identifier
        n_rooms: Number of rooms
        seed: Random seed

    Returns:
        Dictionary in node-edge format
    """
    if seed is not None:
        np.random.seed(seed + file_id)

    room_types = [0, 1, 2, 3]  # Living, bedroom, kitchen, bathroom
    nodes = []
    edges = []
    node_id = 0
    x_offset = 0

    for room_idx in range(n_rooms):
        # Random dimensions
        width = np.random.randint(80, 150)
        height = np.random.randint(80, 150)
        semantic = np.random.choice(room_types)

        # Create room corners
        room_nodes = [
            {"id": node_id, "x": x_offset, "y": 0, "semantic": semantic},
            {"id": node_id + 1, "x": x_offset + width, "y": 0, "semantic": semantic},
            {"id": node_id + 2, "x": x_offset + width, "y": height, "semantic": semantic},
            {"id": node_id + 3, "x": x_offset, "y": height, "semantic": semantic}
        ]
        nodes.extend(room_nodes)

        # Create room edges
        room_edges = [
            {"source": node_id, "target": node_id + 1},
            {"source": node_id + 1, "target": node_id + 2},
            {"source": node_id + 2, "target": node_id + 3},
            {"source": node_id + 3, "target": node_id}
        ]
        edges.extend(room_edges)

        node_id += 4
        x_offset += width

    return {"nodes": nodes, "edges": edges}


def generate_example_json_files(output_dir, num_files=10, seed=None):
    """
    Generate example JSON files in all supported formats.

    Args:
        output_dir: Directory to save JSON files
        num_files: Number of files to generate
        seed: Random seed for reproducibility

    Returns:
        Statistics dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_files} example JSON files...")
    print(f"Output directory: {output_dir}")
    print()

    generators = [
        ("node-edge", create_node_edge_json),
        ("corners-adjacency", create_corners_adjacency_json),
        ("corners-adjacency-list", create_corners_adjacency_list_json),
        ("rooms", create_rooms_json),
        ("complex-rooms", create_complex_rooms_json),
    ]

    stats = {'total': 0, 'by_format': {}}

    for i in range(num_files):
        # Cycle through different formats
        if i < len(generators):
            format_name, generator = generators[i]
            floor_plan = generator(file_id=i)
        else:
            # Rest: random node-edge format
            format_name = "random-node-edge"
            floor_plan = create_random_node_edge_json(file_id=i, seed=seed)

        # Save to JSON
        output_path = output_dir / f"{i}.json"
        with open(output_path, 'w') as f:
            json.dump(floor_plan, f, indent=2)

        # Update statistics
        stats['total'] += 1
        stats['by_format'][format_name] = stats['by_format'].get(format_name, 0) + 1

        # Count nodes/corners
        if 'nodes' in floor_plan:
            n_nodes = len(floor_plan['nodes'])
            n_edges = len(floor_plan['edges'])
            print(f"  [{i:3d}] {format_name:30s} | Nodes: {n_nodes:3d} | Edges: {n_edges:3d}")
        elif 'corners' in floor_plan:
            n_corners = len(floor_plan['corners'])
            if 'adjacency' in floor_plan:
                n_edges = len(floor_plan['adjacency'])
            elif 'adjacency_list' in floor_plan:
                n_edges = sum(len(neighbors) for neighbors in floor_plan['adjacency_list'].values()) // 2
            else:
                n_edges = 0
            print(f"  [{i:3d}] {format_name:30s} | Corners: {n_corners:3d} | Edges: {n_edges:3d}")
        elif 'rooms' in floor_plan:
            n_rooms = len(floor_plan['rooms'])
            n_corners = sum(len(room['corners']) for room in floor_plan['rooms'])
            print(f"  [{i:3d}] {format_name:30s} | Rooms: {n_rooms:3d} | Total corners: {n_corners:3d}")

    print(f"\n{'='*70}")
    print(f"Generated {stats['total']} JSON files successfully!")
    print(f"\nBreakdown by format:")
    for format_name, count in stats['by_format'].items():
        print(f"  - {format_name}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate example JSON files for GSDiff conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 10 example JSON files
    python example_create_test_json.py --output_dir data/example_json --num_files 10

    # Generate 100 random JSON files
    python example_create_test_json.py --output_dir data/my_json --num_files 100 --seed 42

Then convert to .npy:
    python json_to_npy_converter.py --input_dir data/example_json --output_dir data/example_npy

Supported formats:
    1. Node-edge: {"nodes": [...], "edges": [...]}
    2. Corners-adjacency: {"corners": [...], "adjacency": [...]}
    3. Corners-adjacency-list: {"corners": [...], "adjacency_list": {...}}
    4. Rooms: {"rooms": [...]}
        """
    )

    parser.add_argument('--output_dir', type=str, default='data/example_json',
                       help='Output directory for JSON files')
    parser.add_argument('--num_files', type=int, default=10,
                       help='Number of files to generate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set global seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate JSON files
    stats = generate_example_json_files(
        args.output_dir,
        args.num_files,
        args.seed
    )

    print(f"\n{'='*70}")
    print("Next steps:")
    print(f"  1. Verify JSON file:")
    print(f"     python -c \"import json; print(json.load(open('{args.output_dir}/0.json')))\"")
    print(f"\n  2. Convert to .npy:")
    print(f"     python json_to_npy_converter.py --input_dir {args.output_dir} --output_dir data/example_npy")
    print(f"\n  3. Verify conversion:")
    print(f"     python -c \"import numpy as np; g = np.load('data/example_npy/0.npy', allow_pickle=True).item(); print(f'Corners: {{int(g[\\\"padding_mask\\\"].sum())}}')\"")


if __name__ == '__main__':
    main()
