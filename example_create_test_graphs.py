#!/usr/bin/env python3
"""
example_create_test_graphs.py

Creates example NetworkX graphs to demonstrate the conversion process.

This script generates simple floor plan graphs that can be converted
to GSDiff .npy format using gpickle_to_npy_converter.py.

Usage:
    python example_create_test_graphs.py --output_dir data/example_graphs --num_graphs 10
"""

import argparse
import networkx as nx
import numpy as np
from pathlib import Path


def create_simple_rectangular_floor_plan(file_id=0, width=200, height=150):
    """
    Create a simple rectangular floor plan with one room.

    Args:
        file_id: Unique identifier
        width: Width of the room
        height: Height of the room

    Returns:
        NetworkX graph
    """
    G = nx.Graph()

    # Add 4 corners
    G.add_node(0, x=0, y=0, semantic=0)
    G.add_node(1, x=width, y=0, semantic=0)
    G.add_node(2, x=width, y=height, semantic=0)
    G.add_node(3, x=0, y=height, semantic=0)

    # Add edges (walls)
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    return G


def create_two_room_floor_plan(file_id=0):
    """
    Create a floor plan with two rooms (living room + kitchen).

    Returns:
        NetworkX graph
    """
    G = nx.Graph()

    # Living room corners (semantic=0)
    G.add_node(0, x=0, y=0, semantic=0)
    G.add_node(1, x=100, y=0, semantic=0)
    G.add_node(2, x=100, y=100, semantic=0)
    G.add_node(3, x=0, y=100, semantic=0)

    # Kitchen corners (semantic=2)
    G.add_node(4, x=100, y=0, semantic=2)
    G.add_node(5, x=180, y=0, semantic=2)
    G.add_node(6, x=180, y=100, semantic=2)
    G.add_node(7, x=100, y=100, semantic=2)

    # Living room edges
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    # Kitchen edges
    G.add_edges_from([(4, 5), (5, 6), (6, 7), (7, 4)])

    # Shared wall between rooms
    # Note: nodes 1 and 4 have same position (100, 0) - shared corner
    # Similarly nodes 2 and 7 share position (100, 100)

    return G


def create_three_room_floor_plan(file_id=0):
    """
    Create a floor plan with three rooms (living, bedroom, bathroom).

    Returns:
        NetworkX graph
    """
    G = nx.Graph()

    # Living room corners (semantic=0)
    G.add_node(0, x=0, y=0, semantic=0)
    G.add_node(1, x=120, y=0, semantic=0)
    G.add_node(2, x=120, y=100, semantic=0)
    G.add_node(3, x=0, y=100, semantic=0)

    # Bedroom corners (semantic=1)
    G.add_node(4, x=120, y=0, semantic=1)
    G.add_node(5, x=200, y=0, semantic=1)
    G.add_node(6, x=200, y=70, semantic=1)
    G.add_node(7, x=120, y=70, semantic=1)

    # Bathroom corners (semantic=3)
    G.add_node(8, x=120, y=70, semantic=3)
    G.add_node(9, x=200, y=70, semantic=3)
    G.add_node(10, x=200, y=100, semantic=3)
    G.add_node(11, x=120, y=100, semantic=3)

    # Living room edges
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    # Bedroom edges
    G.add_edges_from([(4, 5), (5, 6), (6, 7), (7, 4)])

    # Bathroom edges
    G.add_edges_from([(8, 9), (9, 10), (10, 11), (11, 8)])

    return G


def create_complex_floor_plan(file_id=0):
    """
    Create a more complex floor plan with multiple rooms and irregular shapes.

    Returns:
        NetworkX graph
    """
    G = nx.Graph()

    # Define corners with positions and semantics
    corners = [
        # Living room
        (0, 0, 0, 0),      # (node_id, x, y, semantic)
        (1, 150, 0, 0),
        (2, 150, 120, 0),
        (3, 0, 120, 0),

        # Kitchen (L-shaped)
        (4, 150, 0, 2),
        (5, 220, 0, 2),
        (6, 220, 80, 2),
        (7, 180, 80, 2),
        (8, 180, 120, 2),
        (9, 150, 120, 2),

        # Bedroom 1
        (10, 0, 120, 1),
        (11, 100, 120, 1),
        (12, 100, 200, 1),
        (13, 0, 200, 1),

        # Bathroom
        (14, 100, 120, 3),
        (15, 150, 120, 3),
        (16, 150, 160, 3),
        (17, 100, 160, 3),

        # Bedroom 2
        (18, 150, 120, 7),
        (19, 220, 120, 7),
        (20, 220, 200, 7),
        (21, 150, 200, 7),
    ]

    # Add nodes
    for node_id, x, y, semantic in corners:
        G.add_node(node_id, x=x, y=y, semantic=semantic)

    # Add edges to form rooms
    edges = [
        # Living room
        (0, 1), (1, 2), (2, 3), (3, 0),

        # Kitchen (L-shaped)
        (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4),

        # Bedroom 1
        (10, 11), (11, 12), (12, 13), (13, 10),

        # Bathroom
        (14, 15), (15, 16), (16, 17), (17, 14),

        # Bedroom 2
        (18, 19), (19, 20), (20, 21), (21, 18),

        # Connections between rooms (shared walls)
        (2, 9), (3, 10), (11, 14), (15, 18)
    ]

    G.add_edges_from(edges)

    return G


def create_random_floor_plan(file_id=0, n_rooms=3, seed=None):
    """
    Create a random floor plan with specified number of rooms.

    Args:
        file_id: Unique identifier
        n_rooms: Number of rooms to generate
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph
    """
    if seed is not None:
        np.random.seed(seed + file_id)

    G = nx.Graph()

    # Room types to sample from
    room_types = [0, 1, 2, 3]  # Living, bedroom, kitchen, bathroom

    node_id = 0
    x_offset = 0

    for room_idx in range(n_rooms):
        # Random room dimensions
        width = np.random.randint(80, 150)
        height = np.random.randint(80, 150)

        # Random room type
        semantic = np.random.choice(room_types)

        # Create room corners
        corners = [
            (x_offset, 0),
            (x_offset + width, 0),
            (x_offset + width, height),
            (x_offset, height)
        ]

        # Add nodes
        start_node = node_id
        for (x, y) in corners:
            G.add_node(node_id, x=x, y=y, semantic=semantic)
            node_id += 1

        # Add edges for this room
        for i in range(4):
            G.add_edge(start_node + i, start_node + (i + 1) % 4)

        # Update offset for next room
        x_offset += width

    return G


def generate_example_graphs(output_dir, num_graphs=10, split=None):
    """
    Generate example floor plan graphs.

    Args:
        output_dir: Directory to save .gpickle files
        num_graphs: Number of graphs to generate
        split: Optional dict with 'train', 'val', 'test' counts

    Returns:
        Dictionary with statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_graphs} example floor plan graphs...")
    print(f"Output directory: {output_dir}")

    # Graph generation functions
    generators = [
        create_simple_rectangular_floor_plan,
        create_two_room_floor_plan,
        create_three_room_floor_plan,
        create_complex_floor_plan,
    ]

    stats = {'total': 0, 'by_type': {}}

    for i in range(num_graphs):
        # Alternate between different graph types
        if i < 4:
            # First 4: one of each type
            generator = generators[i]
            graph_type = generator.__name__
        else:
            # Rest: random floor plans
            generator = create_random_floor_plan
            graph_type = 'random'

        # Generate graph
        G = generator(file_id=i)

        # Save to gpickle
        output_path = output_dir / f"{i}.gpickle"
        nx.write_gpickle(G, output_path)

        # Update statistics
        stats['total'] += 1
        stats['by_type'][graph_type] = stats['by_type'].get(graph_type, 0) + 1

        # Print info
        print(f"  [{i:3d}] {graph_type:40s} | Nodes: {len(G.nodes()):3d} | Edges: {len(G.edges()):3d}")

    print(f"\n{'='*70}")
    print(f"Generated {stats['total']} graphs successfully!")
    print(f"\nBreakdown by type:")
    for graph_type, count in stats['by_type'].items():
        print(f"  - {graph_type}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate example NetworkX graphs for GSDiff conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 10 example graphs
    python example_create_test_graphs.py --output_dir data/example_graphs --num_graphs 10

    # Generate 100 random graphs
    python example_create_test_graphs.py --output_dir data/my_graphs --num_graphs 100 --seed 42

Then convert to .npy:
    python gpickle_to_npy_converter.py --input_dir data/example_graphs --output_dir data/example_npy
        """
    )

    parser.add_argument('--output_dir', type=str, default='data/example_graphs',
                       help='Output directory for .gpickle files')
    parser.add_argument('--num_graphs', type=int, default=10,
                       help='Number of graphs to generate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set global seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate graphs
    stats = generate_example_graphs(
        args.output_dir,
        args.num_graphs
    )

    print(f"\n{'='*70}")
    print("Next steps:")
    print(f"  1. Verify graphs:")
    print(f"     python -c \"import networkx as nx; G = nx.read_gpickle('{args.output_dir}/0.gpickle'); print(f'Nodes: {{len(G.nodes())}}, Edges: {{len(G.edges())}}')")
    print(f"\n  2. Convert to .npy:")
    print(f"     python gpickle_to_npy_converter.py --input_dir {args.output_dir} --output_dir data/example_npy")
    print(f"\n  3. Verify conversion:")
    print(f"     python -c \"import numpy as np; g = np.load('data/example_npy/0.npy', allow_pickle=True).item(); print(f'Corners: {{int(g[\\\"padding_mask\\\"].sum())}}')")


if __name__ == '__main__':
    main()
