#!/usr/bin/env python3
"""
augment_floor_plans.py

Augment GSDiff NPY floor plan files with geometric transformations.
Preserves the graph structure and semantic information while applying:
- Horizontal flipping
- Vertical flipping
- 90° rotations
- Combinations of the above

Usage:
    # Augment all files in directory (creates 4x or 8x dataset)
    python augment_floor_plans.py --input_dir data/npy_files --output_dir data/npy_augmented

    # Augment single file
    python augment_floor_plans.py --input data/0.npy --output_dir data/augmented

    # Only use flips (2x dataset)
    python augment_floor_plans.py --input_dir data/npy --output_dir data/aug --flip_only

    # Full augmentation including rotations (8x dataset)
    python augment_floor_plans.py --input_dir data/npy --output_dir data/aug --full
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def flip_horizontal(coords):
    """Flip coordinates horizontally (mirror across Y-axis)."""
    flipped = coords.copy()
    flipped[:, 0] = -flipped[:, 0]  # Negate X coordinates
    return flipped


def flip_vertical(coords):
    """Flip coordinates vertically (mirror across X-axis)."""
    flipped = coords.copy()
    flipped[:, 1] = -flipped[:, 1]  # Negate Y coordinates
    return flipped


def rotate_90(coords):
    """Rotate coordinates 90° counterclockwise."""
    rotated = coords.copy()
    # Rotation matrix for 90°: [0, -1; 1, 0]
    # (x, y) -> (-y, x)
    new_x = -coords[:, 1]
    new_y = coords[:, 0]
    rotated[:, 0] = new_x
    rotated[:, 1] = new_y
    return rotated


def rotate_180(coords):
    """Rotate coordinates 180°."""
    rotated = coords.copy()
    # (x, y) -> (-x, -y)
    rotated[:, 0] = -coords[:, 0]
    rotated[:, 1] = -coords[:, 1]
    return rotated


def rotate_270(coords):
    """Rotate coordinates 270° counterclockwise (or 90° clockwise)."""
    rotated = coords.copy()
    # Rotation matrix for 270°: [0, 1; -1, 0]
    # (x, y) -> (y, -x)
    new_x = coords[:, 1]
    new_y = -coords[:, 0]
    rotated[:, 0] = new_x
    rotated[:, 1] = new_y
    return rotated


def apply_transformation(graph_dict, transform_func, transform_name):
    """
    Apply a transformation to a GSDiff graph dictionary.

    Args:
        graph_dict: Dictionary containing graph data
        transform_func: Function to apply to coordinates
        transform_name: Name of transformation (for metadata)

    Returns:
        Transformed graph dictionary
    """
    # Create a copy to avoid modifying original
    transformed = {}

    # Transform corner coordinates
    if 'corner_list_np_normalized' in graph_dict:
        transformed['corner_list_np_normalized'] = transform_func(
            graph_dict['corner_list_np_normalized']
        )

    if 'corner_list_np_normalized_padding' in graph_dict:
        transformed['corner_list_np_normalized_padding'] = transform_func(
            graph_dict['corner_list_np_normalized_padding']
        )

    # Transform corners with semantics (keep semantics unchanged)
    if 'corner_list_np_normalized_padding_withsemantics' in graph_dict:
        original = graph_dict['corner_list_np_normalized_padding_withsemantics']
        coords = original[:, :2]  # First 2 columns are coordinates
        semantics = original[:, 2:]  # Remaining columns are semantics

        transformed_coords = transform_func(coords)
        transformed['corner_list_np_normalized_padding_withsemantics'] = np.concatenate(
            [transformed_coords, semantics], axis=1
        )

    # Transform edge coordinates
    if 'edge_coords' in graph_dict:
        edge_coords = graph_dict['edge_coords']
        # Edge coords are [x1, y1, x2, y2]
        start_coords = edge_coords[:, :2]
        end_coords = edge_coords[:, 2:]

        transformed_start = transform_func(start_coords)
        transformed_end = transform_func(end_coords)

        transformed['edge_coords'] = np.concatenate(
            [transformed_start, transformed_end], axis=1
        )

    # Copy non-coordinate fields unchanged
    preserve_fields = [
        'file_id', 'adjacency_matrix', 'adjacency_list', 'padding_mask',
        'global_matrix_np_padding', 'adjacency_matrix_np_padding',
        'edges', 'semantics', 'corners', 'corners_np'
    ]

    for field in preserve_fields:
        if field in graph_dict:
            transformed[field] = graph_dict[field].copy() if isinstance(
                graph_dict[field], np.ndarray
            ) else graph_dict[field]

    # Add augmentation metadata
    transformed['augmentation'] = transform_name
    if 'file_id' in transformed:
        transformed['original_file_id'] = transformed['file_id']

    return transformed


def augment_file(input_path, output_dir, augmentation_strategy='flip_only', file_id_offset=0):
    """
    Augment a single NPY file.

    Args:
        input_path: Path to input .npy file
        output_dir: Directory to save augmented files
        augmentation_strategy: 'flip_only' or 'full'
        file_id_offset: Offset for new file IDs

    Returns:
        List of created file paths
    """
    # Load original
    graph_dict = np.load(input_path, allow_pickle=True).item()

    if not isinstance(graph_dict, dict):
        print(f"Warning: {input_path} is not a dictionary, skipping")
        return []

    # Get original filename stem
    original_stem = Path(input_path).stem

    # Define transformations based on strategy
    if augmentation_strategy == 'flip_only':
        transformations = [
            (None, 'original'),  # Keep original
            (flip_horizontal, 'hflip'),
            (flip_vertical, 'vflip'),
            (lambda c: flip_vertical(flip_horizontal(c)), 'hvflip'),
        ]
    else:  # 'full'
        transformations = [
            (None, 'original'),  # Keep original
            (flip_horizontal, 'hflip'),
            (flip_vertical, 'vflip'),
            (lambda c: flip_vertical(flip_horizontal(c)), 'hvflip'),
            (rotate_90, 'rot90'),
            (rotate_180, 'rot180'),
            (rotate_270, 'rot270'),
            (lambda c: flip_horizontal(rotate_90(c)), 'rot90_hflip'),
        ]

    created_files = []

    for idx, (transform_func, transform_name) in enumerate(transformations):
        # Apply transformation
        if transform_func is None:
            # Original - just copy
            augmented = graph_dict.copy()
            augmented['augmentation'] = 'original'
        else:
            augmented = apply_transformation(graph_dict, transform_func, transform_name)

        # Update file_id
        if 'file_id' in augmented:
            augmented['file_id'] = file_id_offset + idx

        # Generate output filename
        output_filename = f"{original_stem}_{transform_name}.npy"
        output_path = output_dir / output_filename

        # Save
        np.save(output_path, augmented)
        created_files.append(output_path)

    return created_files


def augment_directory(input_dir, output_dir, augmentation_strategy='flip_only',
                     preserve_structure=True):
    """
    Augment all NPY files in a directory.

    Args:
        input_dir: Input directory containing .npy files
        output_dir: Output directory for augmented files
        augmentation_strategy: 'flip_only' (4x) or 'full' (8x)
        preserve_structure: If True, maintains train/val/test subdirectories
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .npy files
    if preserve_structure:
        # Check for train/val/test structure
        subdirs = ['train', 'val', 'test']
        has_structure = all((input_dir / subdir).exists() for subdir in subdirs)

        if has_structure:
            print("Detected train/val/test structure, preserving...")
            npy_files = []
            for subdir in subdirs:
                (output_dir / subdir).mkdir(exist_ok=True)
                subdir_files = list((input_dir / subdir).glob('*.npy'))
                npy_files.extend([(f, subdir) for f in subdir_files])
        else:
            npy_files = [(f, None) for f in input_dir.glob('*.npy')]
    else:
        npy_files = [(f, None) for f in input_dir.glob('*.npy')]

    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} .npy files")
    print(f"Augmentation strategy: {augmentation_strategy}")

    multiplier = 4 if augmentation_strategy == 'flip_only' else 8
    print(f"Will create {len(npy_files) * multiplier} total files ({multiplier}x augmentation)")

    # Track statistics
    stats = {
        'processed': 0,
        'failed': 0,
        'created': 0
    }

    # Process each file
    for file_path, subdir in tqdm(npy_files, desc="Augmenting files"):
        try:
            # Determine output directory
            if subdir:
                file_output_dir = output_dir / subdir
            else:
                file_output_dir = output_dir

            # Calculate file ID offset (to keep IDs unique)
            file_id_offset = stats['created']

            # Augment file
            created = augment_file(
                file_path,
                file_output_dir,
                augmentation_strategy,
                file_id_offset
            )

            stats['processed'] += 1
            stats['created'] += len(created)

        except Exception as e:
            stats['failed'] += 1
            print(f"\nError processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Augmentation complete!")
    print(f"{'='*60}")
    print(f"Original files processed: {stats['processed']}")
    print(f"Total files created: {stats['created']}")
    print(f"Failed: {stats['failed']}")
    print(f"Augmentation factor: {stats['created'] / max(stats['processed'], 1):.1f}x")
    print(f"\nOutput directory: {output_dir}")


def create_augmentation_manifest(output_dir):
    """Create a manifest file documenting the augmentations."""
    output_dir = Path(output_dir)
    manifest = []

    for npy_file in output_dir.rglob('*.npy'):
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            if isinstance(data, dict):
                manifest.append({
                    'filename': str(npy_file.relative_to(output_dir)),
                    'augmentation': data.get('augmentation', 'unknown'),
                    'original_file_id': data.get('original_file_id', data.get('file_id', 'unknown')),
                    'file_id': data.get('file_id', 'unknown')
                })
        except:
            pass

    # Save manifest
    manifest_path = output_dir / 'augmentation_manifest.txt'
    with open(manifest_path, 'w') as f:
        f.write("Floor Plan Augmentation Manifest\n")
        f.write("="*60 + "\n\n")
        for entry in manifest:
            f.write(f"File: {entry['filename']}\n")
            f.write(f"  Augmentation: {entry['augmentation']}\n")
            f.write(f"  Original File ID: {entry['original_file_id']}\n")
            f.write(f"  New File ID: {entry['file_id']}\n\n")

    print(f"Created manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Augment GSDiff NPY floor plan files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Augmentation Strategies:

1. flip_only (4x dataset):
   - Original
   - Horizontal flip
   - Vertical flip
   - Horizontal + Vertical flip

2. full (8x dataset):
   - All flips (4)
   - 90° rotation
   - 180° rotation
   - 270° rotation
   - 90° rotation + horizontal flip

Examples:
    # Basic augmentation (4x)
    python augment_floor_plans.py --input_dir data/npy --output_dir data/npy_augmented

    # Full augmentation (8x)
    python augment_floor_plans.py --input_dir data/npy --output_dir data/npy_augmented --full

    # Single file
    python augment_floor_plans.py --input data/0.npy --output_dir data/augmented

    # With train/val/test structure preservation
    python augment_floor_plans.py --input_dir data/npy --output_dir data/augmented --preserve_structure
        """
    )

    # Input/output
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='Input .npy file')
    group.add_argument('--input_dir', type=str, help='Input directory containing .npy files')

    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for augmented files')

    # Augmentation options
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument('--flip_only', action='store_true',
                               help='Only use flips (4x dataset, default)')
    strategy_group.add_argument('--full', action='store_true',
                               help='Use flips and rotations (8x dataset)')

    parser.add_argument('--preserve_structure', action='store_true',
                       help='Preserve train/val/test directory structure')
    parser.add_argument('--create_manifest', action='store_true',
                       help='Create augmentation manifest file')

    args = parser.parse_args()

    # Determine strategy
    if args.full:
        strategy = 'full'
    else:
        strategy = 'flip_only'  # Default

    output_dir = Path(args.output_dir)

    # Process
    if args.input:
        # Single file
        print(f"Augmenting single file: {args.input}")
        output_dir.mkdir(parents=True, exist_ok=True)
        created = augment_file(args.input, output_dir, strategy)
        print(f"\n✓ Created {len(created)} augmented files")
        for path in created:
            print(f"  - {path.name}")
    else:
        # Directory
        augment_directory(
            args.input_dir,
            args.output_dir,
            strategy,
            args.preserve_structure
        )

    # Create manifest if requested
    if args.create_manifest:
        create_augmentation_manifest(output_dir)


if __name__ == '__main__':
    main()
