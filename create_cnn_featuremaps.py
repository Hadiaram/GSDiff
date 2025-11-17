"""
Create CNN feature maps for GSDiff test data.

This creates the required prerunning_cnn_featuremaps files that match
your actual test data files.

The feature maps are dummy/random data, but they have the correct format
that GSDiff expects: {16: [numpy_array_of_shape_(1024, 16, 16)]}
"""
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_featuremaps_for_directory(test_data_dir, output_dir, max_corners=150):
    """
    Create CNN feature maps matching the test data files.

    Args:
        test_data_dir: Directory containing your test NPY files
        output_dir: Where to save the feature maps (usually datasets/prerunning_cnn_featuremaps)
        max_corners: Maximum number of corners (default: 150)
    """
    test_data_path = Path(test_data_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all NPY files in test directory
    test_files = sorted(test_data_path.glob('*.npy'))

    if len(test_files) == 0:
        print(f"No NPY files found in {test_data_dir}")
        return

    print(f"Found {len(test_files)} test files")
    print(f"Creating matching feature maps...")

    for test_file in tqdm(test_files):
        # Create feature map with correct format
        # Format: {16: [array of shape (1024, 16, 16)]}
        feat = {
            16: [np.random.randn(1024, 16, 16).astype(np.float32)]
        }

        # Save with same filename as test file
        output_file = output_path / test_file.name
        np.save(output_file, feat)

    print(f"\n✓ Created {len(test_files)} feature maps in {output_dir}")
    print("\n⚠️  Note: These are random dummy feature maps for testing.")
    print("   They allow the code to run but won't produce meaningful results.")
    print("   For real experiments, you need actual CNN-extracted features.")


def create_withboundary_files(test_data_dir, output_dir, max_corners=150):
    """
    Create withboundary files if they don't exist.

    These are simplified versions needed by RPlanGEdgeSemanSimplified_81.

    Args:
        test_data_dir: Directory containing your test NPY files
        output_dir: Where to save withboundary files
        max_corners: Maximum number of corners (default: 150)
    """
    test_data_path = Path(test_data_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all NPY files in test directory
    test_files = sorted(test_data_path.glob('*.npy'))

    if len(test_files) == 0:
        print(f"No NPY files found in {test_data_dir}")
        return

    print(f"\nCreating withboundary files...")

    for test_file in tqdm(test_files):
        # Load original file
        try:
            data = np.load(test_file, allow_pickle=True)

            # Handle different formats
            if data.ndim == 0:
                graph = data.item()
            elif data.size == 1:
                graph = data.flatten()[0]
            else:
                print(f"Skipping {test_file.name}: wrong format")
                continue

            if not isinstance(graph, dict):
                print(f"Skipping {test_file.name}: not a dictionary")
                continue

            # Create simplified withboundary version
            # Just needs the corner data and padding mask
            graph_withboundary = {
                'corner_list_np_normalized_padding_withsemantics': graph.get(
                    'corner_list_np_normalized_padding_withsemantics',
                    np.random.randn(max_corners, 16).astype(np.float32)
                ),
                'padding_mask': graph.get(
                    'padding_mask',
                    np.ones((max_corners, 1), dtype=np.uint8)
                )
            }

            # Save
            output_file = output_path / test_file.name
            np.save(output_file, graph_withboundary)

        except Exception as e:
            print(f"Error processing {test_file.name}: {e}")

    print(f"\n✓ Created {len(test_files)} withboundary files in {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create CNN feature maps for GSDiff',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single split (test)
    python create_cnn_featuremaps.py \\
        --test_dir datasets/test \\
        --feature_dir datasets/prerunning_cnn_featuremaps/test \\
        --withboundary_dir datasets/withboundary/test

    # Process all splits automatically
    python create_cnn_featuremaps.py \\
        --split_root augmented_npy_split \\
        --create_withboundary

    # Auto-detect split type from directory name
    python create_cnn_featuremaps.py \\
        --test_dir augmented_npy_split/test \\
        --auto_structure \\
        --create_withboundary
        """
    )

    parser.add_argument('--test_dir', type=str,
                        default=None,
                        help='Directory containing test NPY files')
    parser.add_argument('--split_root', type=str,
                        default=None,
                        help='Root directory containing train/val/test subdirs (processes all)')
    parser.add_argument('--feature_dir', type=str,
                        default='datasets/prerunning_cnn_featuremaps',
                        help='Where to save feature maps')
    parser.add_argument('--withboundary_dir', type=str,
                        default='datasets/rplang-v3-withsemantics-withboundary/test',
                        help='Where to save withboundary files')
    parser.add_argument('--auto_structure', action='store_true',
                        help='Auto-detect train/val/test from path and create matching structure')
    parser.add_argument('--create_withboundary', action='store_true',
                        help='Also create withboundary files')
    parser.add_argument('--max_corners', type=int, default=150,
                        help='Maximum number of corners (default: 150)')

    args = parser.parse_args()

    # Handle batch processing of all splits
    if args.split_root:
        split_root = Path(args.split_root)
        splits = ['train', 'val', 'test']

        print(f"Processing all splits in {args.split_root}...")
        for split in splits:
            split_dir = split_root / split
            if not split_dir.exists():
                print(f"⚠️  Skipping {split}: directory not found")
                continue

            print(f"\n{'='*60}")
            print(f"Processing {split} split...")
            print(f"{'='*60}")

            # Create output directories with split names
            feature_out = Path(args.feature_dir) / split
            withboundary_out = Path(args.withboundary_dir.replace('/test', '')) / split

            create_featuremaps_for_directory(str(split_dir), str(feature_out), args.max_corners)

            if args.create_withboundary:
                create_withboundary_files(str(split_dir), str(withboundary_out), args.max_corners)

        print(f"\n{'='*60}")
        print("✓ Done processing all splits!")
        print(f"{'='*60}")

    # Handle single directory with auto-structure detection
    elif args.test_dir:
        test_path = Path(args.test_dir)

        # Auto-detect split type from directory name
        if args.auto_structure:
            split_name = None
            for part in test_path.parts[::-1]:  # Check from end to start
                if part in ['train', 'val', 'test']:
                    split_name = part
                    break

            if split_name:
                print(f"Auto-detected split: {split_name}")
                feature_out = Path(args.feature_dir) / split_name
                withboundary_out = Path(args.withboundary_dir.replace('/test', '')) / split_name
            else:
                print("⚠️  Could not auto-detect split, using default paths")
                feature_out = Path(args.feature_dir)
                withboundary_out = Path(args.withboundary_dir)
        else:
            feature_out = Path(args.feature_dir)
            withboundary_out = Path(args.withboundary_dir)

        # Create feature maps
        create_featuremaps_for_directory(args.test_dir, str(feature_out), args.max_corners)

        # Create withboundary files if requested
        if args.create_withboundary:
            create_withboundary_files(args.test_dir, str(withboundary_out), args.max_corners)

        print("\n✓ Done!")

    else:
        parser.error("Must specify either --test_dir or --split_root")

