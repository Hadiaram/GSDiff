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
    test_files = sorted(test_data_path.glob('*.npy'), key=lambda x: int(x.stem))

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
    test_files = sorted(test_data_path.glob('*.npy'), key=lambda x: int(x.stem))

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
    # Create feature maps for your test data
    python create_cnn_featuremaps.py --test_dir datasets/rplang-v3-withsemantics/test

    # Also create withboundary files
    python create_cnn_featuremaps.py --test_dir datasets/rplang-v3-withsemantics/test --create_withboundary
        """
    )

    parser.add_argument('--test_dir', type=str,
                        default='datasets/rplang-v3-withsemantics/test',
                        help='Directory containing test NPY files')
    parser.add_argument('--feature_dir', type=str,
                        default='datasets/prerunning_cnn_featuremaps',
                        help='Where to save feature maps')
    parser.add_argument('--withboundary_dir', type=str,
                        default='datasets/rplang-v3-withsemantics-withboundary/test',
                        help='Where to save withboundary files')
    parser.add_argument('--create_withboundary', action='store_true',
                        help='Also create withboundary files')
    parser.add_argument('--max_corners', type=int, default=150,
                        help='Maximum number of corners (default: 150)')

    args = parser.parse_args()

    # Create feature maps
    create_featuremaps_for_directory(args.test_dir, args.feature_dir, args.max_corners)

    # Create withboundary files if requested
    if args.create_withboundary:
        create_withboundary_files(args.test_dir, args.withboundary_dir, args.max_corners)

    print("\n✓ Done! You can now run test_boun.py")
