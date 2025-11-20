#!/usr/bin/env python3
"""
Dataset Splitting Script for GSDiff

Splits NPY graph files into train/validation/test sets with configurable ratios.
Useful for preparing custom floor plan data for GSDiff training.

Usage:
    python split_dataset.py --input_dir data/ --train_ratio 0.9 --val_ratio 0.05 --test_ratio 0.05

Output:
    Creates three subdirectories: train/, val/, test/ with split files
"""

import argparse
import shutil
from pathlib import Path
import random
from tqdm import tqdm


def split_dataset(input_dir, output_dir=None, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, seed=42):
    """
    Split NPY files into train/val/test sets.

    Args:
        input_dir: Directory containing NPY files to split
        output_dir: Base output directory (default: input_dir parent + '_split')
        train_ratio: Fraction for training set (default: 0.9)
        val_ratio: Fraction for validation set (default: 0.05)
        test_ratio: Fraction for test set (default: 0.05)
        seed: Random seed for reproducibility (default: 42)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative, and train_ratio must be > 0")

    # Setup paths
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if output_dir is None:
        output_dir = input_path.parent / (input_path.name + '_split')
    else:
        output_dir = Path(output_dir)

    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get all NPY files
    npy_files = list(input_path.glob('*.npy'))

    if len(npy_files) == 0:
        raise ValueError(f"No NPY files found in {input_dir}")

    print(f"Found {len(npy_files)} NPY files")
    print(f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

    # Shuffle files with seed for reproducibility
    random.seed(seed)
    random.shuffle(npy_files)

    # Calculate split indices
    n_total = len(npy_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Ensure all files are used

    print(f"Split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Split files
    train_files = npy_files[:n_train]
    val_files = npy_files[n_train:n_train + n_val]
    test_files = npy_files[n_train + n_val:]

    # Copy files to respective directories
    print("\nCopying files to train/...")
    for f in tqdm(train_files, desc="Train"):
        shutil.copy2(f, train_dir / f.name)

    print("Copying files to val/...")
    for f in tqdm(val_files, desc="Val"):
        shutil.copy2(f, val_dir / f.name)

    print("Copying files to test/...")
    for f in tqdm(test_files, desc="Test"):
        shutil.copy2(f, test_dir / f.name)

    # Print summary
    print("\n" + "="*60)
    print("Dataset splitting complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"  train/: {len(train_files)} files")
    print(f"  val/:   {len(val_files)} files")
    print(f"  test/:  {len(test_files)} files")
    print(f"  Total:  {len(npy_files)} files")
    print("="*60)

    return train_dir, val_dir, test_dir


def main():
    parser = argparse.ArgumentParser(
        description='Split NPY dataset into train/val/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 90/5/5 split
  python split_dataset.py --input_dir data/graph_npy

  # Custom split ratios (80/10/10)
  python split_dataset.py --input_dir data/graph_npy \\
      --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

  # Specify output directory
  python split_dataset.py --input_dir data/graph_npy \\
      --output_dir data/my_split

  # Use different random seed
  python split_dataset.py --input_dir data/graph_npy --seed 123
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing NPY files to split')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: input_dir + "_split")')

    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Training set ratio (default: 0.9)')

    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation set ratio (default: 0.05)')

    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='Test set ratio (default: 0.05)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    try:
        split_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
