# GSDiff Data Augmentation Guide

> **📌 RECOMMENDED WORKFLOW:** If you're preparing custom data, see **[PRE_AUGMENTATION_WORKFLOW.md](PRE_AUGMENTATION_WORKFLOW.md)** for a streamlined guide on augmenting BEFORE placing data in the repo. This is the simpler approach!
>
> This guide covers general augmentation usage and technical details.

## Overview

The `augment_floor_plans.py` script provides geometric data augmentation for GSDiff NPY floor plan files. It increases your dataset size by 4× or 8× while preserving the graph structure, adjacency relationships, and semantic information.

## Why Use Data Augmentation?

**Benefits:**

1. **Increase dataset size** without collecting new floor plans
2. **Improve model generalization** by exposing it to different orientations
3. **Reduce overfitting** especially when training data is limited
4. **Maintain data quality** - transformations preserve geometric validity

**When to use:**

- You have a small dataset (< 1000 floor plans)
- You want to increase model robustness to different orientations
- You're retraining with limited custom data
- You want to boost performance without manual data collection

---

## Installation

No additional dependencies needed beyond standard GSDiff requirements:

```bash
# Already included in GSDiff requirements
pip install numpy tqdm
```

Make the script executable:

```bash
chmod +x augment_floor_plans.py
```

---

## Quick Start

### Basic Usage (4× Dataset)

Augment all files in a directory with flips only:

```bash
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics/train \
    --output_dir datasets/rplang-v3-withsemantics-augmented/train \
    --flip_only \
    --preserve_structure
```

### Full Augmentation (8× Dataset)

Add rotations for maximum augmentation:

```bash
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics/train \
    --output_dir datasets/rplang-v3-withsemantics-augmented/train \
    --full \
    --preserve_structure
```

---

## Augmentation Strategies

### Strategy 1: Flip Only (4× Dataset)

**Transformations applied:**

1. **Original** - Unchanged copy
2. **Horizontal flip** - Mirror across Y-axis
3. **Vertical flip** - Mirror across X-axis
4. **Both flips** - Horizontal + Vertical (equivalent to 180° rotation)

**Use case:** Conservative augmentation, maintains architectural conventions

**Example:**

```bash
python augment_floor_plans.py \
    --input_dir data/train \
    --output_dir data/train_augmented \
    --flip_only
```

**Output files (from `0.npy`):**

- `0_original.npy` - Original floor plan
- `0_hflip.npy` - Horizontally flipped
- `0_vflip.npy` - Vertically flipped
- `0_hvflip.npy` - Both directions flipped

### Strategy 2: Full Augmentation (8× Dataset)

**Transformations applied:**

1. **Original**
2. **Horizontal flip**
3. **Vertical flip**
4. **Both flips**
5. **90° rotation** (counterclockwise)
6. **180° rotation**
7. **270° rotation** (or 90° clockwise)
8. **90° rotation + horizontal flip**

**Use case:** Maximum augmentation, best for small datasets

**Example:**

```bash
python augment_floor_plans.py \
    --input_dir data/train \
    --output_dir data/train_augmented \
    --full
```

**Output files (from `0.npy`):**

- `0_original.npy`
- `0_hflip.npy`
- `0_vflip.npy`
- `0_hvflip.npy`
- `0_rot90.npy`
- `0_rot180.npy`
- `0_rot270.npy`
- `0_rot90_hflip.npy`

---

## Command-Line Options

### Required Arguments

```bash
# Input (choose one)
--input FILE              # Single .npy file
--input_dir DIRECTORY     # Directory of .npy files

# Output
--output_dir DIRECTORY    # Where to save augmented files
```

### Augmentation Strategy (choose one)

```bash
--flip_only              # Flips only (4× dataset, default)
--full                   # Flips + rotations (8× dataset)
```

### Optional Flags

```bash
--preserve_structure     # Maintain train/val/test subdirectories
--create_manifest        # Generate augmentation_manifest.txt
```

---

## Usage Examples

### Example 1: Augment Training Set Only

```bash
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics/train \
    --output_dir datasets/rplang-v3-withsemantics-augmented/train \
    --flip_only
```

**Result:** Training set grows from 60,000 → 240,000 files (4×)

### Example 2: Augment All Splits (train/val/test)

```bash
# Create output directories
mkdir -p datasets/rplang-v3-withsemantics-augmented/{train,val,test}

# Augment training data with full strategy
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics/train \
    --output_dir datasets/rplang-v3-withsemantics-augmented/train \
    --full

# Augment validation data (flip only to keep it smaller)
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics/val \
    --output_dir datasets/rplang-v3-withsemantics-augmented/val \
    --flip_only

# Don't augment test set (keep original for fair evaluation)
cp -r datasets/rplang-v3-withsemantics/test/* \
      datasets/rplang-v3-withsemantics-augmented/test/
```

### Example 3: Augment With Preserved Structure

```bash
# If your input has train/val/test subdirectories
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics \
    --output_dir datasets/rplang-v3-withsemantics-augmented \
    --full \
    --preserve_structure
```

**Note:** Script automatically detects and preserves train/val/test structure.

### Example 4: Single File Augmentation

```bash
python augment_floor_plans.py \
    --input datasets/rplang-v3-withsemantics/train/0.npy \
    --output_dir augmented_samples \
    --full
```

**Output:**

```
augmented_samples/
├── 0_original.npy
├── 0_hflip.npy
├── 0_vflip.npy
├── 0_hvflip.npy
├── 0_rot90.npy
├── 0_rot180.npy
├── 0_rot270.npy
└── 0_rot90_hflip.npy
```

### Example 5: Create Manifest

```bash
python augment_floor_plans.py \
    --input_dir data/train \
    --output_dir data/train_augmented \
    --full \
    --create_manifest
```

**Creates:** `data/train_augmented/augmentation_manifest.txt`

**Manifest content:**

```
Floor Plan Augmentation Manifest
============================================================

File: 0_original.npy
  Augmentation: original
  Original File ID: 0
  New File ID: 0

File: 0_hflip.npy
  Augmentation: hflip
  Original File ID: 0
  New File ID: 1

...
```

---

## Technical Details

### What Gets Transformed

The script transforms **only coordinate data**, preserving all other information:

**Transformed:**

- ✅ `corner_list_np_normalized` - Corner coordinates
- ✅ `corner_list_np_normalized_padding` - Padded corner coordinates
- ✅ `corner_list_np_normalized_padding_withsemantics` - Coordinates (columns 0-1 only)
- ✅ `edge_coords` - Edge endpoint coordinates

**Preserved (unchanged):**

- ✅ `adjacency_matrix` - Corner connectivity (structure preserved)
- ✅ `adjacency_matrix_np_padding` - Padded adjacency
- ✅ `edges` - Flattened edge matrix
- ✅ `global_matrix_np_padding` - Attention matrix
- ✅ `padding_mask` - Padding indicators
- ✅ `semantics` - Room type labels (semantic meaning unchanged)
- ✅ All semantic columns in `corner_list_np_normalized_padding_withsemantics`

### Transformation Mathematics

All transformations work in **normalized [-1, 1] coordinate space**:

#### Horizontal Flip

```python
x' = -x
y' = y
```

Mirrors across Y-axis (left ↔ right)

#### Vertical Flip

```python
x' = x
y' = -y
```

Mirrors across X-axis (top ↔ bottom)

#### 90° Rotation (Counterclockwise)

```python
x' = -y
y' = x
```

Rotation matrix: `[0, -1; 1, 0]`

#### 180° Rotation

```python
x' = -x
y' = -y
```

Equivalent to horizontal + vertical flip

#### 270° Rotation (90° Clockwise)

```python
x' = y
y' = -x
```

Rotation matrix: `[0, 1; -1, 0]`

### Why Adjacency Is Preserved

The adjacency matrix represents **which corners are connected**, not their spatial positions. Since transformations:

- Don't add or remove corners
- Don't change which corners connect to which
- Only change spatial positions

The adjacency relationships remain valid after transformation.

**Example:**

```
Original floor plan:
  Corner 0 connected to Corner 1
  Corner 1 connected to Corner 2

After any transformation:
  Corner 0 still connected to Corner 1
  Corner 1 still connected to Corner 2
```

The wall is simply rotated or flipped, but the connectivity is identical.

---

## Integration with Training

### Option 1: Pre-Augment Before Training

**Workflow:**

```bash
# 1. Augment your dataset
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics/train \
    --output_dir datasets/rplang-v3-withsemantics-augmented/train \
    --full

# 2. Update dataset path in training script
# In trainval_main_unconstrained.py:
dataset_train = RPlanGEdgeSemanSimplified(
    'train',
    data_root='datasets/rplang-v3-withsemantics-augmented'
)

# 3. Train as normal
python scripts/trainval_main_unconstrained.py
```

**Pros:**

- ✅ Augmentation done once, not during every epoch
- ✅ Faster training (no on-the-fly augmentation overhead)
- ✅ Easy to verify augmented data

**Cons:**

- ❌ Uses more disk space (4× or 8× original size)
- ❌ Less flexible (can't change augmentation during training)

### Option 2: Keep Original Dataset, Use Augmented for Specific Experiments

```bash
# Augment to separate directory
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics \
    --output_dir datasets/rplang-v3-withsemantics-4x \
    --flip_only \
    --preserve_structure

# Keep both versions:
# - Original: datasets/rplang-v3-withsemantics (for baseline)
# - Augmented: datasets/rplang-v3-withsemantics-4x (for comparison)
```

### Disk Space Requirements

**Example calculation for RPLAN dataset:**

Original dataset:

- Train: 60,000 files × 150KB = 9GB
- Val: 10,000 files × 150KB = 1.5GB
- Test: 10,000 files × 150KB = 1.5GB
- **Total: 12GB**

After 4× augmentation (flip_only):

- Train: 60,000 × 4 = 240,000 files × 150KB = 36GB
- Val: 10,000 × 4 = 40,000 files × 150KB = 6GB
- Test: 10,000 files (unchanged) × 150KB = 1.5GB
- **Total: 43.5GB** (+31.5GB)

After 8× augmentation (full):

- Train: 60,000 × 8 = 480,000 files × 150KB = 72GB
- Val: 10,000 × 8 = 80,000 files × 150KB = 12GB
- Test: 10,000 files (unchanged) × 150KB = 1.5GB
- **Total: 85.5GB** (+73.5GB)

**Recommendation:** Augment only training set, keep val/test original.

---

## Best Practices

### 1. Don't Augment Test Set

**Why:** Test set should represent real-world distribution for fair evaluation.

```bash
# ✅ Good: Only augment training data
python augment_floor_plans.py \
    --input_dir datasets/rplang-v3-withsemantics/train \
    --output_dir datasets/rplang-v3-withsemantics-augmented/train \
    --full

# ❌ Bad: Augmenting test set inflates metrics artificially
```

### 2. Use Conservative Augmentation for Validation

```bash
# Training: Full augmentation (8×)
python augment_floor_plans.py \
    --input_dir datasets/train \
    --output_dir datasets/train_aug \
    --full

# Validation: Flip only (4×) or no augmentation
python augment_floor_plans.py \
    --input_dir datasets/val \
    --output_dir datasets/val_aug \
    --flip_only
```

### 3. Verify Augmented Data

Always inspect a few augmented files to ensure correctness:

```bash
# Generate single file with all augmentations
python augment_floor_plans.py \
    --input datasets/train/0.npy \
    --output_dir test_augmentation \
    --full

# Visualize to verify
python test_gt_rendering.py  # Use your visualization script
```

### 4. Use Flip-Only for Architectural Datasets

**Reason:** Buildings are typically aligned to cardinal directions. Rotations might create unrealistic orientations.

```bash
# For real architectural data, prefer flips
python augment_floor_plans.py \
    --input_dir datasets/train \
    --output_dir datasets/train_aug \
    --flip_only  # More realistic than arbitrary rotations
```

### 5. Track Augmentation in Experiments

Use `--create_manifest` to document which files came from which augmentation:

```bash
python augment_floor_plans.py \
    --input_dir datasets/train \
    --output_dir datasets/train_aug \
    --full \
    --create_manifest

# Check manifest
cat datasets/train_aug/augmentation_manifest.txt
```

---

## Troubleshooting

### Issue: "No .npy files found"

**Solution:** Check that input directory contains .npy files:

```bash
ls -lh datasets/rplang-v3-withsemantics/train/*.npy
```

### Issue: "Warning: X.npy is not a dictionary, skipping"

**Cause:** NPY file doesn't contain a dictionary (might be raw array)

**Solution:** Ensure your NPY files are in GSDiff format (dictionary with keys like `corner_list_np_normalized_padding_withsemantics`)

### Issue: Out of Disk Space

**Solution:** Augment in batches or use flip_only instead of full:

```bash
# Smaller augmentation
python augment_floor_plans.py \
    --input_dir datasets/train \
    --output_dir datasets/train_aug \
    --flip_only  # 4× instead of 8×
```

### Issue: File IDs Clash

**Solution:** The script automatically assigns unique file_id values sequentially. If you're merging with existing data, offset manually:

```python
# In augment_floor_plans.py, modify file_id_offset calculation
file_id_offset = stats['created'] + 100000  # Add offset
```

---

## Performance

**Processing speed:**

- ~50-100 files/second on standard CPU
- Processing 60,000 files (full augmentation) takes ~10-15 minutes

**Memory usage:**

- Processes one file at a time
- Peak memory: ~500MB

**Example benchmark:**

```bash
time python augment_floor_plans.py \
    --input_dir datasets/train \
    --output_dir datasets/train_aug \
    --full

# Output:
# Found 60000 .npy files
# Will create 480000 total files (8x augmentation)
# Augmenting files: 100%|███████| 60000/60000 [12:34<00:00, 79.5it/s]
#
# Augmentation complete!
# Original files processed: 60000
# Total files created: 480000
# Failed: 0
# Augmentation factor: 8.0x
```

---

## Advanced Usage

### Custom Augmentation Strategy

Edit the script to add custom transformations:

```python
# In augment_floor_plans.py

# Add new transformation function
def rotate_45(coords):
    """Rotate 45 degrees."""
    angle = np.pi / 4
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = coords.copy()
    rotated[:, 0] = coords[:, 0] * cos_a - coords[:, 1] * sin_a
    rotated[:, 1] = coords[:, 0] * sin_a + coords[:, 1] * cos_a
    return rotated

# Add to transformations list (in augment_file function)
transformations = [
    # ... existing transformations
    (rotate_45, 'rot45'),
]
```

### Selective Augmentation

Augment only certain files:

```python
import numpy as np
from pathlib import Path

# Example: Only augment files with < 30 corners
input_dir = Path('datasets/train')
output_dir = Path('datasets/train_selective_aug')
output_dir.mkdir(exist_ok=True)

for npy_file in input_dir.glob('*.npy'):
    data = np.load(npy_file, allow_pickle=True).item()
    num_corners = data['padding_mask'].sum()

    if num_corners < 30:
        # Augment this file
        os.system(f"python augment_floor_plans.py --input {npy_file} --output_dir {output_dir} --full")
    else:
        # Just copy original
        shutil.copy(npy_file, output_dir)
```

---

## Summary

### Quick Reference

| Task | Command |
|------|---------|
| **Basic 4× augmentation** | `python augment_floor_plans.py --input_dir data/train --output_dir data/train_aug --flip_only` |
| **Full 8× augmentation** | `python augment_floor_plans.py --input_dir data/train --output_dir data/train_aug --full` |
| **Single file test** | `python augment_floor_plans.py --input data/0.npy --output_dir test_aug --full` |
| **With manifest** | Add `--create_manifest` flag |
| **Preserve structure** | Add `--preserve_structure` flag |

### Dataset Size Impact

| Original Size | Flip Only (4×) | Full (8×) |
|--------------|----------------|-----------|
| 10,000 files | 40,000 files | 80,000 files |
| 60,000 files | 240,000 files | 480,000 files |
| 100,000 files | 400,000 files | 800,000 files |

### Recommendations

✅ **Use flip_only for:**

- Architectural floor plans (real buildings)
- Large datasets (> 50,000 samples)
- Limited disk space

✅ **Use full for:**

- Small datasets (< 10,000 samples)
- Abstract/synthetic floor plans
- Maximum model robustness

✅ **Always:**

- Keep test set un-augmented
- Verify augmented samples visually
- Use `--create_manifest` for tracking

---

**Script Location:** `/home/user/GSDiff/augment_floor_plans.py`

**Created:** 2025-11-13

**Compatible with:** GSDiff NPY format (dictionary with normalized coordinates)
