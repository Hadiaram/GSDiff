# Pre-Augmentation Workflow Guide

## Overview

This guide covers augmenting your dataset **before** placing it in the GSDiff repository as training input. This is the recommended approach when preparing custom data.

---

## Workflow: Augment First, Then Train

```
Your Custom Data (JSON/Images)
    ↓
Convert to NPY format (with correct image_size)
    ↓
Augment NPY files (4× or 8×)
    ↓
Place augmented files in datasets/ directory
    ↓
Train directly (no configuration changes needed)
```

---

## Step-by-Step Process

### Step 1: Convert Your Data to NPY Format

First, convert your custom floor plans to GSDiff NPY format:

```bash
# Using the raster-to-graph converter
python raster_to_graph_converter.py \
    --input_dir /path/to/your/json_or_raster_files \
    --output_dir converted_npy \
    --image_size 2575 \  # Use your actual raster size!
    --validate

# Verify conversion succeeded
ls -lh converted_npy/
```

**Result:** You now have NPY files in correct GSDiff format (e.g., `0.npy`, `1.npy`, `2.npy`, ...)

### Step 2: Augment the Converted Files

Now augment to increase dataset size:

```bash
# Option A: Conservative augmentation (4× increase)
python augment_floor_plans.py \
    --input_dir converted_npy \
    --output_dir augmented_npy \
    --flip_only

# Option B: Maximum augmentation (8× increase)
python augment_floor_plans.py \
    --input_dir converted_npy \
    --output_dir augmented_npy \
    --full
```

**Result:** Augmented directory now contains:

```
augmented_npy/
├── 0_original.npy
├── 0_hflip.npy
├── 0_vflip.npy
├── 0_hvflip.npy
├── 0_rot90.npy        # (if using --full)
├── 0_rot180.npy       # (if using --full)
├── 0_rot270.npy       # (if using --full)
├── 0_rot90_hflip.npy  # (if using --full)
├── 1_original.npy
├── 1_hflip.npy
└── ...
```

### Step 3: Split Into Train/Val/Test Sets

Organize your augmented files into train/val/test splits:

```bash
# Create directory structure
mkdir -p datasets/my-custom-floorplans/{train,val,test}

# Example split (adjust percentages as needed):
# - Training: 70% of augmented files
# - Validation: 15% of augmented files
# - Test: 15% of ORIGINAL (non-augmented) files

# Option A: Manual split
# Move augmented files to train
mv augmented_npy/*_*.npy datasets/my-custom-floorplans/train/

# Move some originals to val/test
mv converted_npy/0.npy converted_npy/1.npy ... datasets/my-custom-floorplans/val/
mv converted_npy/10.npy converted_npy/11.npy ... datasets/my-custom-floorplans/test/
```

**Automated split script:**

```python
# split_augmented_dataset.py
import shutil
from pathlib import Path
import random

# Configuration
augmented_dir = Path('augmented_npy')
output_dir = Path('datasets/my-custom-floorplans')
train_ratio = 0.70
val_ratio = 0.15
# test_ratio = 0.15 (remaining)

# Create directories
(output_dir / 'train').mkdir(parents=True, exist_ok=True)
(output_dir / 'val').mkdir(parents=True, exist_ok=True)
(output_dir / 'test').mkdir(parents=True, exist_ok=True)

# Get all augmented files
all_files = sorted(augmented_dir.glob('*.npy'))

# Group by original file ID
from collections import defaultdict
file_groups = defaultdict(list)

for f in all_files:
    # Extract base ID (e.g., "0" from "0_hflip.npy")
    base_id = f.stem.split('_')[0]
    file_groups[base_id].append(f)

# Shuffle groups for random split
group_ids = list(file_groups.keys())
random.shuffle(group_ids)

# Calculate split points
n_groups = len(group_ids)
n_train = int(n_groups * train_ratio)
n_val = int(n_groups * val_ratio)

train_groups = group_ids[:n_train]
val_groups = group_ids[n_train:n_train + n_val]
test_groups = group_ids[n_train + n_val:]

# Copy files to appropriate directories
for group_id in train_groups:
    for f in file_groups[group_id]:
        shutil.copy(f, output_dir / 'train' / f.name)

for group_id in val_groups:
    for f in file_groups[group_id]:
        # Only use original (non-augmented) for validation
        if '_original.npy' in f.name or '_' not in f.stem:
            shutil.copy(f, output_dir / 'val' / f.name)

for group_id in test_groups:
    for f in file_groups[group_id]:
        # Only use original (non-augmented) for test
        if '_original.npy' in f.name or '_' not in f.stem:
            shutil.copy(f, output_dir / 'test' / f.name)

print(f"✓ Train: {len(list((output_dir / 'train').glob('*.npy')))} files")
print(f"✓ Val: {len(list((output_dir / 'val').glob('*.npy')))} files")
print(f"✓ Test: {len(list((output_dir / 'test').glob('*.npy')))} files")
```

### Step 4: Generate CNN Feature Maps (If Using Boundary Constraint)

If you plan to use the boundary-constrained variant, generate CNN feature maps:

```bash
python create_cnn_featuremaps.py \
    --test_dir datasets/my-custom-floorplans/train \
    --create_withboundary

# Repeat for val and test
python create_cnn_featuremaps.py \
    --test_dir datasets/my-custom-floorplans/val \
    --create_withboundary

python create_cnn_featuremaps.py \
    --test_dir datasets/my-custom-floorplans/test \
    --create_withboundary
```

### Step 5: Update Dataset Path in Training Script

Modify training script to point to your custom dataset:

```python
# In scripts/trainval_main_unconstrained.py (or whichever variant you're using)

# Find this line:
dataset_train = RPlanGEdgeSemanSimplified('train')

# Change to:
dataset_train = RPlanGEdgeSemanSimplified(
    'train',
    data_root='datasets/my-custom-floorplans'  # Your augmented dataset
)

# Also update validation dataset:
dataset_val = RPlanGEdgeSemanSimplified(
    'val',
    data_root='datasets/my-custom-floorplans'
)
```

### Step 6: Train Normally

```bash
# Train with your augmented custom data
python scripts/trainval_main_unconstrained.py

# The training script will automatically see all augmented files
# No special handling needed - they're just more training samples!
```

---

## File Naming Considerations

### Understanding Augmented File Names

After augmentation, your files will have suffixes:

```
Original files → Augmented files

0.npy → 0_original.npy    # Unchanged copy
        0_hflip.npy       # Horizontally flipped
        0_vflip.npy       # Vertically flipped
        0_hvflip.npy      # Both flips
        0_rot90.npy       # 90° rotation (if --full)
        0_rot180.npy      # 180° rotation (if --full)
        0_rot270.npy      # 270° rotation (if --full)
        0_rot90_hflip.npy # Combo (if --full)
```

### Option 1: Keep Descriptive Names (Recommended for Development)

**Pros:**

- ✅ Easy to identify augmentation type
- ✅ Easy to debug issues
- ✅ Can track augmentation manifest

**Cons:**

- ❌ File names are longer
- ❌ May need to handle underscore in filenames

**GSDiff Compatibility:** ✅ Works fine - dataset loaders don't care about filenames

### Option 2: Rename to Sequential Numbers

If you prefer clean sequential numbering:

```bash
# Rename augmented files to sequential numbers
cd augmented_npy
python << 'EOF'
from pathlib import Path

files = sorted(Path('.').glob('*.npy'))
for idx, f in enumerate(files):
    f.rename(f'{idx}.npy')
print(f"Renamed {len(files)} files to 0.npy through {len(files)-1}.npy")
EOF
```

**Result:**

```
0.npy, 1.npy, 2.npy, ..., 7999.npy
# (if you had 1000 originals × 8 augmentations = 8000 files)
```

**Pros:**

- ✅ Clean sequential numbering
- ✅ Matches original RPLAN format

**Cons:**

- ❌ Loses track of which augmentation was applied
- ❌ Harder to debug issues

---

## Best Practices for Pre-Augmentation

### 1. Always Keep Original Non-Augmented Files

```bash
# Before augmentation, make a backup
cp -r converted_npy converted_npy_originals_backup

# Then augment
python augment_floor_plans.py \
    --input_dir converted_npy \
    --output_dir augmented_npy \
    --full
```

**Why:** You may need originals for:

- Test set (use only non-augmented)
- Debugging issues
- Trying different augmentation strategies

### 2. Use Only Original Files for Test Set

```bash
# Test set should be NON-AUGMENTED for fair evaluation
cp converted_npy_originals_backup/*.npy datasets/my-custom-floorplans/test/
```

### 3. Consider Partial Augmentation for Validation

```bash
# Validation: Use originals OR light augmentation (flip_only)
python augment_floor_plans.py \
    --input_dir val_files \
    --output_dir datasets/my-custom-floorplans/val \
    --flip_only  # 4× instead of 8×
```

### 4. Verify Data Quality After Augmentation

```bash
# Test that augmented files load correctly
python inspect_test_data.py  # Your inspection script

# Or quick check:
python -c "
import numpy as np
from pathlib import Path

augmented_dir = Path('augmented_npy')
for f in list(augmented_dir.glob('*.npy'))[:5]:  # Check first 5
    data = np.load(f, allow_pickle=True).item()
    coords = data['corner_list_np_normalized_padding_withsemantics'][:, :2]
    print(f'{f.name}: {coords.shape}, coords in range [{coords.min():.2f}, {coords.max():.2f}]')
    assert coords.min() >= -1.5 and coords.max() <= 1.5, 'Coords out of range!'
print('✓ All checks passed')
"
```

### 5. Document Your Augmentation Strategy

Create a metadata file in your dataset directory:

```bash
cat > datasets/my-custom-floorplans/DATASET_INFO.txt << 'EOF'
Dataset: Custom Floor Plans
Created: 2025-11-14

Original Files: 1,000 floor plans
Augmentation: Full (8× - flips + rotations)
Total Training Files: 8,000

Augmentation Command:
python augment_floor_plans.py \
    --input_dir converted_npy \
    --output_dir augmented_npy \
    --full

Split:
- Train: 5,600 files (70% of augmented, groups 0-699)
- Val: 1,200 files (15% originals only, groups 700-849)
- Test: 1,200 files (15% originals only, groups 850-999)
EOF
```

---

## Recommended Directory Structure

After completing all steps, your repository should look like:

```
GSDiff/
├── datasets/
│   └── my-custom-floorplans/       # Your augmented dataset
│       ├── train/
│       │   ├── 0_original.npy      # Or renamed to 0.npy, 1.npy, etc.
│       │   ├── 0_hflip.npy
│       │   ├── 0_vflip.npy
│       │   ├── ... (8× augmented files)
│       │   └── 999_rot90_hflip.npy
│       ├── val/
│       │   ├── 700.npy             # Original files only
│       │   ├── 701.npy
│       │   └── ...
│       ├── test/
│       │   ├── 850.npy             # Original files only
│       │   ├── 851.npy
│       │   └── ...
│       └── DATASET_INFO.txt        # Metadata
│
├── datasets/prerunning_cnn_featuremaps/  # (if using boundary constraint)
│   ├── 0_original.npy
│   ├── 0_hflip.npy
│   └── ...
│
├── scripts/
│   └── trainval_main_unconstrained.py    # Modified to point to your dataset
│
├── augment_floor_plans.py
└── README_DOCS.md
```

---

## Quick Reference Commands

### Complete Workflow (Copy-Paste Ready)

```bash
# 1. Convert your data to NPY
python raster_to_graph_converter.py \
    --input_dir /path/to/your/data \
    --output_dir converted_npy \
    --image_size 2575 \
    --validate

# 2. Augment (8× increase)
python augment_floor_plans.py \
    --input_dir converted_npy \
    --output_dir augmented_npy \
    --full \
    --create_manifest

# 3. Create dataset directories
mkdir -p datasets/my-custom-floorplans/{train,val,test}

# 4. Split data (manual example - adjust paths)
# Train: All augmented files from groups 0-699
cp augmented_npy/{0..699}_*.npy datasets/my-custom-floorplans/train/

# Val: Original files only from groups 700-849
cp converted_npy/{700..849}.npy datasets/my-custom-floorplans/val/

# Test: Original files only from groups 850-999
cp converted_npy/{850..999}.npy datasets/my-custom-floorplans/test/

# 5. Generate CNN features (if using boundary constraint)
python create_cnn_featuremaps.py \
    --test_dir datasets/my-custom-floorplans/train \
    --create_withboundary

# 6. Update training script
# Edit scripts/trainval_main_unconstrained.py:
#   data_root='datasets/my-custom-floorplans'

# 7. Train!
python scripts/trainval_main_unconstrained.py
```

---

## Disk Space Planning

Calculate required space for your augmented dataset:

**Formula:**

```
Augmented size = Original size × Augmentation factor

Example with 1,000 files:
- Original: 1,000 files × 150KB = 150MB
- After flip_only (4×): 4,000 files = 600MB (+450MB)
- After full (8×): 8,000 files = 1.2GB (+1.05GB)
```

**Your calculation:**

```python
# Quick calculator
original_files = 1000        # Your number of files
avg_file_size_mb = 0.15      # Average NPY file size in MB
augmentation_factor = 8      # 4 for flip_only, 8 for full

original_size_mb = original_files * avg_file_size_mb
augmented_size_mb = original_size_mb * augmentation_factor
increase_mb = augmented_size_mb - original_size_mb

print(f"Original: {original_size_mb:.1f} MB")
print(f"Augmented: {augmented_size_mb:.1f} MB")
print(f"Increase: +{increase_mb:.1f} MB")
```

---

## Advantages of Pre-Augmentation

✅ **Simpler workflow** - Augment once, train many times
✅ **No code changes** - Training scripts work as-is
✅ **Faster training** - No on-the-fly augmentation overhead
✅ **Easy to verify** - Inspect augmented files before training
✅ **Reproducible** - Same augmented dataset for all experiments
✅ **Easy to share** - Can share augmented dataset with collaborators

---

## Summary Checklist

- [ ] Convert your data to GSDiff NPY format (correct image_size!)
- [ ] Augment with `augment_floor_plans.py` (flip_only or full)
- [ ] Verify augmented files load correctly
- [ ] Split into train (augmented) / val (original) / test (original)
- [ ] Generate CNN feature maps (if using boundary constraint)
- [ ] Update `data_root` in training script
- [ ] Document your augmentation strategy
- [ ] Train normally

**Result:** Your custom augmented dataset is now ready for training!

---

**Created:** 2025-11-14
**For use with:** GSDiff repository pre-augmentation workflow
