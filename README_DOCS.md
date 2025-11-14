# GSDiff Documentation and Tools

This branch contains comprehensive documentation and tools for training and augmenting GSDiff floor plan generation models.

## 📚 Documentation

### 1. [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
**Complete training documentation (1,150 lines)**

Topics covered:
- **Training Pipeline Overview** - Multi-stage training workflow
- **All Training Scripts** - Location and purpose of 15+ training scripts
- **The 53 Corner Limit** - Complete analysis of where it's used (50+ files)
- **Data Preprocessing Pipeline** - Step-by-step from raw images to NPY files
- **How to Retrain on New Data** - Complete workflow with examples
- **How to Increase Corner Capacity** - Detailed guide to change from 53 to any limit
- **Model Architecture Files** - Reference for all models
- **Configuration Parameters** - All hyperparameters explained

**Use this when:**
- Starting training from scratch
- Retraining with custom data
- Increasing corner limit beyond 53
- Understanding the preprocessing pipeline

---

### 2. [DATA_AUGMENTATION_GUIDE.md](DATA_AUGMENTATION_GUIDE.md)
**Data augmentation documentation (1,000+ lines)**

Topics covered:
- **Quick Start Examples** - Get started in minutes
- **Augmentation Strategies** - Flip-only (4×) vs Full (8×)
- **Technical Details** - Transformation mathematics
- **Training Integration** - How to use augmented data
- **Best Practices** - When and how to augment
- **Disk Space Calculations** - Plan storage requirements
- **Troubleshooting** - Common issues and solutions

**Use this when:**
- You have limited training data (< 10,000 samples)
- Want to improve model generalization
- Need to increase dataset size without manual collection
- Training with custom floor plan data

---

## 🛠️ Tools

### [augment_floor_plans.py](augment_floor_plans.py)
**Geometric data augmentation script**

**Features:**
- Increase dataset by 4× (flips) or 8× (flips + rotations)
- Preserves graph structure and semantic information
- Processes ~50-100 files/second
- Maintains train/val/test directory structure
- Creates augmentation manifests for tracking

**Quick Examples:**

```bash
# Basic augmentation (4× dataset)
python augment_floor_plans.py \
    --input_dir datasets/train \
    --output_dir datasets_augmented/train \
    --flip_only

# Full augmentation (8× dataset)
python augment_floor_plans.py \
    --input_dir datasets/train \
    --output_dir datasets_augmented/train \
    --full

# Test on single file
python augment_floor_plans.py \
    --input datasets/train/0.npy \
    --output_dir test_aug \
    --full
```

**Transformations:**
- Horizontal flip (mirror left-right)
- Vertical flip (mirror top-bottom)
- 90°, 180°, 270° rotations
- Combinations (e.g., rotation + flip)

---

## 🚀 Quick Start Guide

### For Training New Models

1. **Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Understand the pipeline
2. **Prepare your data** - Follow preprocessing steps in guide
3. **(Optional) Augment data** - Use `augment_floor_plans.py` to increase dataset
4. **Run training** - Use scripts in `scripts/trainval_*.py`

### For Data Augmentation

1. **Read [DATA_AUGMENTATION_GUIDE.md](DATA_AUGMENTATION_GUIDE.md)**
2. **Test on single file** - Verify augmentation works correctly
3. **Augment full dataset** - Process all training data
4. **Update training config** - Point to augmented data directory

### For Increasing Corner Capacity

1. **Read section in [TRAINING_GUIDE.md](TRAINING_GUIDE.md#how-to-increase-corner-capacity)**
2. **Update `padding_to_number` in rplan-process4.py**
3. **Regenerate all data** - Run preprocessing pipeline
4. **Update all 50+ files** - Change hardcoded 53 values
5. **Retrain models** - Train with new capacity

---

## 📊 Use Cases

### Use Case 1: Small Custom Dataset
**Scenario:** 500 custom floor plans from JSON files

**Solution:**
```bash
# 1. Augment to increase dataset
python augment_floor_plans.py \
    --input_dir my_data/train \
    --output_dir my_data_augmented/train \
    --full
# Result: 500 → 4,000 files (8× increase)

# 2. Train with augmented data
# Update dataset path in trainval_main_unconstrained.py
python scripts/trainval_main_unconstrained.py
```

### Use Case 2: Increase Corner Limit
**Scenario:** Your floor plans have 80+ corners, but GSDiff max is 53

**Solution:**
1. Follow **"How to Increase Corner Capacity"** in TRAINING_GUIDE.md
2. Change `padding_to_number = 53` → `100` in rplan-process4.py
3. Update all 50+ files with hardcoded 53
4. Regenerate all NPY files
5. Retrain models with new capacity

### Use Case 3: Retrain on Different Domain
**Scenario:** Train on commercial buildings instead of residential

**Solution:**
1. Follow **"How to Retrain on New Data"** in TRAINING_GUIDE.md
2. Run preprocessing pipeline on new images
3. (Optional) Augment training data with `augment_floor_plans.py`
4. Update semantic labels if needed
5. Run training scripts

---

## 📁 File Reference

| File | Type | Purpose | Size |
|------|------|---------|------|
| `TRAINING_GUIDE.md` | Documentation | Complete training reference | 35KB |
| `DATA_AUGMENTATION_GUIDE.md` | Documentation | Augmentation guide | 17KB |
| `augment_floor_plans.py` | Script | Data augmentation tool | 14KB |

---

## 🔗 Related Files in Repository

**Training Scripts:**
- `scripts/trainval_main_unconstrained.py` - Stage 1 unconstrained training
- `scripts/trainval_main_topo.py` - Stage 1 topology-constrained
- `scripts/trainval_main_boun.py` - Stage 1 boundary-constrained
- `scripts/trainval_main_edge_*.py` - Stage 2 edge prediction
- `scripts/train-CNN-autoe*.py` - CNN boundary encoder training

**Data Preprocessing:**
- `datasets/rplan-extract.py` - Extract corners from images
- `datasets/rplan-process4.py` - **Main preprocessing** (defines corner limit)
- `datasets/rplan-process5-7.py` - Add boundary features
- `datasets/rplan-process8-10.py` - Generate bubble diagrams

**Model Architectures:**
- `gsdiff/heterhouse_*.py` - Node generation models
- `gsdiff/boundary_78_10.py` - CNN boundary encoder
- `gsdiff/bubble_diagram_57_9.py` - Topology transformer

**Dataset Loaders:**
- `datasets/rplang_edge_semantics_simplified*.py` - Graph dataset loaders

---

## 💡 Tips and Best Practices

### Training
✅ **Do:**
- Read TRAINING_GUIDE.md thoroughly before starting
- Start with unconstrained variant (simplest)
- Use smaller batch size if GPU memory limited
- Monitor TensorBoard logs during training

❌ **Don't:**
- Skip preprocessing steps
- Augment test set (keep original for fair evaluation)
- Change corner limit without updating all 50+ files
- Train without validating data format first

### Data Augmentation
✅ **Do:**
- Test on single file first (`--input single_file.npy`)
- Use `--flip_only` for real architectural data
- Use `--full` for small datasets (< 10,000 samples)
- Create manifest (`--create_manifest`) for tracking

❌ **Don't:**
- Augment test/validation sets (inflates metrics)
- Use rotations for real buildings (unrealistic orientations)
- Forget to check disk space (8× = significant increase)
- Skip visual verification of augmented samples

---

## 🆘 Getting Help

### Common Questions

**Q: Where do I start?**
A: Read TRAINING_GUIDE.md sections 1-3, then run test augmentation on one file.

**Q: My dataset is small (< 1000 files). What should I do?**
A: Use `augment_floor_plans.py --full` to increase it 8×, then follow training guide.

**Q: How do I change the 53 corner limit?**
A: See "How to Increase Corner Capacity" in TRAINING_GUIDE.md (complete file list provided).

**Q: Can I augment only some files?**
A: Yes, see "Selective Augmentation" in DATA_AUGMENTATION_GUIDE.md for examples.

**Q: Training failed with shape mismatch error?**
A: Check that all NPY files have correct padding (53, ...) shape and validate with inspect_test_data.py.

---

## 📝 Summary

This branch provides everything needed to:
- ✅ Understand GSDiff training pipeline
- ✅ Retrain on custom data
- ✅ Increase dataset size with augmentation
- ✅ Change corner capacity limit
- ✅ Configure training parameters
- ✅ Troubleshoot common issues

**Branch:** `claude/add-documentation-and-tools-011CV5ZXcHng7Pc1J8k7KVsG`

**Created:** 2025-11-14

**Status:** Ready to merge to main
