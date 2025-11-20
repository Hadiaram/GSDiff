# GSDiff Documentation and Tools

This branch contains comprehensive documentation and tools for training and augmenting GSDiff floor plan generation models.

## 📚 Documentation

### 1. [PRE_AUGMENTATION_WORKFLOW.md](PRE_AUGMENTATION_WORKFLOW.md) ⭐ RECOMMENDED
**Step-by-step workflow for preparing custom data**

**This is the guide you need if you're augmenting BEFORE placing data in the repo!**

Topics covered:
- **Complete workflow** - Convert → Augment → Place in repo → Train
- **Step-by-step process** - 6 clear steps with copy-paste commands
- **File naming** - How to handle augmented filenames
- **Dataset splitting** - Train/val/test organization
- **Best practices** - What to augment, what to keep original
- **Automated scripts** - Python script for splitting dataset
- **Quick reference** - All commands in one place

**Use this when:**
- Preparing custom floor plan data for training
- You want to augment BEFORE adding to repository
- You need a simple, streamlined workflow
- You're new to the augmentation process

---

### 2. [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
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

### 3. [BOUNDARY_CONSTRAINED_TRAINING_150.md](BOUNDARY_CONSTRAINED_TRAINING_150.md) ⭐ READY TO TRAIN
**Complete workflow for boundary-constrained training with 150 corners**

Topics covered:
- **Complete Step-by-Step Workflow** - From data organization to testing
- **Directory Structure** - Exact paths and organization required
- **Support File Generation** - CNN feature maps and withboundary files
- **Training Configuration** - Hyperparameters, paths, and GPU settings
- **Stage 1 & Stage 2 Training** - Detailed instructions for both stages
- **Updated Scripts** - All 6 files updated from 53 to 150 corners
- **Testing & Evaluation** - How to test trained models
- **Troubleshooting** - Common issues and solutions
- **Performance Optimization** - Memory usage and training speed tips

**Use this when:**
- You have 1000+ augmented files ready for training
- Want to use boundary-constrained variant (more accurate)
- Need 150-corner capacity for complex floor plans
- Ready to start training right now
- Have split data into train/val/test

---

### 4. [RETRAINING_WITH_EXISTING_DATA.md](RETRAINING_WITH_EXISTING_DATA.md) ⭐
**How to retrain while preserving knowledge from previous data**

Topics covered:
- **Two Approaches** - Combined training vs fine-tuning from checkpoint
- **Step-by-Step Workflows** - Complete examples for both approaches
- **Checkpoint Loading** - How to resume from saved models
- **Fine-Tuning Strategy** - Learning rates, steps, and avoiding catastrophic forgetting
- **Handling Different Corner Counts** - Mixing datasets with different capacities
- **Training Configuration** - All three stages explained
- **Monitoring Progress** - Loss curves, FID/KID metrics, validation
- **Best Practices** - When to combine data vs fine-tune
- **Complete Example** - Adding 10k new floor plans to existing 65k dataset

**Use this when:**
- Adding new data to an already-trained model
- Want to preserve previous knowledge while learning new patterns
- Have checkpoints from previous training runs
- Need to retrain without starting completely from scratch
- Combining multiple datasets (old + new data)

---

### 5. [DATA_AUGMENTATION_GUIDE.md](DATA_AUGMENTATION_GUIDE.md)
**Data augmentation documentation (1,000+ lines) - Technical reference**

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

### 6. [DIFFUSION_MODEL_RETRAINING_THEORY.md](DIFFUSION_MODEL_RETRAINING_THEORY.md) 📖 THEORY
**Deep theoretical guide to diffusion model retraining (for experienced ML practitioners)**

Topics covered:
- **Foundational Concepts** - How diffusion models learn, knowledge representation, forward/reverse processes
- **Adaptation Methods** - Fine-tuning, LoRA, DreamBooth, adapters, transfer learning, continual learning
- **Avoiding Catastrophic Forgetting** - EWC, knowledge distillation, regularization techniques
- **Layer-Specific Strategies** - UNet architecture breakdown, selective freezing, progressive unfreezing
- **Hyperparameter Deep Dive** - Learning rates, batch sizes, noise schedules, optimization
- **Evaluation Methodology** - Retention testing, mode collapse detection, bias assessment
- **Advanced Techniques** - Gradient surgery, task arithmetic, adaptive loss weighting
- **Deployment Best Practices** - Weight merging, versioning, integration strategies

**Use this when:**
- Need deep understanding of diffusion model theory
- Exploring different retraining strategies (LoRA vs full fine-tuning)
- Implementing parameter-efficient fine-tuning methods
- Understanding trade-offs between different approaches
- Want theoretical foundation for practical GSDiff work
- Coming from general diffusion model background (Stable Diffusion, etc.)

**Note:** This is a theoretical/educational guide covering general diffusion model concepts. For GSDiff-specific practical workflows, see [RETRAINING_WITH_EXISTING_DATA.md](RETRAINING_WITH_EXISTING_DATA.md).

---

## 🛠️ Tools

### [json_to_npy_floodfill.py](json_to_npy_floodfill.py)
**Convert BIM JSON files to raster NPY format (Step 1 of pipeline)**

**Features:**
- Converts JSON floor plans (BIM format) to raster arrays
- Uses flood fill algorithm for proper room boundaries
- Handles walls, doors, and room separation
- Configurable resolution and wall thickness
- Creates visualization PNGs for verification

**Quick Example:**

```bash
# Convert single JSON file
python json_to_npy_floodfill.py \
    --input apartment_1.json \
    --output apartment_1.npy \
    --resolution 10.0

# Batch convert directory
python json_to_npy_floodfill.py \
    --input_dir JSON_files \
    --output_dir raster_npy_files
```

**Output:** Raster NPY files (2D arrays where each pixel = room ID)

---

### [raster_to_graph_converter.py](raster_to_graph_converter.py)
**Convert raster NPY to graph NPY format (Step 2 of pipeline)**

**Features:**
- Converts raster arrays to GSDiff graph format
- Extracts corners using OpenCV contour detection
- Builds adjacency matrices automatically
- Normalizes coordinates to [-1, 1] range
- Pads to 53 corners (or custom size)
- Validates output format

**Quick Example:**

```bash
# Convert raster NPY files to graph format
python raster_to_graph_converter.py \
    --input_dir raster_npy_files \
    --output_dir datasets/my-custom-floorplans/train \
    --image_size 2575 \
    --validate
```

**Critical:** Must specify `--image_size` matching your raster dimensions!

---

### [augment_floor_plans.py](augment_floor_plans.py)
**Geometric data augmentation script (Step 3 of pipeline)**

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

### [split_dataset.py](split_dataset.py)
**Split NPY files into train/validation/test sets**

**Features:**
- Configurable split ratios (default: 90/5/5)
- Reproducible with random seed
- Progress tracking with tqdm
- Preserves original files (copies instead of moving)
- Validates ratio sum to 1.0

**Quick Examples:**

```bash
# Standard 90/5/5 split
python split_dataset.py \
    --input_dir graph_npy

# Custom ratios (80/10/10)
python split_dataset.py \
    --input_dir graph_npy \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

# Specify output directory
python split_dataset.py \
    --input_dir graph_npy \
    --output_dir my_split_data

# Use different random seed
python split_dataset.py \
    --input_dir graph_npy \
    --seed 123
```

**Output Structure:**
```
graph_npy_split/
├── train/     # 90% of files
├── val/       # 5% of files
└── test/      # 5% of files
```

---

### [create_cnn_featuremaps.py](create_cnn_featuremaps.py)
**Generate CNN feature maps for boundary-constrained testing**

**Features:**
- Creates dummy CNN feature maps for testing boundary-constrained models
- Generates withboundary files for test datasets
- Configurable corner capacity (default: 150)
- Matches test data file names automatically

**Quick Examples:**

```bash
# Create feature maps for test data
python create_cnn_featuremaps.py --test_dir datasets/test

# With custom corner limit
python create_cnn_featuremaps.py \
    --test_dir datasets/test \
    --max_corners 150

# Also create withboundary files
python create_cnn_featuremaps.py \
    --test_dir datasets/test \
    --create_withboundary

# Custom output directories
python create_cnn_featuremaps.py \
    --test_dir datasets/test \
    --feature_dir my_features \
    --withboundary_dir my_withboundary
```

**When to use:**
- Running boundary-constrained model testing (`test_boun.py`)
- Need CNN feature maps but don't have trained CNN encoder
- Quick testing/debugging of boundary-constrained generation

**Note:** Creates **dummy/random** feature maps for testing only. For real experiments, train CNN boundary encoder with `scripts/train-CNN-autoe-final.py`.

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

### Use Case 3: Add New Data While Preserving Old Knowledge
**Scenario:** You have a trained model on 60k floor plans, want to add 10k new ones

**Solution:**
```bash
# Follow RETRAINING_WITH_EXISTING_DATA.md

# Option A: Combine all data (recommended)
cp new_data/*.npy ../datasets/rplang-v3-withsemantics/train/
python scripts/trainval_main_unconstrained.py

# Option B: Fine-tune from checkpoint (faster)
# Modify training script to load checkpoint
# Set lr = 1e-5 (lower than default 1e-4)
# Train for 200k steps instead of 1M
python scripts/trainval_main_unconstrained.py
```

### Use Case 4: Retrain on Different Domain
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
| `README_DOCS.md` | Documentation | Master index (this file) | 13KB |
| `PRE_AUGMENTATION_WORKFLOW.md` | Documentation | ⭐ Recommended workflow for custom data | 15KB |
| `TRAINING_GUIDE.md` | Documentation | Complete training reference | 35KB |
| `BOUNDARY_CONSTRAINED_TRAINING_150.md` | Documentation | ⭐ Boundary-constrained 150-corner workflow | 18KB |
| `RETRAINING_WITH_EXISTING_DATA.md` | Documentation | ⭐ Retrain with old + new data | 24KB |
| `DATA_AUGMENTATION_GUIDE.md` | Documentation | Technical augmentation reference | 17KB |
| `DIFFUSION_MODEL_RETRAINING_THEORY.md` | Documentation | 📖 Theoretical diffusion model guide | 31KB |
| `json_to_npy_floodfill.py` | Script | JSON → Raster NPY converter | 17KB |
| `raster_to_graph_converter.py` | Script | Raster NPY → Graph NPY converter | 16KB |
| `augment_floor_plans.py` | Script | Graph NPY augmentation tool | 14KB |
| `split_dataset.py` | Script | Train/val/test splitter | 6KB |
| `create_cnn_featuremaps.py` | Script | CNN feature map generator (for testing) | 5KB |

**Total: 12 files** (7 documentation guides + 5 Python scripts)

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
- ✅ Retrain on custom data (with or without preserving previous knowledge)
- ✅ Deep theoretical understanding of diffusion model retraining
- ✅ Increase dataset size with augmentation
- ✅ Change corner capacity limit
- ✅ Configure training parameters
- ✅ Troubleshoot common issues
- ✅ Learn advanced retraining techniques (LoRA, DreamBooth, EWC, etc.)

**Branch:** `claude/add-documentation-and-tools-011CV5ZXcHng7Pc1J8k7KVsG`

**Created:** 2025-11-14

**Status:** Ready to merge to main
