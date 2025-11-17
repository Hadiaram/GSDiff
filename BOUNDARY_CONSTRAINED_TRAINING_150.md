# Boundary-Constrained Training Guide (150 Corners)

**Complete workflow for training GSDiff with boundary constraints and 150-corner capacity**

---

## Overview

This guide covers the complete process for boundary-constrained training of GSDiff with increased corner capacity (150 instead of the original 53). Boundary-constrained training uses CNN-extracted boundary features as conditioning, providing more accurate control over floor plan generation.

## Prerequisites

- Augmented floor plan dataset (1000+ files recommended)
- Dataset split into train/val/test (use `split_dataset.py`)
- All graph NPY files with 150-corner capacity
- GPU with sufficient VRAM (boundary-constrained training requires ~16GB+)

---

## Step 1: Organize Your Data

### 1.1 Directory Structure

Your data needs to be in these specific locations:

```
datasets/
├── rplang-v3-withsemantics/           # Full graph data
│   ├── train/                          # Training set
│   ├── val/                            # Validation set
│   └── test/                           # Test set
├── rplang-v3-withsemantics-withboundary/  # Simplified withboundary data
│   ├── train/
│   ├── val/
│   └── test/
└── prerunning_cnn_featuremaps/        # CNN feature maps (NO SPLITS!)
    ├── file1.npy
    ├── file2.npy
    └── ...
```

### 1.2 Copy Your Data

If you have your augmented data in `augmented_npy_split/`, copy it to the correct locations:

```bash
# Copy full graph data
cp augmented_npy_split/train/* datasets/rplang-v3-withsemantics/train/
cp augmented_npy_split/val/* datasets/rplang-v3-withsemantics/val/
cp augmented_npy_split/test/* datasets/rplang-v3-withsemantics/test/
```

---

## Step 2: Generate Required Support Files

Boundary-constrained training requires two additional file types:

1. **CNN feature maps**: Extracted features from CNN encoder (format: `{16: [array(1024, 16, 16)]}`)
2. **Withboundary files**: Simplified graph format with only corners and padding masks

### 2.1 Generate All Support Files

Run this single command to generate both CNN feature maps and withboundary files for all splits:

```bash
python create_cnn_featuremaps.py \
    --split_root augmented_npy_split \
    --feature_dir datasets/prerunning_cnn_featuremaps \
    --withboundary_dir datasets/rplang-v3-withsemantics-withboundary \
    --create_withboundary \
    --max_corners 150
```

**Important Notes:**
- CNN feature maps go into a **single directory** (no train/val/test splits) - this is by design!
- The dataset loader expects all feature maps in `datasets/prerunning_cnn_featuremaps/`
- Withboundary files **do** split into train/val/test subdirectories
- These are dummy/random feature maps for testing - real training needs actual CNN-extracted features

### 2.2 For Production Training (Optional)

For real experiments with meaningful results, you need actual CNN-extracted features:

```bash
# Extract real CNN features using pretrained CNN encoder
python scripts/prerunningCNN.py \
    --input_dir datasets/rplang-v3-withsemantics \
    --output_dir datasets/prerunning_cnn_featuremaps \
    --max_corners 150
```

---

## Step 3: Verify Data Integrity

Before training, verify all files are in place:

```bash
# Check file counts
echo "Train files: $(ls datasets/rplang-v3-withsemantics/train/*.npy | wc -l)"
echo "Val files: $(ls datasets/rplang-v3-withsemantics/val/*.npy | wc -l)"
echo "Test files: $(ls datasets/rplang-v3-withsemantics/test/*.npy | wc -l)"

echo "Withboundary train: $(ls datasets/rplang-v3-withsemantics-withboundary/train/*.npy | wc -l)"
echo "Withboundary val: $(ls datasets/rplang-v3-withsemantics-withboundary/val/*.npy | wc -l)"
echo "Withboundary test: $(ls datasets/rplang-v3-withsemantics-withboundary/test/*.npy | wc -l)"

echo "CNN feature maps: $(ls datasets/prerunning_cnn_featuremaps/*.npy | wc -l)"
```

**Expected output:**
- All counts should match (e.g., 1000 train + 50 val + 50 test = 1100 total)
- CNN feature maps count should equal total files across all splits

---

## Step 4: Configure Training Scripts

### 4.1 Update System Paths

Edit the training scripts to match your system paths:

**File: `scripts/trainval_main_boun.py`** (Lines 2-5)
```python
sys.path.append('/home/user/GSDiff')  # Update to your project root
sys.path.append('/home/user/GSDiff/datasets')
sys.path.append('/home/user/GSDiff/gsdiff')
sys.path.append('/home/user/GSDiff/scripts/metrics')
```

**File: `scripts/trainval_main_edge_boun.py`** (Lines 3-5)
```python
sys.path.append('/home/user/GSDiff')  # Update to your project root
sys.path.append('/home/user/GSDiff/datasets')
sys.path.append('/home/user/GSDiff/gsdiff')
```

### 4.2 Configure Hyperparameters

**Stage 1 (Node Generation) - `trainval_main_boun.py`:**
```python
diffusion_steps = 1000
lr = 1e-4
weight_decay = 1e-7
total_steps = 1000000
batch_size = 256
batch_size_val = 3000
device = 'cuda:0'  # Update to your GPU
```

**Stage 2 (Edge Prediction) - `trainval_main_edge_boun.py`:**
```python
lr = 1e-4
weight_decay = 1e-5
total_steps = float("inf")  # Or set specific value like 200000
batch_size = 4  # Lower batch size due to memory constraints
device = 'cuda:0'  # Update to your GPU
```

### 4.3 Create Output Directories

Training scripts will create output directories to save checkpoints and logs:

```bash
# Stage 1 output (configure in script)
mkdir -p outputs/stage1_boun_150corners

# Stage 2 output (configure in script)
mkdir -p outputs/stage2_boun_150corners
```

---

## Step 5: Train Stage 1 - Node Generation (Boundary-Constrained)

Stage 1 trains the diffusion model to generate corner positions and semantic labels conditioned on CNN boundary features.

### 5.1 Start Training

```bash
cd /home/user/GSDiff
python scripts/trainval_main_boun.py
```

### 5.2 Training Details

**What happens during training:**
- Loads data from `RPlanGEdgeSemanSimplified_81` dataset loader
- Uses `BoundHeterHouseModel` from `heterhouse_81_106_3.py`
- Applies truncated normal noise to corners
- Computes diffusion loss with boundary conditioning
- Validates every N steps using validation set
- Saves checkpoints periodically

**Expected timeline:**
- Total steps: 1,000,000
- Time per step: ~0.5-1.0 seconds (depends on GPU)
- Total training time: ~5-10 days on single GPU

**Monitoring:**
- Loss should decrease over time
- Validation loss should track training loss
- Check tensorboard logs if configured

### 5.3 Checkpoints

Model checkpoints are saved in the output directory:
```
outputs/stage1_boun_150corners/
├── checkpoint_step_10000.pt
├── checkpoint_step_20000.pt
└── ...
```

---

## Step 6: Train Stage 2 - Edge Prediction (Boundary-Constrained)

Stage 2 trains the edge prediction model to generate walls between corners.

### 6.1 Load Stage 1 Checkpoint

Edit `scripts/trainval_main_edge_boun.py` to load your Stage 1 checkpoint:

```python
# Around line 50-60, add checkpoint loading:
model_stage1 = BoundHeterHouseModel().to(device)
checkpoint = torch.load('outputs/stage1_boun_150corners/checkpoint_final.pt')
model_stage1.load_state_dict(checkpoint['model_state_dict'])
model_stage1.eval()
```

### 6.2 Start Training

```bash
python scripts/trainval_main_edge_boun.py
```

### 6.3 Training Details

**What happens during training:**
- Loads corners from Stage 1 model
- Uses `BoundEdgeModel` to predict edge connectivity
- Applies perturbations to corners and semantics
- Computes edge prediction loss
- Validates on validation set

**Expected timeline:**
- Steps: Until convergence (monitor validation loss)
- Typically converges in 50,000-200,000 steps
- Time: 1-3 days on single GPU

---

## Step 7: Testing and Evaluation

### 7.1 Run Boundary-Constrained Testing

```bash
python scripts/test_boun.py \
    --stage1_checkpoint outputs/stage1_boun_150corners/checkpoint_final.pt \
    --stage2_checkpoint outputs/stage2_boun_150corners/checkpoint_final.pt \
    --output_dir results/test_boun_150 \
    --max_corners 150
```

### 7.2 Evaluation Metrics

The test script will compute:
- **FID (Fréchet Inception Distance)**: Measures generation quality
- **KID (Kernel Inception Distance)**: Alternative quality metric
- **Validity**: Percentage of valid floor plans
- **Diversity**: Variety in generated layouts
- **Boundary Alignment**: How well generated plans match boundary constraints

### 7.3 Visualize Results

Results are saved as NPY files. To visualize:

```bash
python visualize_results.py \
    --input_dir results/test_boun_150 \
    --output_dir visualizations/
```

---

## File Updates Summary

All files have been updated from 53 to 150 corners:

### Updated Files (6 total):

1. **`datasets/rplang_edge_semantics_simplified_81.py`**
   - Dataset loader for boundary-constrained training
   - Fixed filename sorting for descriptive names
   - Updated attention matrix from (53, 53) to (150, 150)

2. **`scripts/trainval_main_boun.py`**
   - Stage 1 training script
   - Updated distance matrix operations
   - Updated corner count normalization

3. **`scripts/trainval_main_edge_boun.py`**
   - Stage 2 training script
   - Updated corner noise tensors
   - Updated semantic perturbation tensors

4. **`scripts/trainval_simplified_edge_boun.py`**
   - Simplified Stage 2 variant
   - Updated corner and semantic tensor sizes

5. **`scripts/test_boun.py`**
   - Boundary-constrained testing script
   - Updated all tensor allocations to 150 corners

6. **Model files** (may need additional updates):
   - `gsdiff/heterhouse_81_106_3.py` - BoundHeterHouseModel
   - Check for any remaining hardcoded 53 references

---

## Troubleshooting

### Issue: "RuntimeError: shape mismatch"
**Solution:** Verify all files are using 150-corner capacity. Check conversion logs.

### Issue: "FileNotFoundError: prerunning_cnn_featuremaps"
**Solution:** Run `create_cnn_featuremaps.py` to generate feature maps for all files.

### Issue: "CUDA out of memory"
**Solution:** Reduce batch_size in training scripts. For Stage 1, try 128 instead of 256.

### Issue: "ValueError: invalid literal for int()"
**Solution:** Descriptive filenames are supported. Dataset loader has fallback sorting.

### Issue: Training loss not decreasing
**Solution:**
- Check data integrity (all files have correct shape)
- Verify CNN feature maps match graph files
- Try reducing learning rate (1e-5 instead of 1e-4)

---

## Performance Optimization

### GPU Memory Usage

**Expected memory usage:**
- Stage 1 training: ~12-16GB with batch_size=256
- Stage 2 training: ~8-12GB with batch_size=4
- Testing: ~6-8GB

**To reduce memory:**
- Decrease batch_size
- Use gradient checkpointing (requires model modification)
- Use mixed precision training (FP16)

### Training Speed

**Optimization tips:**
- Use `num_workers > 0` in DataLoader for parallel data loading
- Pin memory: `pin_memory=True`
- Use faster storage (SSD) for datasets
- Preload CNN feature maps to memory (currently done in `__init__`)

---

## Next Steps

After successful boundary-constrained training:

1. **Compare with unconstrained training** - See if boundary constraints improve quality
2. **Fine-tune hyperparameters** - Experiment with learning rates, batch sizes
3. **Retrain with real data** - If you used dummy CNN features, retrain with actual CNN features
4. **Scale up** - Train on larger datasets for better generalization
5. **Deploy** - Use trained models for floor plan generation tasks

---

## References

- Main documentation: `README_DOCS.md`
- Training guide: `TRAINING_GUIDE.md`
- Theoretical background: `DIFFUSION_MODEL_RETRAINING_THEORY.md`
- Data augmentation: `DATA_AUGMENTATION_GUIDE.md`

---

## Contact & Support

For issues, questions, or contributions, refer to the main GSDiff repository and documentation.

**Last Updated:** 2025-11-17
**Corner Capacity:** 150
**GSDiff Version:** Boundary-Constrained Variant
