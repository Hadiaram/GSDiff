# Retraining GSDiff with Existing Data

This guide explains how to retrain GSDiff while preserving knowledge from your previous dataset. You have two main approaches: combining datasets or fine-tuning from checkpoints.

---

## Table of Contents

1. [Quick Answer](#quick-answer)
2. [Approach 1: Combine All Data (Recommended)](#approach-1-combine-all-data-recommended)
3. [Approach 2: Fine-Tuning from Checkpoint](#approach-2-fine-tuning-from-checkpoint)
4. [Handling Different Corner Counts](#handling-different-corner-counts)
5. [Training Configuration](#training-configuration)
6. [Monitoring Training Progress](#monitoring-training-progress)
7. [Best Practices](#best-practices)

---

## Quick Answer

**Yes, you can absolutely retrain while preserving previous data!**

The simplest and most effective approach is to **combine your old and new data** into a single directory. GSDiff's dataset loader will automatically use all `.npy` files in the training directory, giving you a model that learns from both datasets.

---

## Approach 1: Combine All Data (Recommended)

This is the **most straightforward and reliable** method. It ensures the model learns from both old and new data equally.

### How the Dataset Loader Works

The dataset class in `datasets/rplang_edge_semantics_simplified_55_100.py` loads **all `.npy` files** from the specified directory:

```python
# From datasets/rplang_edge_semantics_simplified_55_100.py:30-32
if self.mode == 'train':
    self.files = os.listdir('../datasets/rplang-v3-withsemantics/train')
```

It doesn't care about file names or subdirectories - it simply loads every NPY file it finds.

### Step-by-Step Process

#### 1. Prepare Your New Data

Follow the complete pipeline documented in [PRE_AUGMENTATION_WORKFLOW.md](PRE_AUGMENTATION_WORKFLOW.md):

```bash
# Step 1: JSON → Raster NPY
python json_to_npy_floodfill.py \
    --input_dir /path/to/new_json_files \
    --output_dir new_raster_npy \
    --resolution 10.0

# Step 2: Raster NPY → Graph NPY
python raster_to_graph_converter.py \
    --input_dir new_raster_npy \
    --output_dir new_graph_npy \
    --image_size 2575 \
    --validate

# Step 3: Augment (optional but recommended)
python augment_floor_plans.py \
    --input_dir new_graph_npy \
    --output_dir new_augmented_npy \
    --full
```

#### 2. Combine Old and New Data

Simply copy all your new NPY files into the existing training directory:

```bash
# Copy new data to training directory
cp new_augmented_npy/*.npy ../datasets/rplang-v3-withsemantics/train/

# Verify the count
echo "Total training files:"
ls ../datasets/rplang-v3-withsemantics/train/*.npy | wc -l
```

**Example Directory Structure:**
```
datasets/rplang-v3-withsemantics/
├── train/
│   ├── 0.npy          # Old data
│   ├── 1.npy          # Old data
│   ├── ...
│   ├── 65762.npy      # Old data (last original file)
│   ├── 65763.npy      # New data
│   ├── 65764.npy      # New data
│   └── ...            # More new data
├── val/
│   └── ...
└── test/
    └── ...
```

#### 3. Update File Numbering (Optional)

For cleaner organization, you can renumber your new files to continue from where the old dataset ends:

```bash
#!/bin/bash
# Assuming old dataset has files 0.npy to 65762.npy
# Start numbering new files from 65763

cd new_augmented_npy
counter=65763

for file in *.npy; do
    cp "$file" "../datasets/rplang-v3-withsemantics/train/${counter}.npy"
    ((counter++))
done

echo "Added $((counter - 65763)) new training files"
```

#### 4. Create Validation and Test Sets from New Data

Don't forget to add some of your new data to validation and test sets:

```bash
# Example: Use 80% for training, 10% for validation, 10% for test
# (Adjust ratios based on your data size)

total_new=$(ls new_augmented_npy/*.npy | wc -l)
train_count=$(echo "$total_new * 0.8" | bc | cut -d'.' -f1)
val_count=$(echo "$total_new * 0.1" | bc | cut -d'.' -f1)

# Split your new data
# ... (implement splitting logic based on your needs)
```

#### 5. Start Training

Use the standard training scripts:

```bash
# For unconstrained model (Stage 1)
cd scripts
python trainval_main_unconstrained.py

# For topology-constrained model
python trainval_main_topo.py

# For boundary-constrained model
python trainval_main_boun.py
```

### Advantages

- **Simple**: Just copy files to a directory
- **Balanced**: Model sees old and new data with equal probability
- **Reliable**: No complex checkpoint loading or learning rate tuning
- **Complete Training**: Model learns the full distribution from scratch
- **Best for Large Changes**: Ideal when adding significantly different data

### Disadvantages

- **Training Time**: Requires training from scratch (1M+ steps)
- **Computational Cost**: Full training is expensive
- **No Transfer**: Doesn't leverage previously learned weights

---

## Approach 2: Fine-Tuning from Checkpoint

This approach starts from a pre-trained model and continues training on combined data. It's **faster** but requires more careful configuration.

### How Checkpoints Work

Training scripts save checkpoints every 250,000 steps:

```python
# From scripts/trainval_main_unconstrained.py:466-471
if step % interval == 0:
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        state_dict[name] = list(model.parameters())[i]
    torch.save(state_dict, output_dir + f"model{step:07d}.pt")
    torch.save(optimizer.state_dict(), output_dir + f"optim{step:07d}.pt")
```

Saved files:
- `model0250000.pt` - Model weights at step 250k
- `optim0250000.pt` - Optimizer state at step 250k
- `model0500000.pt` - Model weights at step 500k
- etc.

### Step-by-Step Process

#### 1. Prepare Combined Dataset

Same as Approach 1 - combine old and new data in the training directory.

#### 2. Modify Training Script to Load Checkpoint

Edit your chosen training script (e.g., `scripts/trainval_main_unconstrained.py`):

```python
# Find the model initialization section (around line 296)
model = HeterHouseModel().to(device)

# Add checkpoint loading AFTER model initialization:
checkpoint_path = 'outputs/previous-training/model1000000.pt'  # Your checkpoint
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully")
else:
    print("No checkpoint found, starting from scratch")

# Optionally load optimizer state
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)
optim_checkpoint_path = 'outputs/previous-training/optim1000000.pt'
if os.path.exists(optim_checkpoint_path):
    print(f"Loading optimizer state from {optim_checkpoint_path}")
    optimizer.load_state_dict(torch.load(optim_checkpoint_path, map_location=device))
    print("Optimizer state loaded successfully")
```

#### 3. Adjust Learning Rate

When fine-tuning, use a **lower learning rate** than training from scratch:

```python
# Original learning rate (line 26)
# lr = 1e-4

# Fine-tuning learning rate (10x smaller)
lr = 1e-5
```

**Why?** The model already knows the old data well. A lower learning rate prevents "catastrophic forgetting" where the model forgets old knowledge while learning new data.

#### 4. Adjust Training Steps

You don't need a full 1M steps for fine-tuning. Depending on your new data size:

```python
# Original (line 28)
# total_steps = 1000000

# Fine-tuning (example: 200k steps)
total_steps = 200000
```

**Rule of thumb:**
- Small new dataset (< 10% of old data): 100k-200k steps
- Medium new dataset (10-50% of old data): 200k-500k steps
- Large new dataset (> 50% of old data): 500k-1M steps

#### 5. Reset Step Counter (Optional)

If you want to start step counting from 0:

```python
# Find the step initialization (line 301)
step = 0

# Or continue from checkpoint step:
# step = 1000000  # Continue from where checkpoint left off
```

#### 6. Create New Output Directory

Change the output directory to avoid overwriting your old checkpoints:

```python
# Original (line 168)
# output_dir = 'outputs/structure-1/'

# New directory for fine-tuning
output_dir = 'outputs/finetune-with-new-data/'
os.makedirs(output_dir, exist_ok=False)
```

#### 7. Start Fine-Tuning

```bash
cd scripts
python trainval_main_unconstrained.py
```

### Advantages

- **Faster**: Significantly fewer training steps needed
- **Transfer Learning**: Leverages previously learned features
- **Lower Cost**: Less GPU time required
- **Good for Small Additions**: Ideal when adding < 50% new data

### Disadvantages

- **Catastrophic Forgetting Risk**: May lose old knowledge if LR too high
- **Requires Tuning**: Learning rate and steps need careful adjustment
- **Less Thorough**: May not fully integrate new data patterns
- **Checkpoint Dependency**: Requires access to previous checkpoints

---

## Handling Different Corner Counts

### If Both Datasets Have Same Corner Limit (e.g., 53)

No special handling needed - just combine the data as described above.

### If New Data Has More Corners (e.g., 100)

You need to **increase the corner capacity** before training. See [TRAINING_GUIDE.md](TRAINING_GUIDE.md#how-to-increase-corner-capacity) for detailed instructions.

**Important Considerations:**

1. **Model Architecture Changes**: Increasing corner capacity requires modifying 50+ files
2. **Cannot Reuse Checkpoints**: Old checkpoints (trained on 53 corners) are incompatible with new architecture (100 corners)
3. **Must Train from Scratch**: Fine-tuning is NOT possible when changing corner capacity
4. **Memory Impact**: 100 corners = 10,000 edges vs 53 corners = 2,809 edges (4× memory)

**Recommended Workflow:**

```bash
# 1. Increase corner capacity to 100 (see TRAINING_GUIDE.md)
# 2. Convert both old and new data to use 100-corner padding
# 3. Combine all data
# 4. Train from scratch (cannot use old checkpoints)
```

### If Old Data Has More Corners Than New Data

The model will work fine - padding handles variable corner counts. No special action needed.

---

## Training Configuration

### Three Training Stages

GSDiff has three sequential training stages. You must complete them in order:

#### Stage 1: Node Generation (Corner Prediction)

**Scripts:**
- `scripts/trainval_main_unconstrained.py` - Unconstrained
- `scripts/trainval_main_topo.py` - Topology-constrained
- `scripts/trainval_main_boun.py` - Boundary-constrained

**Purpose:** Learn to predict corner locations and room types

**Typical Training Time:** 1M steps ≈ 3-7 days on single GPU

**Output:** Checkpoint like `outputs/structure-1/model1000000.pt`

#### Stage 2: Edge Prediction (Wall Connectivity)

**Scripts:**
- `scripts/trainval_main_edge_unconstrained.py`
- `scripts/trainval_main_edge_topo.py`
- `scripts/trainval_main_edge_boun.py`

**Purpose:** Learn to predict which corners connect with walls

**Depends On:** Stage 1 checkpoint (referenced in script)

**Typical Training Time:** 100k-300k steps

**Output:** Checkpoint like `outputs/structure-3/model_stage2_best_010300.pt`

#### Stage 3: CNN Boundary Encoder (Optional)

**Scripts:**
- `scripts/train-CNN-autoe-final.py`
- `scripts/train-TopoTransformer-autoe-final.py`

**Purpose:** Encode boundary images for boundary-constrained generation

**Only Needed For:** Boundary-constrained variant

### Key Training Parameters

```python
# From scripts/trainval_main_unconstrained.py

diffusion_steps = 1000        # Diffusion process timesteps
lr = 1e-4                     # Learning rate (use 1e-5 for fine-tuning)
weight_decay = 0              # No weight decay
total_steps = 1000000         # Total training steps
batch_size = 256              # Training batch size
batch_size_val = 3000         # Validation batch size
device = 'cuda:0'             # GPU device
interval = 250000             # Checkpoint save interval
```

### Dataset Paths

All training scripts reference these hardcoded paths (relative from `scripts/` directory):

```python
# Training data
'../datasets/rplang-v3-withsemantics/train'

# Validation data
'../datasets/rplang-v3-withsemantics/val'

# Test data
'../datasets/rplang-v3-withsemantics/test'
```

To use different paths, modify the dataset class in `datasets/rplang_edge_semantics_simplified_55_100.py`.

---

## Monitoring Training Progress

### Checkpoints

Checkpoints are saved every 250,000 steps:

```
outputs/structure-1/
├── model0250000.pt
├── optim0250000.pt
├── model0500000.pt
├── optim0500000.pt
├── model0750000.pt
├── optim0750000.pt
├── model1000000.pt
└── optim1000000.pt
```

### Loss Curve

Training loss is saved to `loss_curve.npy`:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load loss curve
loss_curve = np.load('outputs/structure-1/loss_curve.npy')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(loss_curve[:, 0], label='Total Loss')
plt.plot(loss_curve[:, 1], label='Corner Loss')
plt.plot(loss_curve[:, 2], label='Alignment Loss')
plt.xlabel('Step')
plt.ylabel('Loss (×100000)')
plt.legend()
plt.savefig('loss_curve.png')
```

### Validation Metrics

Validation runs every 250,000 steps and computes FID/KID:

```python
# From scripts/trainval_main_unconstrained.py:707-710
current_Fid = fid(gt_dir_val, output_dir_val, fid_batch_size=128, fid_device=device)
current_Kid = kid(gt_dir_val, output_dir_val, kid_batch_size=128, kid_device=device)
print('step: ', step, 'FID: ', current_Fid, 'KID: ', current_Kid)
```

Metrics are saved to `val_metrics.npy`:

```python
# Load validation metrics
val_metrics = np.load('outputs/structure-1/val_metrics.npy')

# val_metrics columns: [step, FID, KID]
print("Step | FID   | KID")
for row in val_metrics:
    print(f"{int(row[0])} | {row[1]:.3f} | {row[2]:.6f}")
```

**Good Performance Indicators:**
- **FID**: Lower is better. Good models: < 40. Excellent: < 30.
- **KID**: Lower is better. Good models: < 0.02. Excellent: < 0.01.

### Generated Samples

Validation samples are rendered every 250,000 steps:

```
outputs/structure-1/
├── val_0250000/
│   ├── val_pred_0.png
│   ├── val_pred_1.png
│   └── ...
├── val_0500000/
│   └── ...
└── val_gt/
    ├── val_gt_0.png
    └── ...
```

Visually inspect these to ensure:
- Floor plans look realistic
- Room types are correct
- Walls connect properly
- No weird artifacts or distortions

---

## Best Practices

### Data Preparation

1. **Always Augment**: Use `augment_floor_plans.py --full` to get 8× data
2. **Balance Datasets**: If new data is tiny (< 1k samples), consider augmenting more
3. **Validate Data**: Run `raster_to_graph_converter.py --validate` to catch errors
4. **Split Properly**: Maintain ~90% train, 5% val, 5% test ratio

### Training Strategy

1. **Start with Combined Data**: Approach 1 is simpler and more reliable
2. **Use Fine-Tuning for Small Additions**: Only use Approach 2 if new data < 30% of old
3. **Lower LR for Fine-Tuning**: Use `lr = 1e-5` instead of `1e-4`
4. **Monitor Early**: Check first checkpoint (250k steps) before committing to full training
5. **Train All Three Stages**: Don't skip Stage 2 or 3

### Memory Management

1. **Reduce Batch Size**: If OOM error, decrease `batch_size` from 256 to 128 or 64
2. **Clear Cache**: Add `torch.cuda.empty_cache()` if needed
3. **Use Mixed Precision**: Consider AMP (Automatic Mixed Precision) for faster training

### Avoiding Common Pitfalls

1. **Don't Mix Corner Counts**: Ensure all data uses same corner padding (53 or 100)
2. **Don't Skip Validation Set**: Always keep ~5% data for validation
3. **Don't Stop Too Early**: 1M steps is needed for good convergence
4. **Don't Use High LR for Fine-Tuning**: Catastrophic forgetting is real
5. **Don't Forget Stage 2**: You need both Stage 1 and Stage 2 for generation

### Disk Space Requirements

Estimate storage needs before training:

```
Checkpoints (every 250k steps):
- 1M steps = 4 checkpoints × 2 files × ~500 MB = 4 GB

Loss curves and metrics:
- ~100 MB

Validation renders (every 250k):
- 4 validations × 3000 images × 50 KB = 600 MB

Total per training run: ~5 GB
```

### GPU Requirements

- **Minimum**: NVIDIA GPU with 12 GB VRAM (RTX 3080, V100)
- **Recommended**: 16-24 GB VRAM (RTX 4090, A100)
- **For 100 Corners**: 24+ GB VRAM (edges increase 4×)

---

## Complete Example Workflow

### Scenario: Adding 10,000 New Floor Plans to Existing 65,763

#### Option A: Combined Training (Recommended)

```bash
# 1. Prepare new data
python json_to_npy_floodfill.py \
    --input_dir /data/new_bim_json \
    --output_dir /data/new_raster \
    --resolution 10.0

python raster_to_graph_converter.py \
    --input_dir /data/new_raster \
    --output_dir /data/new_graph \
    --image_size 2575 \
    --validate

python augment_floor_plans.py \
    --input_dir /data/new_graph \
    --output_dir /data/new_augmented \
    --full

# Result: 10,000 × 8 = 80,000 new files

# 2. Split into train/val/test (90/5/5)
python scripts/split_dataset.py \
    --input_dir /data/new_augmented \
    --train_ratio 0.9 \
    --val_ratio 0.05 \
    --test_ratio 0.05

# Result: 72,000 train + 4,000 val + 4,000 test

# 3. Combine with old data
cp /data/new_augmented/train/*.npy ../datasets/rplang-v3-withsemantics/train/
cp /data/new_augmented/val/*.npy ../datasets/rplang-v3-withsemantics/val/
cp /data/new_augmented/test/*.npy ../datasets/rplang-v3-withsemantics/test/

# New totals: 137,763 train + 7,000 val + 7,000 test

# 4. Train from scratch
cd scripts
python trainval_main_unconstrained.py

# Wait ~5 days for 1M steps

# 5. Train Stage 2
# (After Stage 1 completes)
python trainval_main_edge_unconstrained.py

# 6. Evaluate
python test_main.py
```

#### Option B: Fine-Tuning (Faster, if you have checkpoints)

```bash
# 1-3. Same as Option A (prepare and combine data)

# 4. Modify trainval_main_unconstrained.py to load checkpoint
# (See "Approach 2" section above)

# 5. Fine-tune for 200k steps
cd scripts
python trainval_main_unconstrained.py

# Wait ~1 day for 200k steps

# 6. Train Stage 2 with fine-tuned Stage 1
python trainval_main_edge_unconstrained.py

# 7. Evaluate
python test_main.py
```

---

## Summary

**Can you retrain with previous data?** ✅ **Absolutely yes!**

**Best approach for most cases:**
1. Combine old + new data in same directory
2. Train from scratch (1M steps)
3. Simpler, more reliable, no forgetting

**Alternative for small additions:**
1. Load checkpoint from previous training
2. Fine-tune with lower learning rate (1e-5)
3. Train for 200k-500k steps
4. Faster but requires careful tuning

**Key insight:** GSDiff's dataset loader automatically uses all files in the training directory. You don't need any special code - just put all your NPY files together and start training!

---

## Related Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training documentation
- **[PRE_AUGMENTATION_WORKFLOW.md](PRE_AUGMENTATION_WORKFLOW.md)** - Data preparation pipeline
- **[DATA_AUGMENTATION_GUIDE.md](DATA_AUGMENTATION_GUIDE.md)** - Augmentation technical reference
- **[README_DOCS.md](README_DOCS.md)** - Master index of all documentation

---

## Questions?

If you encounter issues:

1. Check validation metrics - are FID/KID improving?
2. Inspect generated samples - do they look reasonable?
3. Review loss curve - is it decreasing?
4. Verify data format - use `raster_to_graph_converter.py --validate`
5. Check GPU memory - reduce batch size if OOM

For more details on training pipeline, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).
