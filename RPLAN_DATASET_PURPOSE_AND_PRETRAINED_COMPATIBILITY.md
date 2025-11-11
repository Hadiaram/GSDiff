# GSDiff: RPLAN Dataset Purpose & Pre-trained Model Compatibility

**Date:** 2025-11-10
**Focus:** Understanding what the 80k RPLAN dataset does and how changing it affects pre-trained models

---

## Executive Summary

**TL;DR:**
- **RPLAN's Purpose:** Training data to teach the model what floor plans look like
- **Pre-trained Weights:** Already learned from RPLAN, don't need it anymore
- **Changing Dataset:** Pre-trained models work WITHOUT any dataset, but may not match your custom data's style

---

## Part 1: What Is The 80,788 RPLAN Dataset Meant To Do?

### 1.1 The Core Purpose

The RPLAN dataset serves **ONE PRIMARY PURPOSE**:

> **To teach the diffusion model what valid residential floor plans look like**

Think of it like teaching an art student:
- You show them 80,000 examples of good paintings
- They learn patterns, styles, proportions, compositions
- Later, they can create new paintings without looking at the examples

**RPLAN = The 80,000 "example paintings" for floor plan generation**

### 1.2 What The Model Learns From RPLAN

During training on RPLAN, the model learns:

| Learned Knowledge | Examples from RPLAN |
|-------------------|---------------------|
| **Typical floor plan sizes** | Usually 10-50 corners, rarely more than 53 |
| **Common room types** | Living rooms, bedrooms, kitchens, bathrooms |
| **Typical proportions** | Bedrooms are ~10-20 sq meters, living rooms ~20-40 sq meters |
| **Layout conventions** | Bathrooms near bedrooms, kitchen near dining |
| **Structural patterns** | Rectangular rooms, orthogonal walls, T/L/X junctions |
| **Semantic relationships** | How room types connect (kitchen-dining, bedroom-bathroom) |
| **Valid geometries** | No overlapping rooms, proper wall connectivity |

**After training:** All this knowledge is **stored in the model weights** (~500MB checkpoint file)

### 1.3 The Training Process (Simplified)

```
Step 1: RPLAN Dataset (80,788 floor plans)
         ↓
      [Preprocessing]
         ↓
Step 2: Create .npy files (71,763 valid samples)
        - Extract corners, edges, room labels
        - Normalize coordinates
        - Pre-compute CNN features
         ↓
      [Training Loop - 1 Million Steps]
         ↓
Step 3: Model learns patterns
        - What makes a floor plan valid?
        - What are typical corner configurations?
        - How do room types relate spatially?
         ↓
      [Save Checkpoint]
         ↓
Step 4: Pre-trained Weights (outputs/model.pt)
        - Contains all learned knowledge
        - File size: ~500MB
        - RPLAN dataset no longer needed!
```

**Critical Insight:** Once training is complete, the model has "memorized" the patterns. You can DELETE the entire 80k RPLAN dataset and the model still works perfectly.

### 1.4 RPLAN's Role: Training Only

```python
# During Training (RPLAN REQUIRED):
for step in range(1_000_000):
    # Load batch from RPLAN dataset
    batch = dataloader.get_next_batch()  # Loads .npy files
    corners = batch['corners']
    semantics = batch['semantics']

    # Add noise (diffusion forward process)
    noisy_corners = add_noise(corners, timestep=t)

    # Train model to denoise
    predicted_noise = model(noisy_corners)
    loss = compute_loss(predicted_noise, actual_noise)

    # Update model weights (learning happens here!)
    optimizer.step()

# After Training (RPLAN NOT NEEDED):
torch.save(model.state_dict(), 'pretrained_weights.pt')
# Model now "knows" floor plans, RPLAN can be deleted
```

### 1.5 Analogy: Recipe Book vs Chef's Knowledge

| RPLAN Dataset | Pre-trained Weights |
|---------------|---------------------|
| **Recipe book** with 80,000 recipes | **Chef's brain** after studying all recipes |
| Takes up space (500GB with features) | Compact (500MB file) |
| Needed during learning | Not needed after learning |
| Can be discarded after training | Contains all learned patterns |

**You wouldn't carry 80,000 recipe books when cooking - just use the knowledge you learned from them!**

---

## Part 2: What Happens When You Change The Dataset With Pre-trained Weights?

### 2.1 Three Scenarios Explained

#### **Scenario 1: Using Pre-trained Weights WITHOUT Any Dataset**

**What you do:**
```python
# Load pre-trained model (trained on RPLAN)
model = load_pretrained_weights('rplan_pretrained.pt')

# Generate from pure random noise (NO dataset)
noise = torch.randn(batch_size, 53, 10)
feat_16 = torch.zeros(batch_size, 1024, 16, 16)  # No boundary constraints

# Generate floor plans
for t in range(999, -1, -1):
    noise = model(noise, mask, t, feat_16)

generated_floor_plans = noise
```

**What happens:**
- ✅ **Works perfectly!**
- Generates floor plans similar to RPLAN style
- No dataset files accessed during generation
- Model uses learned knowledge from training

**Why it works:**
- Model has internalized RPLAN patterns
- Random noise → learned distribution → RPLAN-style floor plans

---

#### **Scenario 2: Using Pre-trained Weights WITH Different Dataset**

**What you do:**
```python
# Load RPLAN pre-trained model
model = load_pretrained_weights('rplan_pretrained.pt')

# Load YOUR custom dataset (commercial buildings, offices, etc.)
custom_dataset = CustomFloorPlanDataset('my_office_buildings/')

# Try to use custom data with pretrained model
for sample in custom_dataset:
    # Your data: (100 corners, 5 semantic types, 512-dim features)
    corners = sample['corners']  # Shape: (100, 2) ❌ Model expects (53, 2)
    semantics = sample['semantics']  # Shape: (100, 5) ❌ Model expects (53, 7)
    features = sample['cnn_features']  # Shape: (512, 32, 32) ❌ Model expects (1024, 16, 16)

    output = model(corners, semantics, features)  # ❌ DIMENSION MISMATCH ERROR
```

**What happens:**
- ❌ **Crashes with dimension mismatch errors**
- Model expects specific tensor shapes from RPLAN
- Your custom data has different shapes

**Compatibility Matrix:**

| Your Data Property | RPLAN Model Expectation | Compatible? |
|--------------------|------------------------|-------------|
| Same format (53 corners, 7 semantics, 1024 features) | ✅ YES | ✅ Works |
| Different corner count (100 instead of 53) | ❌ NO | ❌ Crashes |
| Different semantics (5 types instead of 7) | ❌ NO | ❌ Crashes |
| Different CNN features (512-dim instead of 1024) | ❌ NO | ❌ Crashes |
| Same format but different style (e.g., office buildings) | ✅ YES | ⚠️ Works but generates RPLAN-style |

---

#### **Scenario 3: What If Data Format Matches But Content Differs?**

**Example:** Your custom dataset has the exact same format as RPLAN, but contains **office buildings** instead of residential.

**What you do:**
```python
# Load RPLAN pre-trained model
model = load_pretrained_weights('rplan_pretrained.pt')

# Your data: office buildings in RPLAN format
# - 53 corners max ✅
# - 7 semantic types ✅
# - 1024-dim CNN features ✅
custom_office_data = OfficeFloorPlanDataset()  # Format matches!

# Generate using pretrained model
generated_plans = model.generate()
```

**What happens:**
- ✅ **Runs without errors** (format matches)
- ⚠️ **But generates RESIDENTIAL floor plans, not offices**

**Why:**
- Model learned residential patterns from RPLAN
- Pre-trained weights encode "residential knowledge"
- Even if you provide office data, model doesn't use it during generation
- Generation is from learned distribution, not from input data

**Visual Example:**

```
Your Office Dataset:
┌──────────────────────────────┐
│  Open floor   │   Meeting    │
│  plan office  │   rooms      │
│               │              │
│  ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─  │
│  Cubicles     │  Server room │
└──────────────────────────────┘

RPLAN Pre-trained Model Generates:
┌──────────────────────────────┐
│  Living     │   Bedroom  1   │
│  Room       │                │
│             │                │
│  ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─  │
│  Kitchen    │   Bathroom     │
└──────────────────────────────┘
```

**Explanation:** Pre-trained weights are "locked in" to RPLAN's residential style.

---

### 2.2 Summary Table: Dataset Changes vs Pre-trained Weights

| Scenario | Dataset Change | Pre-trained Model Behavior | Result |
|----------|---------------|---------------------------|--------|
| **A** | No dataset (pure generation) | Uses RPLAN learned distribution | ✅ Generates RPLAN-style residential plans |
| **B** | Different format (100 corners, 5 semantics) | Dimension mismatch error | ❌ Crashes |
| **C** | Same format, different content (offices) | Ignores new data, uses RPLAN knowledge | ⚠️ Still generates residential plans |
| **D** | Same format, same content (more RPLAN data) | Doesn't retrain, uses existing weights | ✅ Generates same RPLAN-style plans |

---

## Part 3: When Does Changing The Dataset Matter?

### 3.1 Changing Dataset Matters During: **TRAINING**

**If you want the model to learn NEW patterns:**

```python
# Option 1: Train from scratch on your data
model = initialize_new_model()
custom_dataset = YourCustomDataset()  # Office buildings, mansions, etc.

for epoch in range(many_epochs):
    for batch in custom_dataset:
        loss = train_step(model, batch)
        optimizer.step()

# Result: Model learns YOUR data's patterns
torch.save(model.state_dict(), 'custom_pretrained.pt')
```

**What happens:**
- Model learns patterns from YOUR dataset
- If office buildings → generates office-style layouts
- If mansions → generates mansion-style layouts
- Training takes 2-4 weeks on 80GB GPU

---

### 3.2 Changing Dataset Does NOT Matter During: **INFERENCE**

**If you're using pre-trained weights for generation:**

```python
# Load pre-trained model
model = load_pretrained('rplan_pretrained.pt')

# Generate (no dataset needed!)
generated = model.generate_from_noise()

# Changing your local dataset files has ZERO effect
# Model only uses weights learned during training
```

**What happens:**
- Dataset files can be deleted, moved, or replaced
- Model doesn't read dataset during generation
- Output depends only on pre-trained weights

---

## Part 4: Practical Examples & Use Cases

### 4.1 Use Case 1: "I want to generate floor plans without downloading RPLAN"

**Solution:** Use pre-trained weights, no dataset needed

```bash
# Download pre-trained weights only (~500MB)
wget <google_drive_link>/rplan_pretrained.pt

# Generate floor plans
python scripts/test_main.py --model_path rplan_pretrained.pt

# Output: 1000 RPLAN-style floor plans
# Dataset required: NONE
```

**Disk space needed:** 500MB (weights only)
**Time:** 5 minutes
**Result:** Residential floor plans in RPLAN style

---

### 4.2 Use Case 2: "I want to see if the model works on my office building dataset"

**Scenario:** You have 100 office building floor plans

**Attempt 1: Use RPLAN pre-trained weights**
```python
model = load_pretrained('rplan_pretrained.pt')
generated = model.generate()
```

**Result:**
- Generates residential plans (not offices)
- Pre-trained on RPLAN = residential knowledge only

**Solution:** Must retrain on office data

```bash
# 1. Preprocess your 100 office plans → .npy format
python preprocess_custom_data.py --input office_buildings/

# 2. Problem: Only 100 samples (need 50,000+ for good results)
# Need to collect more data or use data augmentation

# 3. Train from scratch (2-4 weeks)
python scripts/trainval_main.py --dataset custom_office/
```

**Disk space needed:** 50GB+ (dataset + features)
**Time:** 2-4 weeks training
**Result:** Office-style floor plans

---

### 4.3 Use Case 3: "I have 80k mansion floor plans, will RPLAN weights work?"

**Your data:** 80,000 luxury mansion floor plans

**Question:** Can I use RPLAN pre-trained weights?

**Answer:** Technically yes (if format matches), but results won't be mansions

**Scenario A: Use RPLAN weights directly**
```python
model = load_pretrained('rplan_pretrained.pt')
generated = model.generate()
```
- ✅ Runs without errors
- ⚠️ Generates small residential homes (RPLAN style)
- ❌ Does NOT generate mansions

**Scenario B: Train on your mansion data**
```python
model = initialize_new_model()
mansion_dataset = MansionDataset('my_80k_mansions/')

train(model, mansion_dataset, steps=1_000_000)
torch.save(model.state_dict(), 'mansion_pretrained.pt')

# Now generate
generated = model.generate()
```
- ✅ Generates luxury mansion layouts
- Time: 2-4 weeks
- Result: Model learns mansion patterns (large rooms, multiple wings, etc.)

---

### 4.4 Use Case 4: "I want to fine-tune RPLAN weights on my 5k apartment layouts"

**Your data:** 5,000 apartment floor plans (similar to RPLAN but different style)

**Solution:** Fine-tuning (best of both worlds)

```python
# Start with RPLAN knowledge
model = load_pretrained('rplan_pretrained.pt')

# Continue training on your data with lower learning rate
apartment_dataset = ApartmentDataset('my_5k_apartments/')

for epoch in range(epochs):
    for batch in apartment_dataset:
        loss = train_step(model, batch)
        optimizer.step()  # lr = 1e-5 (much lower than from-scratch)

torch.save(model.state_dict(), 'rplan_plus_apartments.pt')
```

**Result:**
- Model keeps RPLAN's general floor plan knowledge
- Learns your apartment-specific style
- Requires less data (5k vs 60k)
- Faster training (3-7 days vs 2-4 weeks)

---

## Part 5: Key Technical Details

### 5.1 What's Actually Stored In Pre-trained Weights?

**File:** `outputs/structure-81-106-3/model.pt` (~500MB)

**Contains:**
```python
{
    'model_state_dict': {
        # 24 Transformer layers
        'transformer.0.self_attn.weight': tensor(...),
        'transformer.0.ffn.weight': tensor(...),
        ...
        'transformer.23.self_attn.weight': tensor(...),

        # Embedding layers
        'corner_embedding.weight': tensor(...),
        'semantic_embedding.weight': tensor(...),

        # Output heads
        'output_head1.weight': tensor(...),  # Coordinate prediction
        'output_head2.weight': tensor(...),  # Semantic prediction
    },
    'optimizer_state_dict': {...},  # Optional
    'epoch': 1000000,
    'loss': 0.0023
}
```

**What these weights encode:**
- Learned attention patterns (which corners should attend to each other)
- Learned semantic relationships (kitchen near dining)
- Learned geometry patterns (rectangular rooms, orthogonal walls)
- Learned size distributions (typical room sizes)

**What these weights DON'T contain:**
- ❌ No specific RPLAN floor plans (not memorization)
- ❌ No dataset file paths
- ❌ No raw image data
- ✅ Only learned statistical patterns

### 5.2 Why Pre-trained Models Don't Adapt To New Data Automatically

**Key Concept:** Pre-trained weights are **frozen** during inference

```python
# Inference mode (no learning)
model.eval()  # Sets model to evaluation mode
for param in model.parameters():
    param.requires_grad = False  # Freeze all weights

# Generate
with torch.no_grad():  # No gradient computation
    output = model.generate()

# Weights never change during generation!
```

**Even if you load new data:**
```python
new_dataset = OfficeDataset()
for sample in new_dataset:
    output = model(sample)  # Weights don't update
    # Model uses RPLAN patterns regardless of input
```

**To learn from new data, you MUST retrain:**
```python
model.train()  # Enable training mode
for param in model.parameters():
    param.requires_grad = True  # Unfreeze weights

for sample in new_dataset:
    output = model(sample)
    loss = compute_loss(output, target)
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights (learning happens!)
```

---

## Part 6: Decision Matrix

### "Should I use pre-trained RPLAN weights or train my own?"

| Your Situation | Recommendation | Reason |
|----------------|----------------|--------|
| **Want to generate residential floor plans** | ✅ Use RPLAN pre-trained | Exact match, no training needed |
| **Want to understand GSDiff without training** | ✅ Use RPLAN pre-trained | Quick start, 5 minutes |
| **Have 100-1,000 custom floor plans** | ⚠️ Fine-tune RPLAN weights | Too little data for from-scratch |
| **Have 5,000-20,000 custom floor plans** | ✅ Fine-tune RPLAN weights | Leverage RPLAN + adapt to your style |
| **Have 50,000+ custom floor plans** | ✅ Train from scratch | Enough data to learn your patterns |
| **Custom data has different dimensions** | ✅ Train from scratch (modify architecture) | RPLAN weights won't fit |
| **Want office/commercial layouts** | ✅ Train from scratch | RPLAN is residential-only |
| **Want mansions/villas** | ⚠️ Fine-tune or train from scratch | Similar enough to fine-tune |
| **Want to test on RPLAN test set** | ✅ Use RPLAN pre-trained | Direct evaluation |

---

## Part 7: Common Misconceptions

### ❌ Misconception 1: "I need RPLAN to run the model"

**Reality:** You only need pre-trained weights (~500MB). RPLAN (80k images + 500GB features) not required for inference.

---

### ❌ Misconception 2: "If I load my office dataset, the model will generate offices"

**Reality:** Pre-trained models use learned weights, not input data. RPLAN weights → residential plans, regardless of what dataset you load.

---

### ❌ Misconception 3: "Pre-trained weights work with any data format"

**Reality:** Weights expect specific tensor dimensions:
- 53 corners max
- 7 semantic dimensions
- 1024-dim CNN features

Different dimensions → crashes

---

### ❌ Misconception 4: "I can update pre-trained weights by loading new data"

**Reality:** Loading data doesn't update weights. You must explicitly retrain/fine-tune with backpropagation.

---

### ❌ Misconception 5: "RPLAN dataset affects generation quality"

**Reality:** After training, RPLAN is irrelevant. Only pre-trained weights matter. You can delete RPLAN entirely and generation works identically.

---

## Part 8: Practical Workflow Recommendations

### Workflow 1: Quick Exploration (5 minutes)

**Goal:** See what GSDiff can do

```bash
# Download pre-trained weights only
python download_weights.py

# Generate 100 floor plans
python scripts/test_main.py --num_samples 100

# Output: residential floor plans in RPLAN style
```

**Dataset needed:** None
**Disk:** 500MB
**Time:** 5 minutes

---

### Workflow 2: Custom Data - Small Dataset (3-7 days)

**Goal:** Generate floor plans in your style (you have 5k-20k samples)

```bash
# 1. Preprocess your data → .npy format
python preprocess_custom_data.py --input my_data/

# 2. Pre-compute CNN features
python scripts/prerunningCNN.py --dataset custom

# 3. Load RPLAN weights and fine-tune
python scripts/trainval_main.py \
    --model_path rplan_pretrained.pt \
    --dataset custom \
    --lr 1e-5 \
    --steps 100000

# Output: custom_pretrained.pt
```

**Dataset needed:** 5k-20k samples in GSDiff format
**Disk:** 50GB
**Time:** 3-7 days

---

### Workflow 3: Custom Data - Large Dataset (2-4 weeks)

**Goal:** Train from scratch on your domain (you have 50k+ samples)

```bash
# 1. Preprocess 50k+ floor plans
python preprocess_custom_data.py --input my_large_dataset/

# 2. Pre-compute CNN features
python scripts/prerunningCNN.py --dataset custom

# 3. Train from scratch
python scripts/trainval_main.py \
    --dataset custom \
    --lr 1e-4 \
    --steps 1000000

# Output: fully custom model
```

**Dataset needed:** 50k+ samples
**Disk:** 500GB+
**Time:** 2-4 weeks
**GPU:** 80GB VRAM

---

## Part 9: FAQ

### Q1: "Can I mix RPLAN data with my custom data during training?"

**A:** Yes! Concatenate datasets:

```python
rplan_data = RPlanDataset()
custom_data = CustomDataset()
combined = ConcatDataset([rplan_data, custom_data])

train(model, combined)
```

**Result:** Model learns both RPLAN and your patterns

---

### Q2: "Will pre-trained RPLAN weights work better than random initialization on my custom data?"

**A:** Usually yes, if your data is similar enough (residential floor plans). Use fine-tuning.

**Exception:** If your data is very different (circular buildings, multi-story, etc.), random initialization might be better.

---

### Q3: "Can I generate floor plans that don't exist in RPLAN?"

**A:** Yes! The model learns patterns, not specific floor plans. It can generate novel layouts in RPLAN style.

---

### Q4: "Do I need to understand RPLAN format to use pre-trained models?"

**A:** No! For pure generation, just load weights and run. No dataset knowledge needed.

For custom data training, yes, you need to match RPLAN format.

---

### Q5: "What if my floor plans have 100 corners instead of 53?"

**A:** Two options:
1. Modify architecture to accept 100 corners, train from scratch
2. Simplify your floor plans to ≤53 corners, use RPLAN weights

---

## Part 10: Summary & Key Takeaways

### The 80k RPLAN Dataset Serves Two Purposes:

1. **Training:** Teaches the model what residential floor plans look like
   - Used once during training (1M steps, 2-4 weeks)
   - Creates learned weights (~500MB)
   - After training: can be deleted

2. **Format Reference:** Shows how to structure custom data
   - .npy file structure
   - Corner/edge/semantic encoding
   - Preprocessing pipeline

### Changing The Dataset With Pre-trained Weights:

| Change Type | Effect on Pre-trained Model |
|-------------|----------------------------|
| **No dataset (pure generation)** | ✅ Works perfectly |
| **Different format** | ❌ Crashes (dimension mismatch) |
| **Same format, different style** | ⚠️ Runs but generates RPLAN style |
| **New data for training** | ✅ Can fine-tune or retrain |

### Golden Rules:

1. **Pre-trained weights = Frozen knowledge from RPLAN training**
2. **Inference doesn't use dataset files**
3. **To learn new patterns → must retrain/fine-tune**
4. **Format compatibility ≠ style compatibility**
5. **RPLAN weights → residential plans, always**

---

## Appendix: Code Examples

### Example 1: Generate Without Any Dataset

```python
import torch
from gsdiff.heterhouse_81_106_3 import BoundHeterHouseModel

# Load pre-trained weights
model = BoundHeterHouseModel()
model.load_state_dict(torch.load('outputs/structure-81-106-3/model.pt'))
model.eval()
model.cuda()

# Generate from pure noise (NO dataset)
batch_size = 10
x = torch.randn(batch_size, 53, 10).cuda()
mask = torch.ones(batch_size, 53, 53).bool().cuda()
feat_16 = torch.zeros(batch_size, 1024, 16, 16).cuda()

# Reverse diffusion
with torch.no_grad():
    for t in range(999, -1, -1):
        t_batch = torch.tensor([t] * batch_size).cuda()
        output = model(x, mask, t_batch, feat_16)
        x = posterior_sample(x, output, t)

# x now contains 10 generated floor plans
# No dataset files accessed!
```

### Example 2: Check If Your Data Is Compatible

```python
import numpy as np

# Load your custom data
your_data = np.load('your_dataset/sample_0.npy', allow_pickle=True).item()

# Check dimensions
print("Your data shapes:")
print("Corners:", your_data['corner_list_np_normalized_padding_withsemantics'].shape)
print("Expected: (53, 16)")

print("Attention:", your_data['global_matrix_np_padding'].shape)
print("Expected: (53, 53)")

print("Mask:", your_data['padding_mask'].shape)
print("Expected: (53, 1)")

# If all match → compatible with RPLAN weights (format-wise)
# If different → need to modify architecture
```

### Example 3: Fine-tune Pre-trained Weights

```python
from torch.utils.data import DataLoader
from gsdiff.heterhouse_81_106_3 import BoundHeterHouseModel

# Load pre-trained weights
model = BoundHeterHouseModel()
model.load_state_dict(torch.load('outputs/structure-81-106-3/model.pt'))
model.train()  # Enable training mode
model.cuda()

# Load YOUR custom dataset
custom_dataset = CustomFloorPlanDataset('my_data/')
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Fine-tune with low learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Low LR!

for epoch in range(10):
    for batch in dataloader:
        corners, mask, feat = batch

        # Add noise (diffusion forward)
        t = torch.randint(0, 1000, (batch_size,))
        noisy = add_noise(corners, t)

        # Predict noise
        pred = model(noisy, mask, t, feat)

        # Compute loss
        loss = mse_loss(pred, actual_noise)

        # Update weights (fine-tuning happens!)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save fine-tuned weights
torch.save(model.state_dict(), 'finetuned_on_my_data.pt')
```

---

**End of Report**

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│         RPLAN DATASET PURPOSE & COMPATIBILITY               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RPLAN Purpose:                                             │
│  └─ Training corpus to teach model residential patterns    │
│                                                             │
│  Pre-trained Weights:                                       │
│  └─ Learned knowledge from RPLAN (500MB)                   │
│  └─ No dataset needed after training                       │
│                                                             │
│  Changing Dataset:                                          │
│  ├─ Inference: No effect (doesn't use dataset)             │
│  └─ Training: Learns new patterns (if retrained)           │
│                                                             │
│  Compatibility:                                             │
│  ├─ Format must match (53 corners, 7 semantics)            │
│  └─ Style mismatch OK but generates RPLAN style            │
│                                                             │
│  Recommendations:                                           │
│  ├─ Quick test: Use pre-trained, no dataset                │
│  ├─ 5k-20k custom: Fine-tune RPLAN weights                 │
│  └─ 50k+ custom: Train from scratch                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
