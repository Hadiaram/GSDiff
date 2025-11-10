# GSDiff RPLAN Dataset Usage - Quick Reference

## KEY ANSWERS

### 1. RPLAN Dataset Usage

| Question | Answer |
|----------|--------|
| **Is RPLAN required for training?** | YES - to create 71,763 preprocessed `.npy` files |
| **Is RPLAN required for inference?** | NO - only need pre-trained model weights |
| **Is RPLAN required for generation?** | NO - works with pure random noise |
| **Which code depends on RPLAN?** | Training scripts + data preprocessing scripts |
| **Which code doesn't depend on RPLAN?** | Inference/test scripts, model architecture |

### 2. Where RPLAN Is Used

```
RPLAN Usage Flow:
├─ Preprocessing (rplan-extract.py through rplan-process10.py)
│  └─ Transform 80,788 PNGs → 71,763 .npy files
│
├─ Feature Extraction (prerunningCNN.py)
│  └─ Pre-compute CNN features for entire dataset (~500GB)
│
└─ Training (trainval_main_*.py)
   ├─ Load .npy files from disk
   ├─ Load pre-computed CNN features
   └─ Train diffusion models (Stage 1 & 2)

NOT used in:
├─ Inference/Generation (test_*.py)
├─ Model definitions (gsdiff/*.py)
└─ Any post-training operations
```

### 3. Can You Train Without RPLAN?

**YES** - If you have equivalent formatted data:
- 50,000-60,000+ floor plan samples
- Extracted as .npy files with:
  - Corners: (53, 2) normalized to [-1, 1]
  - Semantics: (53, 14) or (53, 7)
  - Adjacency: (53, 53) binary
  - Masks: (53, 1) valid/padding
- Pre-computed CNN features: (1, 1024, 16, 16)

### 4. Can You Generate Without Dataset?

**YES - Completely** - Example:
```python
# Zero dataset needed:
node_model = load_pretrained_weights()
edge_model = load_pretrained_weights()

x = torch.randn(batch_size, 53, 10)  # Random initialization
feat_16 = torch.zeros(batch_size, 1024, 16, 16)  # Or use custom features

with torch.no_grad():
    for t in range(999, -1, -1):
        x = model(x, mask, t, feat_16)

# Output: Generated floor plans
```

### 5. Pre-trained Model Limitations

**RPLAN-trained models work with:**
- RPLAN test data (exact format match)
- Any data with 53 corners, 7-dim semantics, 1024-dim features
- No data (unconstrained generation)

**RPLAN-trained models DON'T work with:**
- Different corner counts (e.g., 100)
- Different semantic dimensions (e.g., 5)
- Different feature dimensions (e.g., 512)

---

## THREE WORKFLOWS

### Workflow 1: Use Pre-trained (5 minutes, 4GB GPU, 200MB disk)

```bash
# Download pre-trained weights
# Run generation with pure random noise
python generate_unconstrained.py

# Outputs: 3000 floor plans without any dataset
```

### Workflow 2: Train From Scratch (2-4 weeks, 80GB GPU, 500GB+ disk)

```bash
# 1. Preprocess your 60,000 floor plans → .npy files
python your_preprocessing_pipeline.py

# 2. Pre-compute CNN features
python scripts/prerunningCNN.py

# 3. Train models
python scripts/trainval_main_unconstrained.py
python scripts/trainval_simplified_edge_unconstrained.py

# Result: Models trained on YOUR data
```

### Workflow 3: Fine-tune Pre-trained (3-7 days, 40GB GPU, 50GB+ disk)

```bash
# 1. Prepare 10,000-20,000 of your floor plans
python your_preprocessing_pipeline.py

# 2. Pre-compute features
python scripts/prerunningCNN.py

# 3. Fine-tune from existing weights
python scripts/trainval_main_unconstrained.py  # with lower LR

# Result: RPLAN knowledge + your data
```

---

## HARDCODED ASSUMPTIONS NEEDING CHANGES

1. **53-corner maximum** (line: `rplang_edge_semantics_simplified_81.py:89`)
   - To change: Modify all .npy creation + model input size

2. **7-dim semantic vectors** (7 room types)
   - To change: Modify dataset loader + model embedding

3. **1024-dim CNN features** (16×16 spatial)
   - To change: Train custom CNN or modify cross-attention

4. **256×256 image resolution**
   - To change: Adjust normalization formula

---

## CRITICAL FILES FOR EACH SCENARIO

### For Inference Only (No Dataset)
- `scripts/test_main.py` - Load weights, generate
- `outputs/` - Model checkpoints (download)
- No dataset files needed!

### For Custom Data Training
- `datasets/rplan-extract.py` - Reference for preprocessing
- `datasets/rplang_edge_semantics_simplified_81.py` - Reference dataset loader
- `scripts/prerunningCNN.py` - Feature extraction
- `scripts/trainval_main_unconstrained.py` - Training

### For Understanding Data Format
- `DATA_GENERATION_PIPELINE.md` - Complete preprocessing pipeline
- `GSDIFF_COMPREHENSIVE_ANALYSIS.md` - Training/inference details
- `/home/user/GSDiff/datasets/*.py` - Dataset loaders showing format

---

## CUSTOM DATASET CHECKLIST

- [ ] Collect 50,000+ floor plan samples
- [ ] Extract graph structure (corners, edges, rooms)
- [ ] Assign semantic labels to rooms
- [ ] Create .npy files in GSDiff format
- [ ] Train CNN feature extractor (or use pre-trained)
- [ ] Pre-compute all CNN features
- [ ] Create custom dataset loader class
- [ ] Modify training script paths
- [ ] Run training (2-4 weeks)

---

## FAQ

**Q: Does RPLAN semantic information (room types) matter?**
A: Yes, during training (creates semantic labels). No, for generation (generated from learned distribution).

**Q: Can I fine-tune pre-trained models on RPLAN?**
A: Yes, but you'd need the preprocessed RPLAN data. Easier to just use pre-trained for RPLAN domain.

**Q: What if my floor plans have 100 corners?**
A: Modify padding to 100, change model input size, retrain from scratch or fine-tune.

**Q: Does CNN feature quality affect generation?**
A: Yes, for boundary-constrained generation. No, for unconstrained (you can use zeros).

**Q: Can I use RPLAN's CNN for my custom data?**
A: Yes! The pre-trained boundary CNN generalizes well. Just use it as-is for feature extraction.

---

## RESOURCE FILES

- **Comprehensive Analysis:** `/home/user/GSDiff/GSDIFF_COMPREHENSIVE_ANALYSIS.md` (24KB)
- **Data Pipeline Details:** `/home/user/GSDiff/DATA_GENERATION_PIPELINE.md` (47KB)  
- **Custom Dataset Guide:** `/home/user/GSDiff/CUSTOM_DATASET_GUIDE.md` (70KB)
- **This Summary:** `/home/user/GSDiff/RPLAN_USAGE_SUMMARY.md`

