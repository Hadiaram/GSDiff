# GSDiff Codebase Exploration - Complete Index

This document provides a comprehensive index of all findings from the thorough exploration of the GSDiff codebase regarding RPLAN dataset usage and custom dataset support.

## Generated Analysis Documents

### 1. RPLAN_USAGE_SUMMARY.md (This file)
**Purpose:** Quick reference answers to all key questions  
**Contents:**
- Key answers table
- Three workflows comparison
- Hardcoded assumptions
- FAQ section

### 2. CUSTOM_DATASET_GUIDE.md (70KB)
**Purpose:** Complete guide for using custom datasets  
**Contents:**
- RPLAN dataset usage detailed
- Training requirements
- Inference/generation without dataset
- Custom dataset requirements with examples
- Code snippets for preprocessing
- Three complete workflows with step-by-step instructions
- Troubleshooting section

### 3. GSDIFF_COMPREHENSIVE_ANALYSIS.md (24KB)
**Purpose:** Deep technical analysis of training and inference  
**Contents:**
- Training architecture (2-stage pipeline)
- Model behavior and loss functions
- Pre-trained models and checkpoints
- Data flow in training vs inference
- CNN feature pre-computation
- Model training vs pre-trained weights

### 4. DATA_GENERATION_PIPELINE.md (47KB)
**Purpose:** Complete documentation of data preprocessing  
**Contents:**
- RPLAN format specification
- 10-stage preprocessing pipeline
- Corner/edge detection algorithms
- Semantic label assignment
- Normalization and padding
- CNN feature extraction
- Final .npy file structure
- Data flow visualization

## Key Code Files Referenced

### Dataset/Data Loading
```
/home/user/GSDiff/datasets/
├── rplang_edge_semantics_simplified_81.py    [Primary dataset loader]
├── rplang_edge_semantics_simplified_*.py     [Variants for different models]
├── lifull.py                                 [Alternative dataset (LIFULL)]
├── rplan-extract.py through rplan-process10.py [Preprocessing pipeline]
└── path_utils.py                             [Data path utility]
```

### Training Scripts
```
/home/user/GSDiff/scripts/
├── trainval_main_unconstrained.py            [Stage 1 node training]
├── trainval_main_boun.py                     [Boundary-constrained training]
├── trainval_main_topo.py                     [Topology-constrained training]
├── trainval_simplified_edge_*.py             [Stage 2 edge training]
├── train-CNN-autoe-final.py                  [CNN feature extractor training]
└── prerunningCNN.py                          [CNN feature pre-computation]
```

### Inference/Test Scripts
```
/home/user/GSDiff/scripts/
├── test_main.py                              [Unconstrained generation]
├── test_boun.py                              [Boundary-constrained generation]
├── test_topo.py                              [Topology-constrained generation]
└── test-final-lifull1.py                     [LIFULL dataset testing]
```

### Model Definitions
```
/home/user/GSDiff/gsdiff/
├── heterhouse_81_106_3.py                    [Boundary-constrained model]
├── heterhouse_56_31.py                       [Unconstrained model]
├── house_nn2.py / heterhouse_56_11.py        [Edge generation models]
├── utils.py                                  [Training utilities]
└── utils_lifull.py                           [LIFULL-specific utilities]
```

---

## Key Findings Summary

### RPLAN Dataset Usage
- **Size:** 80,788 raw PNGs → 71,763 valid .npy files
- **Purpose:** Training data creation only
- **NOT needed for:** Inference, generation, model architecture
- **Location:** `/datasets/rplang-v3-withsemantics/`

### CNN Features
- **Purpose:** Guide corner placement via cross-attention
- **Status:** Pre-computed and frozen (cannot be retrained)
- **Size:** (1, 1024, 16, 16) per sample
- **Location:** `/datasets/prerunning_cnn_featuremaps/`
- **Can be:** Custom features, pre-trained CNN features, or zeros

### Data Format Requirements (For Training)
```
Each sample must have:
├─ corner_list_np_normalized_padding: (53, 2)     [Corners in [-1,1]]
├─ corner_list_np_normalized_padding_withsemantics: (53, 16) [Corners + 14-dim semantics]
├─ adjacency_matrix_np_padding: (53, 53)          [Edge matrix]
├─ global_matrix_np_padding: (53, 53)             [Attention mask]
├─ padding_mask: (53, 1)                          [Valid/padding indicator]
├─ edges: (2809, 1)                               [Flattened edges]
└─ CNN features: (1, 1024, 16, 16)                [Optional, for constrained]
```

### Three Workflows

**Workflow 1: Pre-trained (No Dataset)**
- Time: 5 min | Disk: 200MB | GPU: 4GB
- Download weights → Run generation
- Works immediately, no data preprocessing

**Workflow 2: Train From Scratch (Custom Dataset)**
- Time: 2-4 weeks | Disk: 500GB+ | GPU: 80GB
- Preprocess 60k samples → Feature extraction → Training
- Full control, best results for custom domain

**Workflow 3: Fine-tune Pre-trained**
- Time: 3-7 days | Disk: 50GB+ | GPU: 40GB
- Preprocess 10-20k samples → Feature extraction → Fine-tuning
- Fast adaptation, transfers RPLAN knowledge

---

## Critical Code Locations

### Hardcoded RPLAN Assumptions

1. **53-corner padding (RPLAN max)**
   - Location: `datasets/rplang_edge_semantics_simplified_81.py:89`
   - Change needed: Modify in .npy creation + model input layers

2. **14-dim semantic labels (RPLAN room types)**
   - Simplified to 7-dim during loading
   - Location: `datasets/rplang_edge_semantics_simplified_81.py:73-82`
   - Change needed: Modify semantic dimension across codebase

3. **256×256 image resolution**
   - Normalization formula: `(pixel - 128) / 128`
   - Location: DATA_GENERATION_PIPELINE.md section 6.1
   - Change needed: Adjust normalization if using different resolution

4. **1024-dim CNN features (16×16 spatial)**
   - From specific CNN architecture
   - Location: `scripts/prerunningCNN.py`, `gsdiff/boundary_*.py`
   - Change needed: Train custom CNN or modify cross-attention

### Training Loop
- Location: `scripts/trainval_main_unconstrained.py`
- Key steps:
  1. Load .npy files and CNN features
  2. Sample random timestep
  3. Add noise (forward diffusion)
  4. Model prediction
  5. Compute loss (noise + alignment)
  6. Backward + optimize

### Inference Loop
- Location: `scripts/test_main.py`
- Key steps:
  1. Load pre-trained weights
  2. Initialize from noise
  3. Reverse diffusion (1000 steps)
  4. Extract corners and edges
  5. Post-process and save

---

## Data Flow Diagrams

### Training Data Flow
```
.npy files (71,763)
    ↓
DataLoader (batch_size=256)
    ├─ Load corners, semantics, masks
    ├─ Load pre-computed CNN features
    └─ Concatenate padding indicator
        ↓
Model Forward Pass
    ├─ Input: corners (bs, 53, 10)
    ├─ Input: CNN features (bs, 1024, 16, 16)
    ├─ 24 Transformer layers
    ├─ Multi-head self-attention
    └─ Multi-head cross-attention with CNN
        ↓
Loss Computation
    ├─ Noise prediction loss (L2)
    └─ Geometric alignment loss
        ↓
Backward & Optimization
    ├─ Gradient clipping (norm=0.1)
    └─ Adam update
```

### Generation (Inference) Flow
```
Pre-trained weights + Random noise
    ↓
Stage 1: Node Generation (1000 diffusion steps)
    ├─ For t=999 to t=0:
    │  ├─ Add time embedding
    │  ├─ Self-attention (corner-to-corner)
    │  ├─ Cross-attention (corner-to-image)
    │  └─ Posterior sample
    └─ Output: Corners + semantics
        ↓
Stage 2: Edge Generation
    ├─ Input: Generated corners + semantics
    ├─ Self-attention (edge-to-edge)
    └─ Output: Edge logits
        ↓
Post-processing
    ├─ Denormalize coordinates
    ├─ Remove padding
    ├─ Extract room polygons
    └─ Render image
```

---

## Testing & Evaluation

### Evaluation Scripts
- `evalmetric-no-constrain-fid-kid.py` - FID/KID metrics
- `evalmetric-no-constrain-geometry-topological-metrics.py` - Geometric metrics
- `evalmetric-topoconstrain-ged-roomnumber.py` - Topology metrics
- `evalmetric-boun-constrain-fid-kid.py` - Boundary metrics

### Metrics Computed
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- GED (Graph Edit Distance)
- Room type classification accuracy
- Geometric alignment metrics

---

## Important Statistics

### RPLAN Dataset
- Raw images: 80,788 PNGs (256×256×4)
- After preprocessing: 71,763 valid files
- Train split: 65,763 samples
- Val split: 3,000 samples
- Test split: 3,000 samples
- Typical corners per floor plan: 10-50 (padded to 53)

### Training Configuration
- Diffusion steps: 1000
- Learning rate: 1e-4
- Weight decay: 1e-7
- Batch size: 256 (nodes) / 8 (edges)
- Total training steps: 1,000,000
- Estimated time: 2-4 weeks (8x A100 GPUs)

### Feature Extraction
- CNN feature map: 1024 channels, 16×16 spatial
- Disk space per sample: ~65KB (1024×16×16×4 bytes)
- Total space for 71,763 samples: ~470GB
- Extraction time: 1-2 weeks (1x GPU)

---

## Quick Navigation

**I want to...**
- Use pre-trained models → See `RPLAN_USAGE_SUMMARY.md` Workflow 1
- Train on custom data → See `CUSTOM_DATASET_GUIDE.md` Part 2 & Workflow 2
- Understand data format → See `DATA_GENERATION_PIPELINE.md` Section 11
- See training details → See `GSDIFF_COMPREHENSIVE_ANALYSIS.md` Part 1
- Understand inference → See `GSDIFF_COMPREHENSIVE_ANALYSIS.md` Part 4
- Fine-tune pre-trained → See `CUSTOM_DATASET_GUIDE.md` Workflow 3
- Handle 100 corners → See `CUSTOM_DATASET_GUIDE.md` Part 7 Troubleshooting
- Use custom room types → See `CUSTOM_DATASET_GUIDE.md` Part 4.4

---

## Files in This Repository (Documentation)

Generated during exploration:
```
/home/user/GSDiff/
├── RPLAN_USAGE_SUMMARY.md              [Quick reference - START HERE]
├── CUSTOM_DATASET_GUIDE.md             [Complete custom dataset guide]
├── GSDIFF_COMPREHENSIVE_ANALYSIS.md    [Technical deep dive]
├── DATA_GENERATION_PIPELINE.md         [Data preprocessing details]
├── EXPLORATION_INDEX.md                [This file]
├── README.md                            [Original repository README]
├── DATA_GENERATION_PIPELINE.md         [Original documentation]
└── GSDIFF_COMPREHENSIVE_ANALYSIS.md    [Original documentation]
```

---

**Last Updated:** November 10, 2025  
**Analysis Scope:** Complete GSDiff codebase exploration  
**Total Documentation Generated:** 160+ KB across 4 documents  

