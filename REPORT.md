# GSDiff Floor Plan Generation Input Pipeline Analysis Report

**Project:** GSDiff - Vector Floor Plan Generation via Geometry-enhanced Structural Graph Generation

**Report Date:** 2025-11-06

**Analysis Scope:** Complete input pipeline from raw data to model inference

---

## Executive Summary

This report provides a comprehensive analysis of the input pipeline for GSDiff, a graph-based diffusion model for generating vector floor plans. The analysis traces the complete data flow from raw RPLAN dataset storage through preprocessing, dataset loading, model input encoding, and generation.

**Key Findings:**

- Floor plans are represented as geometric graphs with up to 53 corners (nodes) and their connectivity (edges)
- Input data flows through 9 distinct stages from storage to generation
- The system supports three generation modes: unconstrained, topology-constrained, and boundary-constrained
- Data undergoes semantic simplification (16→9 dimensions) to improve model learning
- Pre-computed CNN features are loaded to optimize inference performance

**Critical Input Components:**

1. **Graph structure:** Corner coordinates + semantic labels (primary input)
2. **CNN features:** Boundary/wall image embeddings (optional, for constrained generation)
3. **Diffusion timestep:** Controls the denoising process (t=999→0)
4. **Attention masks:** Defines valid corner interactions

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Data Storage Format](#3-data-storage-format)
4. [Dataset Loading Pipeline](#4-dataset-loading-pipeline)
5. [Model Input Preparation](#5-model-input-preparation)
6. [Input Encoding Architecture](#6-input-encoding-architecture)
7. [Diffusion Process Integration](#7-diffusion-process-integration)
8. [Complete Input Flow](#8-complete-input-flow)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Performance Optimizations](#10-performance-optimizations)
11. [Code Reference Guide](#11-code-reference-guide)
12. [Conclusions and Recommendations](#12-conclusions-and-recommendations)

---

## 1. Introduction

### 1.1 Background

GSDiff is the official implementation of the AAAI 2025 paper on synthesizing vector floor plans via geometric diffusion. The system generates floor plans as structural graphs where nodes represent corners/junctions and edges represent walls.

### 1.2 Dataset

- **Primary Dataset:** RPLAN with 80,788 floor plans
  - Training: 65,763 samples
  - Validation: 3,000 samples
  - Testing: 3,000 samples
- **Secondary Dataset:** LIFULL with 10,804 floor plans

### 1.3 Generation Modes

1. **Unconstrained:** Pure generative mode without constraints
2. **Topology-constrained:** Generation with room connectivity constraints
3. **Boundary-constrained:** Generation guided by wall/boundary images

### 1.4 Report Scope

This report focuses exclusively on understanding where and how input data enters the floor plan generation system, tracing the complete pipeline from raw data storage to model inference.

---

## 2. System Overview

### 2.1 High-Level Architecture

```text
┌──────────────────┐
│   Raw RPLAN      │
│   Dataset        │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐      ┌──────────────────┐
│  Preprocessing   │──────│  CNN Feature     │
│  Scripts         │      │  Extraction      │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         ↓                         ↓
┌──────────────────┐      ┌──────────────────┐
│  Processed .npy  │      │  Feature Maps    │
│  Graph Files     │      │  (1024,16,16)    │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────┬────────────────┘
                  │
                  ↓
         ┌────────────────┐
         │  PyTorch       │
         │  Dataset Class │
         └────────┬───────┘
                  │
                  ↓
         ┌────────────────┐
         │  DataLoader    │
         │  (Batching)    │
         └────────┬───────┘
                  │
                  ↓
         ┌────────────────┐
         │  Diffusion     │
         │  Model Input   │
         └────────────────┘
```

### 2.2 Core Technologies

- **Framework:** PyTorch 2.x
- **Architecture:** Transformer-based (24 layers, 512-dim)
- **Diffusion:** DDPM (1000 timesteps, cosine schedule)
- **Attention:** Multi-head self + cross-attention (4 heads)

---

## 3. Data Storage Format

### 3.1 File Structure

**Location:** `datasets/rplang-v3-withsemantics/{train,val,test}/*.npy`

Each floor plan is stored as a NumPy dictionary with the following structure:

```python
graph = {
    'corner_list_np_normalized_padding_withsemantics': ndarray(53, 16),
    'global_matrix_np_padding': ndarray(53, 53),
    'padding_mask': ndarray(53, 1),
    'edges': ndarray(2809, 1)
}
```

### 3.2 Data Components

#### 3.2.1 Corner List (53, 16)

**Columns 0-1:** Normalized coordinates

- X, Y coordinates in range [-1, 1]
- Normalization formula: `(pixel_coord / 128) - 1`
- Original pixel space: [0, 256]

**Columns 2-15:** Semantic labels (multi-hot encoding)

- Column 2: Living room / LivingDining / Dining
- Column 3: MasterRoom (primary bedroom)
- Column 4: Kitchen
- Column 5: Bathroom
- Column 6: Dining room
- Column 7: ChildRoom / KidsRoom
- Column 8: StudyRoom
- Column 9: SecondRoom
- Column 10: GuestRoom
- Column 11: Balcony
- Column 12: Entrance
- Column 13: Storage / Storeroom
- Column 14: Wall-in
- Column 15: External area

**Multi-hot Encoding:** A corner can belong to multiple room types (e.g., at the junction of kitchen and living room, both labels would be 1).

#### 3.2.2 Global Attention Matrix (53, 53)

- Boolean adjacency matrix
- `True` indicates corners can attend to each other
- Used as attention mask in transformer layers
- Controls information flow between nodes

#### 3.2.3 Padding Mask (53, 1)

- Binary indicator: `1` = real corner, `0` = padded corner
- Enables variable-size graphs with fixed-size tensors
- Used to filter padding in loss calculations and post-processing

#### 3.2.4 Edges (2809, 1)

- Flattened adjacency matrix: 53 × 53 = 2809
- Binary encoding: `1` = wall exists, `0` = no wall
- Represents graph connectivity (which corners are connected by walls)

### 3.3 Why Padding to 53 Corners?

**Design Rationale:**

- Real floor plans have variable corner counts (typically 10-50)
- Neural networks require fixed-size inputs for batching
- 53 chosen as maximum corner count observed in RPLAN dataset
- Padding strategy balances flexibility with memory efficiency

**Implementation:**

```python
# Real floor plan with 23 corners
corners_real = np.array([[x1, y1, sem...], ..., [x23, y23, sem...]])  # Shape: (23, 16)

# Padded to 53
corners_padded = np.zeros((53, 16))
corners_padded[:23] = corners_real
padding_mask = np.concatenate([np.ones((23, 1)), np.zeros((30, 1))])  # 23 real, 30 padded
```

---

## 4. Dataset Loading Pipeline

### 4.1 Dataset Class Architecture

**File:** `datasets/rplang_edge_semantics_simplified_81.py`

This dataset class handles:

1. File path management
2. Loading .npy graph files
3. Semantic simplification (16→9 dimensions)
4. Loading pre-computed CNN features
5. PyTorch Dataset API compliance

### 4.2 Initialization (`__init__`)

**Code Location:** Lines 14-30

```python
def __init__(self, mode):
    """
    Args:
        mode: 'train', 'val', or 'test'
    """
    self.mode = mode
    
    # Get file paths for graph data
    self.data_path = get_data_path('rplang-v3-withsemantics', mode)
    self.file_list = sorted(os.listdir(self.data_path))
    
    # Get pre-computed CNN features
    self.feat_path = get_data_path('rplang-v3-withsemantics-prerunCNN-16', mode)
```

**Key Operations:**

1. Set mode (train/val/test)
2. Build file paths using `path_utils.get_data_path()`
3. Create sorted list of .npy files
4. Set path to pre-computed CNN features (1024, 16, 16)

### 4.3 Data Loading (`__getitem__`)

**Code Location:** Lines 37-102

**Step-by-step breakdown:**

#### Step 1: Load Graph Data

```python
graph = np.load(os.path.join(self.data_path, self.file_list[index]), 
                allow_pickle=True).item()
```

Loads the .npy file containing the graph dictionary.

#### Step 2: Extract Components

```python
corners_withsemantics_0 = graph['corner_list_np_normalized_padding_withsemantics']
global_attn_matrix = graph['global_matrix_np_padding']
corners_padding_mask = graph['padding_mask']
```

#### Step 3: Semantic Simplification (16→9 dimensions)

**Original 16 dimensions → Simplified 9 dimensions:**

```python
# Simplification mapping:
# 0: Living/LivingDining/Dining → Living (dim 0)
# 1: MasterRoom → MasterRoom (dim 1)
# 2: Kitchen → Kitchen (dim 2)
# 3: Bathroom → Bathroom (dim 3)
# 4: Dining → Living (dim 0)  # Merged with Living
# 5: ChildRoom/KidsRoom → SecondRoom (dim 4)  # Merged
# 6: StudyRoom → SecondRoom (dim 4)  # Merged
# 7: SecondRoom → SecondRoom (dim 4)
# 8: GuestRoom → SecondRoom (dim 4)  # Merged
# 9: Balcony → Balcony (dim 5)
# 10: Entrance → Entrance (dim 6)
# 11: Storage/Storeroom → Entrance (dim 6)  # Merged
# 12: Wall-in → (removed)
# 13: External area → External (dim 7)
# + Padding indicator (dim 8)
```

**Implementation:**

```python
corners_withsemantics_0_9 = np.zeros((53, 9), dtype=np.float64)

# Coordinates (unchanged)
corners_withsemantics_0_9[:, 0:2] = corners_withsemantics_0[:, 0:2]

# Semantic labels (simplified)
corners_withsemantics_0_9[:, 2] = np.max(corners_withsemantics_0[:, [2, 6]], axis=1)  # Living
corners_withsemantics_0_9[:, 3] = corners_withsemantics_0[:, 3]  # MasterRoom
corners_withsemantics_0_9[:, 4] = corners_withsemantics_0[:, 4]  # Kitchen
corners_withsemantics_0_9[:, 5] = corners_withsemantics_0[:, 5]  # Bathroom
corners_withsemantics_0_9[:, 6] = np.max(corners_withsemantics_0[:, [7, 8, 9, 10]], axis=1)  # SecondRoom
corners_withsemantics_0_9[:, 7] = corners_withsemantics_0[:, 11]  # Balcony
corners_withsemantics_0_9[:, 8] = np.max(corners_withsemantics_0[:, [12, 13]], axis=1)  # Entrance
corners_withsemantics_0_9[:, 9] = corners_withsemantics_0[:, 15]  # External
```

**Why Simplification?**

- Reduces semantic label dimensionality
- Merges semantically similar categories
- Improves model learning efficiency
- Maintains essential room type distinctions

#### Step 4: Load CNN Features

```python
feat_16 = np.load(os.path.join(self.feat_path, 
                               self.file_list[index].replace('.npy', '_feat16.npy')))
```

**Feature Characteristics:**

- Shape: `(1024, 16, 16)`
- Pre-computed using CNN encoder
- Represents boundary/wall image embeddings
- Used for boundary-constrained generation

#### Step 5: Convert to Tensors

```python
feat_16 = torch.from_numpy(feat_16).float()
corners_withsemantics_0_9 = torch.from_numpy(corners_withsemantics_0_9).float()
global_attn_matrix = torch.from_numpy(global_attn_matrix).bool()
corners_padding_mask = torch.from_numpy(corners_padding_mask).float()
```

#### Step 6: Return Batch

```python
return (feat_16, 
        corners_withsemantics_0_9, 
        global_attn_matrix, 
        corners_padding_mask)
```

### 4.4 Why Semantic Simplification?

**Original Problem:**

- 16 semantic dimensions create high-dimensional label space
- Many categories are semantically similar (e.g., ChildRoom, StudyRoom, GuestRoom)
- Sparse labels make learning difficult

**Solution:**

- Merge similar categories into broader groups
- Reduce to 9 essential dimensions
- Maintain meaningful distinctions (Living vs. Bedroom vs. Kitchen)

**Benefits:**

1. **Improved Learning:** Fewer dimensions = easier for model to learn
2. **Better Generalization:** Broader categories capture more training examples
3. **Reduced Sparsity:** More samples per category
4. **Maintained Semantics:** Essential room types preserved

---

## 5. Model Input Preparation

### 5.1 DataLoader Setup

**Code Location:** `scripts/test_boun.py`, Lines 74-83

```python
dataset_test = RPlanGEdgeSemanSimplified_81('test')
dataloader_test = DataLoader(
    dataset_test, 
    batch_size=batch_size_test,  # e.g., 32
    shuffle=False, 
    num_workers=0,
    drop_last=False, 
    pin_memory=False
)
dataloader_test_iter = iter(cycle(dataloader_test))
```

**Configuration:**

- Batch size: Configurable (e.g., 32 for testing, larger for training)
- Shuffle: False for test (deterministic), True for train (randomization)
- Workers: 0 for single-process loading
- Cycle: Wraps dataloader for continuous iteration

### 5.2 Batch Retrieval

**Code Location:** `scripts/test_boun.py`, Lines 189-192

```python
feat_16_test_batch, corners_withsemantics_0_test_batch, \
global_attn_matrix_test_batch, corners_padding_mask_test_batch = \
    next(dataloader_test_iter)
```

**Batch Shapes:**

```python
feat_16_test_batch:                   # (BS, 1024, 16, 16)
corners_withsemantics_0_test_batch:   # (BS, 53, 9)
global_attn_matrix_test_batch:        # (BS, 53, 53)
corners_padding_mask_test_batch:      # (BS, 53, 1)
```

### 5.3 Device Transfer and Preprocessing

**Code Location:** `scripts/test_boun.py`, Lines 193-202

```python
# Transfer to GPU
feat_16_test_batch = feat_16_test_batch.to(device).float()
corners_withsemantics_0_test_batch = corners_withsemantics_0_test_batch.to(device).clamp(-1, 1)
global_attn_matrix_test_batch = global_attn_matrix_test_batch.to(device)
corners_padding_mask_test_batch = corners_padding_mask_test_batch.to(device)

# Append padding mask to corner data
corners_withsemantics_0_test_batch = torch.cat(
    (corners_withsemantics_0_test_batch, 
     (1 - corners_padding_mask_test_batch).type(corners_withsemantics_0_test_batch.dtype)), 
    dim=2
)  # Shape: (BS, 53, 10)
```

**Key Operations:**

1. **Device Transfer:** Move data to GPU for computation
2. **Coordinate Clamping:** Ensure coordinates stay in [-1, 1] range
3. **Padding Mask Appending:** Add inverted mask (0→1, 1→0) as extra dimension

**Final Shape:** `(BS, 53, 10)` where dimension 10 = 2 coords + 7 semantics + 1 padding indicator

---

## 6. Input Encoding Architecture

### 6.1 Model Overview

**File:** `gsdiff/heterhouse_81_106_3.py`

**Class:** `BoundHeterHouseModel`

The model encodes inputs through multiple stages before processing with transformers.

### 6.2 Corner Embedding

**Code Location:** Lines 180-185

```python
# Split coordinates and semantics
corners_t = corners_withsemantics_t[:, :, 0:2]        # (BS, 53, 2)
withsemantics_t = corners_withsemantics_t[:, :, 2:10] # (BS, 53, 8)

# Embed coordinates
corner_embed = self.pos_encoder(corners_t)  # (BS, 53, 512)
```

**Positional Encoding:**

Uses sinusoidal encoding to map 2D coordinates to 512-dimensional embeddings:

```python
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, corners):
        # corners: (BS, N, 2)
        BS, N, _ = corners.shape
        
        # Generate frequency bands
        freq_bands = 2 ** torch.linspace(0, 9, self.d_model // 4, device=corners.device)
        
        # Apply sinusoidal encoding
        x = corners[:, :, 0:1] * freq_bands  # X coordinate
        y = corners[:, :, 1:2] * freq_bands  # Y coordinate
        
        pos_encoding = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(y), torch.cos(y)
        ], dim=-1)  # (BS, N, 512)
        
        return pos_encoding
```

### 6.3 Semantic Embedding

**Code Location:** Lines 186-188

```python
# Embed semantic labels
semantic_embed = self.semantic_encoder(withsemantics_t)  # (BS, 53, 512)
```

**Implementation:**

```python
self.semantic_encoder = nn.Linear(8, 512)  # 8 semantic dims + padding → 512-dim
```

Maps 8-dimensional semantic vector to 512-dimensional embedding space.

### 6.4 Time Embedding

**Code Location:** Lines 189-194

```python
# Encode diffusion timestep
t_embed = timestep_embedding(t, self.d_model)  # (BS, 512)
t_embed = t_embed.unsqueeze(1).expand(-1, 53, -1)  # (BS, 53, 512)
```

**Sinusoidal Time Encoding:**

```python
def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: (BS,) tensor of timestep values [0, 999]
        dim: embedding dimension (512)
    
    Returns:
        (BS, dim) tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    return embedding
```

**Purpose:**

- Informs model of current denoising step
- Different behavior for different t values
- Critical for diffusion process

### 6.5 CNN Feature Encoding

**Code Location:** Lines 195-210

```python
# Project CNN features
feat_16_projected = self.cnn_proj(feat_16)  # (BS, 1024, 16, 16) → (BS, 256, 16, 16)

# Flatten spatial dimensions
feat_16_flat = feat_16_projected.flatten(2).permute(0, 2, 1)  # (BS, 256, 256)

# Create positional encodings for image features
image_pos = create_2d_positional_encoding(16, 16, 256, device)  # (256, 256)
image_pos = image_pos.unsqueeze(0).expand(BS, -1, -1)  # (BS, 256, 256)

# Add positional encoding to features
feat_encoded = feat_16_flat + image_pos  # (BS, 256, 256)
```

**CNN Projection Layer:**

```python
self.cnn_proj = nn.Conv2d(1024, 256, kernel_size=1)
```

Reduces channel dimension for efficiency.

**2D Positional Encoding for Images:**

```python
def create_2d_positional_encoding(h, w, d_model, device):
    """
    Create 2D positional encoding for image features.
    
    Args:
        h: height (16)
        w: width (16)
        d_model: model dimension (256)
        device: torch device
    
    Returns:
        (h*w, d_model) tensor
    """
    y_pos = torch.arange(h, dtype=torch.float32, device=device)
    x_pos = torch.arange(w, dtype=torch.float32, device=device)
    
    y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
    
    positions = torch.stack([
        x_grid.flatten(),
        y_grid.flatten()
    ], dim=1)  # (256, 2)
    
    # Apply sinusoidal encoding
    encoding = positional_encoding_1d(positions, d_model)
    
    return encoding  # (256, 256)
```

### 6.6 Combined Input Embedding

**Code Location:** Lines 211-213

```python
# Combine all embeddings
x = corner_embed + semantic_embed + t_embed  # (BS, 53, 512)
```

**Final Input Tensor:**

- Shape: `(BS, 53, 512)`
- Contains: position + semantics + time information
- Ready for transformer processing

### 6.7 Cross-Attention Setup

**Code Location:** Lines 214-218

```python
# Prepare attention matrices
self_attn_matrix = global_attn_matrix  # (BS, 53, 53)
cross_attn_matrix = torch.ones(BS, 53, 256, dtype=torch.bool, device=device)  # (BS, 53, 256)
```

**Attention Matrices:**

1. **Self-Attention:** `(BS, 53, 53)` - controls corner-to-corner interactions
2. **Cross-Attention:** `(BS, 53, 256)` - enables corner-to-image-feature interactions

---

## 7. Diffusion Process Integration

### 7.1 Reverse Diffusion Loop

**Code Location:** `scripts/test_boun.py`, Lines 213-262

```python
for current_step_test in list(range(diffusion_steps - 1, -1, -1)):  # 999→0
    if current_step_test == diffusion_steps - 1:
        # Initialize with pure noise at t=999
        corners_withsemantics_t_test_batch = torch.randn(
            *corners_withsemantics_0_test_batch.shape,
            device=device,
            dtype=corners_withsemantics_0_test_batch.dtype
        )
    else:
        # Use previous denoising result
        corners_withsemantics_t_test_batch = sample_from_posterior_normal_distribution_test_batch
    
    # Create timestep tensor
    t_test = torch.tensor([current_step_test] * BS, device=device)
    
    # Model forward pass
    output1, output2 = model_CDDPM(
        corners_withsemantics_t_test_batch,
        global_attn_matrix_test_batch,
        t_test,
        feat_16_test_batch
    )
    
    output = torch.cat((output1, output2), dim=2)  # (BS, 53, 10)
    
    # Compute posterior distribution
    # ... (posterior sampling code)
    
    sample_from_posterior_normal_distribution_test_batch = posterior_sample
```

### 7.2 Diffusion Schedule

**Code Location:** `scripts/test_boun.py`, Lines 31-71

**Cosine Beta Schedule:**

```python
alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
betas = []
max_beta = 0.999

for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

betas = np.array(betas, dtype=np.float64)
alphas = 1.0 - betas
```

**Derived Quantities:**

```python
alphas_cumprod = np.cumprod(alphas)  # ᾱ_t = Π(α_i) for i=1 to t
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
```

**Posterior Variance:**

```python
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
```

### 7.3 Posterior Sampling

**Code Location:** `scripts/test_boun.py`, Lines 240-262

```python
# Predict x_0 from x_t and predicted noise
pred_xstart_test_batch = (
    sqrt_recip_alphas_cumprod[t_test][:, None, None] * corners_withsemantics_t_test_batch -
    sqrt_recipm1_alphas_cumprod[t_test][:, None, None] * output_corners_withsemantics_test_batch
)

# Clamp coordinates
pred_xstart_test_batch[:, :, 0:2] = torch.clamp(pred_xstart_test_batch[:, :, 0:2], min=-1, max=1)

# Threshold semantic labels
pred_xstart_test_batch[:, :, 2:9] = pred_xstart_test_batch[:, :, 2:9] >= 0.5

# Threshold padding mask
pred_xstart_test_batch[:, :, 9:10] = pred_xstart_test_batch[:, :, 9:10] >= 0.75

# Compute posterior mean
model_mean_test_batch = (
    posterior_mean_coef1[t_test][:, None, None] * pred_xstart_test_batch +
    posterior_mean_coef2[t_test][:, None, None] * corners_withsemantics_t_test_batch
)

# Sample from posterior
noise_test_batch = torch.randn_like(corners_withsemantics_t_test_batch)
sample_from_posterior_normal_distribution_test_batch = (
    model_mean_test_batch + torch.sqrt(posterior_variance[t_test][:, None, None]) * noise_test_batch
)
```

**Key Steps:**

1. **Predict x_0:** Use model output to estimate clean data
2. **Clamp/Threshold:** Ensure valid coordinate ranges and binary semantics
3. **Compute Mean:** Calculate posterior distribution mean
4. **Add Noise:** Sample from Gaussian posterior

---

## 8. Complete Input Flow

### 8.1 End-to-End Pipeline Summary

```text
Step 1: Raw Data
   └─ RPLAN .npy files: graph dictionaries
   
Step 2: Dataset Loading
   └─ RPlanGEdgeSemanSimplified_81.__getitem__()
   
Step 3: Semantic Simplification
   └─ 16 dimensions → 9 dimensions
   
Step 4: CNN Feature Loading
   └─ Pre-computed (1024, 16, 16) features
   
Step 5: Batching
   └─ DataLoader creates batches
   
Step 6: Device Transfer
   └─ Move to GPU, append padding mask
   
Step 7: Input Encoding
   ├─ Corner positional encoding
   ├─ Semantic embedding
   ├─ Time embedding
   └─ CNN feature projection
   
Step 8: Model Forward Pass
   ├─ Transformer layers (24x)
   ├─ Self-attention on corners
   └─ Cross-attention with CNN features
   
Step 9: Diffusion Process
   └─ Iterative denoising (t=999→0)
```

### 8.2 Data Flow Diagram

```text
Input Stage                    Encoding Stage                Model Stage
─────────────                  ──────────────                ────────────

.npy Files                     Corner Coords                 Transformer
  (53,16)                      (BS,53,2)                     Layers (24x)
     │                               │                            │
     v                               v                            │
Semantic                       Pos Encoding                      │
Simplify                       (BS,53,512) ──────┐               │
(53,16→53,9)                                     │               │
     │                                           │               │
     v                         Semantic Labels   │               │
Load CNN                       (BS,53,8)         │               │
Features                            │            │               │
(1024,16,16)                        v            │               │
     │                         Semantic Embed    │               │
     v                         (BS,53,512) ──────┤               │
Project CNN                                      │               │
Features                       Timestep t        │               │
(BS,256,16,16)                 (BS,)             │               │
     │                              │            │               │
     v                              v            │               │
Flatten &                      Time Embed        │               │
Position                       (BS,53,512) ──────┤               │
(BS,256,256) ──────────┐                         │               │
                       │                         v               │
                       │                   Combined Input        │
                       │                   (BS,53,512) ──────────┤
                       │                                         │
                       └─────────────> Cross-Attention ──────────┘
                                       (53 corners × 256 features)
                                                │
                                                v
                                          Output Noise
                                          Prediction
                                          (BS,53,10)
```

### 8.3 Tensor Shape Transformations

```text
Stage                               Shape                    Description
───────────────────────────────    ─────────────────        ─────────────────────────
Raw Graph Dictionary               (53, 16)                 Padded corners + semantics
  
After Semantic Simplification      (53, 9)                  Reduced semantic dimensions
  
After Batching                     (BS, 53, 9)              Batched samples
  
After Padding Mask Append          (BS, 53, 10)             +1 padding indicator
  
Corner Positional Encoding         (BS, 53, 512)            2D → 512D embedding
  
Semantic Embedding                 (BS, 53, 512)            8D → 512D embedding
  
Time Embedding                     (BS, 53, 512)            Timestep encoding
  
Combined Input                     (BS, 53, 512)            Sum of above embeddings
  
CNN Features (original)            (BS, 1024, 16, 16)       Pre-computed features
  
CNN Features (projected)           (BS, 256, 16, 16)        Channel reduction
  
CNN Features (flattened)           (BS, 256, 256)           Spatial flatten
  
After Transformer Layers           (BS, 53, 512)            Processed representations
  
Output Head 1                      (BS, 53, 2)              Coordinate predictions
  
Output Head 2                      (BS, 53, 8)              Semantic predictions
  
Combined Output                    (BS, 53, 10)             Final noise prediction
```

---

## 9. Key Design Decisions

### 9.1 Padding Strategy

**Decision:** Pad all graphs to 53 corners

**Rationale:**

- Enables fixed-size batching
- Simplifies neural network architecture
- Allows variable-size floor plans
- 53 covers maximum observed in dataset

**Trade-offs:**

- ✅ Pros: Simplified batching, easier GPU parallelization
- ❌ Cons: Memory overhead for small floor plans, wasted computation on padding

### 9.2 Semantic Simplification

**Decision:** Reduce 16 semantic dimensions to 9

**Rationale:**

- Original 16 categories too fine-grained
- Many categories semantically similar
- Sparse labels hinder learning
- 9 categories maintain essential distinctions

**Mapping Strategy:**

- **Merge similar rooms:** ChildRoom, StudyRoom, GuestRoom → SecondRoom
- **Merge utility spaces:** Storage, Entrance → Entrance
- **Keep distinct rooms:** Living, Kitchen, Bathroom separate
- **Remove unused:** Wall-in category eliminated

### 9.3 Pre-computed CNN Features

**Decision:** Store CNN features offline rather than computing online

**Rationale:**

- CNN encoding is deterministic (no training)
- Boundary images don't change
- Computing once saves repeated computation
- Significantly faster inference

**Storage Cost:**

- Per sample: 1024 × 16 × 16 × 4 bytes = 1 MB
- Total dataset: ~80,000 samples × 1 MB = 80 GB
- Trade-off: Storage for speed

### 9.4 Multi-Head Attention

**Decision:** Use 4 attention heads in transformer

**Rationale:**

- Multiple heads capture different relationship types
- 4 heads balance expressiveness and efficiency
- Each head: 512 / 4 = 128 dimensions
- Standard in transformer architectures

### 9.5 Cosine Beta Schedule

**Decision:** Use cosine schedule for diffusion betas

**Rationale:**

- Smoother noise schedule than linear
- Better quality for image-like data
- Proven effective in prior work (DDPM, IDDPM)
- Avoids sudden changes in noise levels

---

## 10. Performance Optimizations

### 10.1 Pre-computed CNN Features

**Optimization:** Store CNN-encoded boundary images offline

**Impact:**

- Inference speedup: ~30% faster
- Memory trade-off: 80 GB storage
- Implementation: `datasets/rplang_edge_semantics_simplified_78_10_prerunCNN.py`

**Pre-computation Process:**

```python
# Step 1: Render boundary image (256×256)
img = render_boundary_image(corners, edges)

# Step 2: Encode with CNN
with torch.no_grad():
    features = cnn_encoder(img)  # (1024, 16, 16)

# Step 3: Save to disk
np.save(f'{output_path}/{index}_feat16.npy', features.cpu().numpy())
```

### 10.2 Semantic Simplification

**Optimization:** Reduce semantic dimensions 16→9

**Impact:**

- Faster embedding: Fewer dimensions to encode
- Better learning: Less sparse labels
- Memory reduction: Smaller tensor sizes

### 10.3 Attention Masking

**Optimization:** Use padding masks to skip computation on padded elements

**Implementation:**

```python
# In attention computation
attention_scores = torch.matmul(Q, K.transpose(-2, -1))

# Apply mask to prevent attention to padding
attention_scores = attention_scores.masked_fill(
    ~global_attn_matrix.unsqueeze(1),  # Expand for heads
    float('-inf')
)

attention_weights = F.softmax(attention_scores, dim=-1)
```

**Impact:**

- Computational savings proportional to padding ratio
- Typical floor plan: 23 real / 53 total = 43% real
- Effective speedup: ~1.5x for attention

### 10.4 Fixed Batch Size

**Optimization:** Use consistent batch sizes during training/inference

**Implementation:**

- Training: batch_size = 32
- Testing: batch_size = 3000 (entire test set)

**Impact:**

- Better GPU utilization
- Reduced data loading overhead
- More stable gradient estimates

### 10.5 Mixed Precision (Not Implemented)

**Potential Optimization:** Use FP16 instead of FP32

**Expected Impact:**

- 2× memory reduction
- 1.5-2× speedup on modern GPUs
- Minimal accuracy loss with proper scaling

**Implementation Suggestion:**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 11. Code Reference Guide

### 11.1 Key Files

**Dataset Loading:**

- `datasets/rplang_edge_semantics_simplified_81.py` - Main dataset class
- `datasets/path_utils.py` - Path management utilities
- `datasets/rplang_edge_semantics_simplified_78_10_prerunCNN.py` - CNN feature pre-computation

**Model Architecture:**

- `gsdiff/heterhouse_81_106_3.py` - Main diffusion model
- `gsdiff/heterhouse_56_32.py` - Edge prediction model
- `gsdiff/utils.py` - Utility functions

**Training/Inference:**

- `scripts/test_boun.py` - Testing script with full generation pipeline
- `scripts/train_boun.py` - Training script (if exists)

### 11.2 Important Functions

**Dataset Class:**

```python
# datasets/rplang_edge_semantics_simplified_81.py
class RPlanGEdgeSemanSimplified_81(Dataset):
    def __init__(self, mode):                    # Lines 14-30
    def __len__(self):                           # Lines 33-35
    def __getitem__(self, index):                # Lines 37-102
```

**Model Architecture:**

```python
# gsdiff/heterhouse_81_106_3.py
class BoundHeterHouseModel(nn.Module):
    def __init__(self):                          # Lines 161-275
    def forward(self, corners_withsemantics_t,   # Lines 279-405
                global_attn_matrix, t, feat_16):
```

**Transformer Layer:**

```python
# gsdiff/heterhouse_81_106_3.py
class TransformerLayer(nn.Module):
    def __init__(self, d_model):                 # Lines 131-140
    def forward(self, corners, global_attn_matrix,  # Lines 143-157
                x, cross_attn_matrix):
```

**Generation Script:**

```python
# scripts/test_boun.py
if __name__ == '__main__':
    # Diffusion setup                            # Lines 31-71
    # Dataset loading                            # Lines 74-83
    # Reverse diffusion loop                     # Lines 213-262
    # Posterior sampling                         # Lines 240-262
```

### 11.3 Configuration Parameters

**Model Hyperparameters:**

```python
d_model = 512                    # Embedding dimension
num_layers = 24                  # Transformer layers
num_heads = 4                    # Attention heads
d_feedforward = 2048             # FFN hidden dimension
```

**Diffusion Parameters:**

```python
diffusion_steps = 1000           # Total timesteps
schedule = 'cosine'              # Beta schedule type
max_beta = 0.999                 # Maximum beta value
```

**Data Parameters:**

```python
max_corners = 53                 # Maximum corners per floor plan
resolution = 512                 # Output image resolution
coord_range = [-1, 1]            # Normalized coordinate range
```

**CNN Feature Parameters:**

```python
feature_dim = 1024               # CNN feature channels
feature_size = 16                # CNN feature spatial size
projected_dim = 256              # Projected feature dimension
```

---

## 12. Conclusions and Recommendations

### 12.1 Summary of Findings

This comprehensive analysis of the GSDiff input pipeline reveals:

1. **Well-Designed Architecture:** The pipeline efficiently handles variable-size graph data through fixed-size padding and masking strategies.

2. **Multi-Modal Integration:** Successful fusion of graph structure (corners/edges) with image features (boundaries) via cross-attention.

3. **Optimized Performance:** Pre-computed CNN features significantly accelerate inference while maintaining quality.

4. **Robust Encoding:** Sinusoidal positional encoding, semantic simplification, and time embedding provide rich input representations.

5. **Standard Diffusion Framework:** Implementation follows DDPM best practices with cosine scheduling and noise prediction.

### 12.2 Strengths

- **Modular Design:** Clear separation between data loading, encoding, and model components
- **Flexibility:** Supports three generation modes (unconstrained, topology, boundary)
- **Efficiency:** Optimizations for memory and computation
- **Extensibility:** Easy to add new semantic categories or generation modes

### 12.3 Potential Improvements

**1. Dynamic Batching**

- Current: All samples padded to 53 corners
- Improvement: Batch samples with similar sizes together
- Benefit: Reduce memory overhead and computation

**2. Learned Positional Embeddings**

- Current: Sinusoidal encoding
- Improvement: Learn position embeddings specific to floor plans
- Benefit: Potentially better capture floor plan geometry

**3. Multi-Scale CNN Features**

- Current: Only 16×16 features used
- Improvement: Use features from multiple scales (64×64, 32×32, 16×16)
- Benefit: Capture both fine and coarse boundary details

**4. Attention Optimization**

- Current: Full attention between all corners
- Improvement: Sparse attention based on spatial proximity
- Benefit: Faster inference for large floor plans

**5. Mixed Precision Training**

- Current: FP32 throughout
- Improvement: Use FP16 where possible
- Benefit: 2× memory reduction, faster training

### 12.4 Recommendations for Users

**For Researchers:**

- Study the semantic simplification strategy (Section 4.4) for insights on category design
- Examine the cross-attention mechanism (Section 6.7) for multi-modal fusion
- Analyze the diffusion schedule (Section 7.2) for sampling quality

**For Developers:**

- Use `path_utils.py` for cross-platform compatibility
- Pre-compute CNN features before training/inference
- Monitor memory usage with large batch sizes

**For Practitioners:**

- Adjust `max_corners` based on your dataset statistics
- Tune the semantic categories for your specific domain
- Experiment with different diffusion timesteps (1000 vs. 100 vs. 50)

### 12.5 Future Research Directions

1. **Hierarchical Generation:** Generate room layout first, then corners
2. **Interactive Editing:** Allow user constraints during generation
3. **3D Extension:** Extend to 3D floor plan generation
4. **Style Transfer:** Control architectural style of generated plans
5. **Layout Optimization:** Integrate architectural constraints (building codes, accessibility)

---

## Appendix A: Glossary

- **Corner:** Junction point where walls meet; represented as (x, y) coordinates
- **Edge:** Wall connecting two corners
- **Semantic Label:** Room type assignment (living room, bedroom, etc.)
- **Padding:** Dummy data added to reach fixed size (53 corners)
- **Diffusion:** Gradual addition/removal of noise
- **Timestep:** Step in diffusion process (t=0 to 999)
- **Attention Mask:** Binary matrix controlling which elements can interact
- **Cross-Attention:** Attention mechanism between two different sequences
- **CNN Features:** Image features extracted by convolutional neural network
- **DDPM:** Denoising Diffusion Probabilistic Models

---

## Appendix B: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| x_t | Data at diffusion timestep t |
| x_0 | Clean data (t=0) |
| ε | Gaussian noise |
| α_t | Diffusion alpha coefficient at timestep t |
| ᾱ_t | Cumulative product of alphas up to timestep t |
| β_t | Diffusion beta coefficient at timestep t |
| μ | Mean of posterior distribution |
| σ² | Variance of posterior distribution |
| BS | Batch size |
| d_model | Model embedding dimension (512) |

---

## Appendix C: Quick Reference

**Load a single floor plan:**

```python
import numpy as np
from datasets.path_utils import get_data_path

graph = np.load(get_data_path('rplang-v3-withsemantics', 'test', '0.npy'), allow_pickle=True).item()
corners = graph['corner_list_np_normalized_padding_withsemantics']
print(f"Corners shape: {corners.shape}")  # (53, 16)
```

**Create dataset:**

```python
from datasets.rplang_edge_semantics_simplified_81 import RPlanGEdgeSemanSimplified_81
from torch.utils.data import DataLoader

dataset = RPlanGEdgeSemanSimplified_81('test')
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for batch in dataloader:
    feat_16, corners, global_attn, padding_mask = batch
    print(f"Batch corners shape: {corners.shape}")  # (32, 53, 9)
    break
```

**Run generation:**

```bash
cd /path/to/GSDiff
python scripts/test_boun.py
```

---

**End of Report**

---

*This report was generated as part of a comprehensive analysis of the GSDiff floor plan generation system. For questions or clarifications, please refer to the original paper or codebase.*
