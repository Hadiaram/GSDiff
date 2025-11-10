# GSDiff Codebase Analysis: Training Process, Data Usage, and Model Behavior

**Date:** November 10, 2025  
**Repository:** GSDiff (AAAI 2025)  
**Paper:** "GSDiff: Synthesizing Vector Floor Plans via Geometry-enhanced Structural Graph Generation"

---

## PART 1: TRAINING PROCESS

### 1.1 Training Architecture Overview

GSDiff uses a **two-stage training pipeline**:

1. **Stage 1 (Node/Corner Generation):** Diffusion-based corner prediction with Transformer
2. **Stage 2 (Edge Generation):** Transformer-based edge prediction

Both stages are **jointly trained but independently** during the training process.

### 1.2 Stage 1: Node Generation Training

**Main Script:** `/home/user/GSDiff/scripts/trainval_main_boun.py` (779 lines)

**Model:** `BoundHeterHouseModel` (from `gsdiff/heterhouse_81_106_3.py`)

**Key Training Parameters:**
```python
diffusion_steps = 1000
lr = 1e-4
weight_decay = 1e-7
total_steps = 1000000  # 1 million steps
batch_size = 256
batch_size_val = 3000
device = 'cuda:0'
clamp_trick_training = True  # Important regularization
merge_points = True          # Merges nearby corners
```

**Model Architecture:**
- **Input:** `(batch_size, 53, 10)` tensor containing:
  - Corner coordinates (x, y): normalized to [-1, 1]
  - Semantics: 7-dimensional one-hot room type encoding
  - Padding mask: binary indicator of valid/padded corners (1 or 0)
  
- **Layers:** 24 Transformer layers with:
  - Vertex-to-vertex (v-v) self-attention
  - Vertex-to-image (v-i) cross-attention with CNN features
  - Feed-forward networks
  
- **Output:** 
  - `output_corners1`: (batch_size, 53, 2) - predicted noise
  - `output_corners2`: (batch_size, 53, 8) - semantics + padding mask

**Diffusion Process:**

The training uses a standard **DDPM (Denoising Diffusion Probabilistic Model)** with cosine beta schedule:

```python
# Cosine annealing for variance schedule
alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2

# Precomputed diffusion coefficients:
sqrt_alphas_cumprod[t]         # Coefficient for x_0
sqrt_one_minus_alphas_cumprod[t]  # Coefficient for noise
posterior_variance[t]           # Posterior distribution variance
posterior_mean_coef1/2[t]       # Mean coefficients
```

### 1.3 Training Loop Details

**Single Training Iteration Flow:**

```
1. Sample Batch:
   - feat_16: (256, 1024, 16, 16)  [Pre-computed CNN features]
   - corners_withsemantics_0: (256, 53, 9)  [Clean data]
   - global_attn_matrix: (256, 53, 53)
   - corners_padding_mask: (256, 53, 1)

2. Add Padding Mask to Input:
   corners_withsemantics_0 = torch.cat(
       (corners_withsemantics_0, (1 - corners_padding_mask)), dim=2
   )  # Now (256, 53, 10)

3. Sample Random Timestep:
   t ~ Uniform(0, 999)  # Uniform distribution over all timesteps

4. Add Noise (Forward Process):
   noise = randn_like(corners_withsemantics_0)
   corners_t = sqrt_alphas_cumprod[t] * corners_0 + 
              sqrt_one_minus_alphas_cumprod[t] * noise

5. CNN Feature Processing:
   feat_16_proj = Conv2D(feat_16)  # Project to 256 channels
   feat_16_pos = PositionEmbeddingSine(feat_16_proj)  # Add positional encoding
   x = feat_16_pos + feat_16_proj  # Combine

6. Model Forward Pass:
   output1, output2 = BoundHeterHouseModel(
       corners_t,           # Noised corners
       global_attn_matrix,  # Attention mask
       t,                   # Timestep embedding
       feat_16              # CNN features
   )

7. Compute Loss:
   # Noise prediction loss
   noise_loss = L2(output_concat - noise)
   
   # Geometric alignment loss (local alignment)
   - Distance-based L1 metric on predicted x0
   - Multi-base encoding (binary, quaternary, octal, hexadecimal)
   - Penalizes corners that are too close
   
   total_loss = noise_loss + 1.0 * local_alignment_loss

8. Backpropagation & Update:
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
   optimizer.step()
```

**Key Loss Components:**

1. **Primary Loss (Noise Prediction):** L2 loss on predicted noise
   ```python
   corners_loss_masked1 = (corners_withsemantics_target - output) ** 2
   corners_loss_batch1 = corners_loss_masked1.sum(dim=[1, 2]) / corner_number
   ```

2. **Local Alignment Loss** (Prevents corner collapse):
   - Computes minimum L1 distance between corners
   - Applies logarithmic penalty: `-2*log(1 - 0.5*distance)`
   - Uses time-weighted coefficient: `betas[t]` (heavier at early timesteps)
   - Multi-base encoding for numerical stability
   ```python
   local_aligned_loss = (
       masked_tensor_bin + masked_tensor_four + 
       masked_tensor_eight + masked_tensor_sxt + masked_tensor_inf
   )
   ```

### 1.4 Stage 2: Edge Generation Training

**Main Script:** `/home/user/GSDiff/scripts/trainval_simplified_edge_unconstrained.py` (263 lines)

**Model:** `EdgeModel` (from `gsdiff/heterhouse_56_11.py`)

**Key Training Parameters:**
```python
lr = 1e-4
weight_decay = 1e-5
batch_size = 8
total_steps = float("inf")
interval = 100  # Validation/save every 100 steps
```

**Edge Model Architecture:**
- **Input:** `(batch_size, 53, 2)` corners + semantics
- **Output:** `(batch_size, 2809, 2)` edge logits [not_edge, has_edge]
- **12 Transformer layers** with self-attention
- **Loss:** Cross-entropy loss with padding mask

**Training Details:**
```python
# Model processes edge representation:
# For each pair of corners (i, j), compute:
# - Coordinates embedding (sinusoidal)
# - Semantic embedding
# - Combined edge embedding

output_edges = model(corners, global_attn_matrix, 
                      corners_padding_mask, semantics)
# Output shape: (batch_size, 2809, 2)

# Cross-entropy loss on valid edges only
edges_CELoss = CrossEntropyLoss(reduction='none')(output_edges, target)
edges_loss = sum(loss_valid) / num_valid
```

---

## PART 2: MODEL TRAINING VS PRE-TRAINED WEIGHTS

### 2.1 Pre-trained Models vs Training

**Question: Does the model retrain when given new data or use pre-trained weights?**

**Answer: HYBRID APPROACH**

1. **Two Independent Models with Pre-training:**
   - **Stage 1 (Node Model):** Can be trained from scratch OR loaded from pre-trained weights
   - **Stage 2 (Edge Model):** Can be trained from scratch OR loaded from pre-trained weights

2. **CNN Feature Encoder (Frozen):**
   - **Critical Finding:** The CNN boundary encoder is ALWAYS pre-trained and FROZEN
   - Cannot be retrained; only used for feature extraction
   - Location: Pre-computed features in `datasets/prerunning_cnn_featuremaps/`

3. **Available Pre-trained Checkpoints:**
   - Unconstrained params: Google Drive link
   - Topology-constrained params: Google Drive link
   - Boundary-constrained params: Google Drive link
   - Boundary CNN params: Google Drive link
   - Topology Transformer params: Google Drive link

### 2.2 CNN Feature Pre-computation (Critical!)

**Script:** `/home/user/GSDiff/scripts/prerunningCNN.py`

**Purpose:** Pre-compute CNN features for the entire dataset to save GPU memory during training

**Process:**
```python
# Load pre-trained CNN (frozen)
pretrained_encoder = BoundaryModel()
pretrained_encoder.load_state_dict(
    torch.load('outputs/structure-78-12/model_stage0_best_006700.pt')
)
for param in pretrained_encoder.parameters():
    param.requires_grad = False  # FROZEN

# For each image in dataset:
feat_16 = pretrained_encoder(image)  # Extract features
# Save: feat_16 shape = (1, 1024, 16, 16)
np.save(f'datasets/prerunning_cnn_featuremaps/{file_id}.npy', 
        {16: feat_16.cpu().numpy()})
```

**Key Insight from Code Comments:**
```python
"""
We found that calculating feature maps through CNN in real time 
consumes too much GPU memory, so after training CNN, the feature maps 
corresponding to the boundary images of the entire dataset are stored 
on disk for reading.
Note: this script needs about 500GB disk space.
"""
```

### 2.3 Dataset Loading with Pre-computed Features

**Dataset Class:** `RPlanGEdgeSemanSimplified_81` (from `datasets/rplang_edge_semantics_simplified_81.py`)

**Pre-computed Features Loading:**
```python
def __init__(self, mode):
    self.ftmps = []  # Pre-loaded feature maps
    for fn in tqdm(self.files):
        # Load pre-computed 16x16 feature map (1024 channels)
        feat = np.load(f'datasets/prerunning_cnn_featuremaps/{fn}', 
                      allow_pickle=True).item()
        self.ftmps.append(feat[16][0])  # Store only 16x16 scale

def __getitem__(self, index):
    featmap_16 = self.ftmps[index]  # Retrieve pre-loaded features
    # Load graph data
    graph = np.load(f'datasets/rplang-v3-withsemantics/{file}', 
                   allow_pickle=True).item()
    # Return: (featmap_16, corners, attention_mask, padding_mask)
```

---

## PART 3: DATA FLOW IN TRAINING

### 3.1 Input Data Components

**Each training sample contains:**

```
1. corners_withsemantics_simplified: (53, 9)
   ├─ Columns 0-1: Corner coordinates (normalized to [-1, 1])
   ├─ Columns 2-8: 7-dimensional semantic one-hot encoding
   │  ├─ Combined from original 16 semantic dimensions
   │  ├─ Example combination: col2 = sum(cols[2,6,12])
   │  └─ Represents room types: living, bedroom, kitchen, bathroom, etc.
   └─ Padding: 53 fixed size (padded if fewer corners)

2. global_attn_matrix: (53, 53)
   ├─ Boolean attention mask
   ├─ Value 1 where valid corners exist
   └─ Used to mask out padding corners

3. corners_padding_mask: (53, 1)
   ├─ Binary mask: 1 = valid corner, 0 = padding
   └─ Used for loss computation

4. featmap_16: (1024, 16, 16)  [Pre-computed CNN features]
   ├─ From frozen boundary encoder
   ├─ 16x16 spatial resolution
   └─ 1024 feature channels
```

### 3.2 Data Pre-processing Pipeline

**Source Data Preparation** (from DATA_GENERATION_PIPELINE.md):

```
Raw RPLAN PNGs (256x256)
  ├─ Extract floor plan from RGBA channels
  ├─ Channel 1 (Green) contains semantic labels (0-15 for rooms, ≥14 for walls)
  │
Stage 1-3: Edge & Corner Detection
  ├─ Binary wall extraction (threshold pixel_value ≥ 14)
  ├─ Morphological operations (erosion, skeletonization, dilation)
  ├─ Corner detection (L, T, X junctions)
  ├─ Edge extraction between corners
  │
Stage 4-5: Graph Construction & Semantics
  ├─ Adjacency matrix construction
  ├─ Coordinate normalization to [-1, 1] (256→normalized)
  ├─ Semantic label extraction per corner
  │   └─ Map room pixel values to corner locations
  │
Stage 6-7: Padding & Attention Masks
  ├─ Pad corners to fixed 53 corners
  ├─ Create boolean attention mask (valid=True, padded=False)
  │
Output: .npy file with dictionary:
  {
    'corner_list_np_normalized_padding_withsemantics': (53, 16),
    'global_matrix_np_padding': (53, 53),
    'padding_mask': (53, 1),
    'edges': (2809, 1)
  }
```

**Total Dataset:**
- Train: 65,763 samples
- Val: 3,000 samples
- Test: 3,000 samples

### 3.3 Training Data Flow (Per Iteration)

```
DataLoader (Batch Size 256)
    ↓
[1] Load 256 samples from disk
    - Load .npy graph dictionaries
    - Load pre-computed CNN features
    
[2] Data transformation
    - Simplify semantics (16→7 dimensions)
    - corners_withsemantics: (256, 53, 9)
    - global_attn_matrix: (256, 53, 53)
    - corners_padding_mask: (256, 53, 1)
    - feat_16: (256, 1024, 16, 16)
    
[3] Add padding indicator to input
    - Concatenate mask as 10th dimension
    - corners_withsemantics: (256, 53, 10)
    
[4] Sample timestep
    - t ~ Uniform(0, 999)
    - For each sample, t ∈ {0, 1, ..., 999}
    
[5] Forward diffusion (add noise)
    - corners_t = α_t * x_0 + σ_t * ε
    - Where ε ~ N(0, 1)
    
[6] Model forward
    - Input corners_t, mask, t, feat_16
    - Process through 24 Transformer layers
    - Output: predicted noise, semantics
    
[7] Loss computation
    - Noise prediction loss: L2(pred_noise - actual_noise)
    - Geometric alignment loss: distance-based penalty
    - Total loss = noise_loss + alignment_loss
    
[8] Backward & optimization
    - Gradient clipping (norm=0.1)
    - Adam update
    - Continue to next batch
```

---

## PART 4: DATA FLOW IN INFERENCE/GENERATION

### 4.1 Inference Process (Reverse Diffusion)

**Main Scripts:** 
- Unconstrained: `scripts/test_main.py`
- Boundary-constrained: `scripts/test_boun.py`

**Inference Flow:**

```
Step 1: Initialize from Noise
    x_999 ~ N(0, 1)
    Shape: (batch_size, 53, 10)
    
Step 2: Reverse Diffusion Loop (t = 999 → 0)
    for t in range(999, -1, -1):
        
        [1] Model Prediction
        with torch.no_grad():
            output1, output2 = model(
                x_t,                    # Current noisy corners
                global_attn_matrix,     # Fixed attention mask
                t,                      # Timestep
                feat_16                 # Fixed CNN features
            )
            pred_x_0 = combine_predictions(output1, output2, t)
        
        [2] Posterior Sampling
        # Compute posterior mean from predicted x_0
        posterior_mean = coef1 * pred_x_0 + coef2 * x_t
        
        # Sample from posterior (add small noise)
        z ~ N(0, 1)
        x_{t-1} = posterior_mean + sqrt(posterior_var) * z
        
        [3] Clamp Coordinates
        pred_x_0[:, :, 0:2] = clamp(pred_x_0[:, :, 0:2], -1, 1)
        pred_x_0[:, :, 2:] = (pred_x_0[:, :, 2:] >= 0.5).float()

Step 3: Extract Generated Corners
    corners_pred = x_0[:, :, :2]  # First 2 dimensions
    semantics_pred = x_0[:, :, 2:9]  # Semantic labels
    padding_mask = x_0[:, :, 9]  # Which corners are valid

Step 4: Post-processing
    - Inverse normalize: corners = (corners * 128 + 128) / 256 * image_size
    - Remove padding: only keep corners where mask == 1
    - Merge nearby corners (if enabled)
    - Align corners (if enabled)
```

### 4.2 CNN Features During Inference

**Critical Point:** CNN features are **FIXED during generation**!

```python
# For boundary-constrained generation:
feat_16 = precomputed_features[sample_id]  # Retrieved once at start

# Used in ALL 1000 diffusion steps
for t in range(999, -1, -1):
    x_t = model(x_t, mask, t, feat_16)  # Same feat_16 every iteration
    # feat_16 guides the generation but doesn't change
```

**Purpose of CNN Features:**
- Encode boundary/structure information from pre-computed images
- Guide corner placement via cross-attention
- 16×16 spatial features (256 pixels) attention to 53 corners

### 4.3 Constrained vs Unconstrained Generation

**Unconstrained Generation:**
- No boundary constraint
- Random initialization: x_999 ~ N(0, 1)
- CNN features: zeros or not used
- Generates floor plans with free-form layouts

**Boundary-Constrained Generation:**
- Provide boundary image as input
- CNN feature extraction from boundary image
- x_999 ~ N(0, 1) (still random)
- CNN features guide generation toward valid floor plans
- Reduces invalid configurations

### 4.4 Stage 2: Edge Generation

**After corners are generated:**

```
Step 1: Load Stage 2 Model
    model_2 = EdgeModel()
    model_2.load_state_dict(
        torch.load('outputs/structure-56-16/model_stage2_best_010300.pt')
    )

Step 2: Edge Prediction
    for each sample:
        # Input generated corners + semantics
        output_edges = model_2(
            corners_pred,              # (1, n, 2)
            global_attn_matrix,        # (1, n, n)
            corners_padding_mask,      # (1, n, 1)
            semantics_pred             # (1, n, 7)
        )
        # Output shape: (1, n², 2) - edge logits
        
        # Convert to binary edges
        output_edges = softmax(output_edges, dim=2)
        output_edges = argmax(output_edges, dim=2)

Step 3: Graph Construction
    edges_pred = output_edges[:, :, 1]  # Extract "has_edge" class
    # Reshape to adjacency matrix and construct graph
```

---

## PART 5: DATA COMPONENTS AND THEIR USAGE

### 5.1 Corner Coordinates (Columns 0-1)

**Format:**
- Normalized to [-1, 1] range (from 256-pixel image)
- Transformation: `pixel_coords / 128 - 1`
- Reverse: `(normalized + 1) * 128`

**Usage in Training:**
```python
# Sinusoidal position encoding
corners_t_unnormalized = (corners_t * 128 + 128).float()

# w in sinusoidal encoding
div_term = (1 / 10000) ** (torch.arange(0, d_model//2, 2) / (d_model//2))

# Encode x coordinate
sin_x = torch.sin(corners_t[:, :, 0:1] * div_term)
cos_x = torch.cos(corners_t[:, :, 0:1] * div_term)

# Similar for y, then concatenate to form (bs, 53, 256) embedding
```

**Usage in Loss:**
- Geometric alignment loss computes L1 distances between corner pairs
- Prevents predicted corners from collapsing to same location
- Multi-base encoding for numerical precision

### 5.2 Semantic Labels (Columns 2-8)

**Original Format (16 dimensions, pre-simplification):**
- Room type one-hot encoding
- Categories: living, bedroom, kitchen, bathroom, dining, closet, study, etc.

**Simplified Format (7 dimensions):**
```python
# Combine semantics by summing specific dimensions:
# The 16 semantic types are combined into 7 more general categories
semantics_simplified[:, 2] = semantics_original[:, [2, 6, 12]].sum()  # Category 0
semantics_simplified[:, 3] = semantics_original[:, [3, 7, 8, 9, 10]].sum()  # Category 1
# ... etc
```

**Usage in Training:**
```python
# Semantic embedding
semantics_embedding = nn.Linear(8, d_model)(semantics_t.float())

# Concatenated with corner coordinates and time embedding
corners_total_embedding = corners_embedding + semantics_embedding + time_embedding
```

**Usage in Loss:**
- No explicit semantic loss for Stage 1
- Learned implicitly via end-to-end training
- Important for Stage 2 edge model (directly uses semantics)

### 5.3 Padding Mask (Column 10 after concatenation)

**Format:**
- 1 = valid corner
- 0 = padding corner

**Usage:**
```python
# During data loading
corners_withsemantics = torch.cat(
    (corners_withsemantics, (1 - corners_padding_mask)), dim=2
)

# In loss computation - mask out padding corners
mask = pred_x0[:, :, -1] == 0  # Find padded locations
# Only compute loss for valid corners

# In attention - prevent padding from attending
global_attn_matrix[:, i, j] = False if i or j is padding
```

### 5.4 Global Attention Matrix (53×53)

**Format:**
- Boolean matrix
- True where corners are valid
- False where corners are padding

**Usage:**
```python
# Self-attention masking in Transformer
scores = torch.matmul(q, k.T) / sqrt(d_k)
scores = scores.masked_fill(mask == False, -1e9)  # Large negative value
scores = softmax(scores)  # Padded positions → ~0 attention
```

### 5.5 CNN Features (16×16 spatial, 1024 channels)

**Source:**
- Pre-computed from frozen CNN encoder
- Extracted from boundary/floor plan image
- Stored in `datasets/prerunning_cnn_featuremaps/`

**Processing:**
```python
# Feature projection
proj_16 = Conv2D(1024 → 256)(feat_16)  # (bs, 256, 16, 16)

# Positional encoding
pos_16 = PositionEmbeddingSine(feat_16)  # (bs, 256, 16, 16)

# Combine
x = pos_16 + proj_16  # (bs, 256, 16, 16)

# Reshape for cross-attention
x = x.permute(0, 2, 3, 1).view(bs, 256, 256)  # (bs, 256 spatial, 256 channels)
```

**Usage in Cross-Attention:**
```python
# For each corner, compute cross-attention with all spatial features
# Query (Q): corner embeddings (bs, 53, 256)
# Key (K): image features (bs, 256, 256)
# Value (V): image features (bs, 256, 256)

cross_attn = MultiHeadCrossAttention()(Q=corners, K=feat, V=feat)
# Output: (bs, 53, 256) - enriched corner embeddings
```

**Attention Mask for Cross-Attention:**
```python
# Prevent padding corners from attending to features
cross_attn_matrix = global_attn_matrix[:, :, 0:1].repeat(1, 1, x.shape[1])
# Shape: (bs, 53, 256) - 1 for valid corners, 0 for padding
```

### 5.6 Edges (2809 values = 53×53)

**Format:**
- Flattened adjacency matrix
- 1 = edge exists between corners
- 0 = no edge

**Usage in Training:**
- **Stage 1 Model:** Not directly used in training (only for evaluation)
- **Stage 2 Model:** Target output for edge prediction

**Usage in Inference:**
- Reconstructed from Stage 2 predictions
- Used to form complete floor plan graph

---

## PART 6: PRE-TRAINED MODELS AND CHECKPOINTS

### 6.1 Available Pre-trained Weights

**From README.md:**

1. **Unconstrained Node Generation**
   - Download: Google Drive link
   - Location: `outputs/structure-1/`
   - Model class: `HeterHouseModel`

2. **Topology-Constrained Node Generation**
   - Download: Google Drive link
   - Location: `outputs/structure-?/`
   - Model class: `HeterHouseModel`

3. **Boundary-Constrained Node Generation**
   - Download: Google Drive link
   - Location: `outputs/structure-81-106-3/`
   - Model class: `BoundHeterHouseModel`

4. **Boundary CNN Encoder**
   - Download: Google Drive link
   - Location: `outputs/structure-78-12/`
   - Model class: `BoundaryModel`
   - Used for: Feature extraction (frozen)

5. **Edge Models**
   - Location: `outputs/structure-56-16/`
   - Model class: `EdgeModel`
   - Stage 2 edge prediction

### 6.2 Loading Pre-trained Models

**Stage 1 (Nodes):**
```python
model = BoundHeterHouseModel().to(device)
model.load_state_dict(
    torch.load('outputs/structure-81-106-3/model0001000.pt', 
               map_location=device)
)
model.eval()
for param in model.parameters():
    param.requires_grad = False
```

**Stage 2 (Edges):**
```python
model_2 = EdgeModel().to(device)
model_2.load_state_dict(
    torch.load('outputs/structure-56-16/model_stage2_best_010300.pt',
               map_location="cpu")
)
model_2.eval()
```

**CNN Features (Frozen):**
```python
pretrained_encoder = BoundaryModel().to(device)
pretrained_encoder.load_state_dict(
    torch.load('outputs/structure-78-12/model_stage0_best_006700.pt',
               map_location=device)
)
for param in pretrained_encoder.parameters():
    param.requires_grad = False  # CRUCIAL: Frozen
```

### 6.3 Can Users Train from Scratch or Must Use Pre-trained?

**Answer: BOTH OPTIONS AVAILABLE**

**Option 1: Use Pre-trained Models**
- Download weights from Google Drive
- Load into model
- Run inference with fixed weights
- Fastest approach, ready-to-use

**Option 2: Train from Scratch**
- Initialize model weights randomly
- Run training scripts (trainval_*.py)
- Requires:
  - Full dataset (71,763 samples)
  - Pre-computed CNN features (~500GB disk)
  - Long training time (1 million steps)
- Flexibility to modify architecture/loss

**Key Requirement:** 
- **CNN features must always be pre-computed**
- Cannot change during training
- Even if training stage 1/2 from scratch, CNN features are fixed

### 6.4 Training Configuration Options

**Different Constraint Types (Different Checkpoints):**

1. **No Constraints**
   - Model: `HeterHouseModel` 
   - Training script: `trainval_main_unconstrained.py`
   - CNN features: NOT used
   - Output: Random floor plans

2. **Topology Constraints**
   - Model: `HeterHouseModel` with topology encoder
   - Training script: `trainval_main_topo.py`
   - CNN features: From topology bubble diagram
   - Output: Topology-guided floor plans

3. **Boundary Constraints**
   - Model: `BoundHeterHouseModel`
   - Training script: `trainval_main_boun.py`
   - CNN features: From boundary image
   - Output: Boundary-consistent floor plans

---

## PART 7: MODEL BEHAVIOR SUMMARY

### Key Behaviors:

1. **Retraining:** Not required; pre-trained weights provided
2. **Data Dependency:** Requires structured .npy graph files
3. **CNN Features:** Always pre-computed, never retrained
4. **Two-Stage:** Sequential - first generate corners, then edges
5. **Diffusion:** 1000 steps in training, 1000 steps in inference
6. **Loss:** Multi-component (noise prediction + geometric alignment)
7. **Constraint Types:** Unconstrained, topology-constrained, boundary-constrained
8. **Inference:** Pure generation with NO label conditioning

---

## PART 8: DATA USAGE MATRIX

| Component | Training | Inference | Purpose |
|-----------|----------|-----------|---------|
| Corners (x,y) | Yes | Generated | Position reference |
| Semantics (7D) | Yes | Generated | Room type information |
| Padding mask | Yes | Generated | Valid corner indicator |
| Attention matrix | Yes | Fixed | Self-attention mask |
| CNN features | Yes (pre-computed) | Fixed | Boundary/structure guidance |
| Edges | No | Generated in Stage 2 | Graph connectivity |

---

