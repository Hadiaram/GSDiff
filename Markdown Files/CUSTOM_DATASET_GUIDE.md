# GSDiff: RPLAN Dataset Usage & Custom Dataset Support Analysis

**Date:** November 10, 2025  
**Repository:** GSDiff (AAAI 2025)  
**Analysis Scope:** Dataset requirements, inference capabilities, and custom dataset workflows

---

## EXECUTIVE SUMMARY

### Key Findings

1. **RPLAN Dataset is Required for Training Only**
   - NOT required for inference/generation
   - Used to create ~71,763 preprocessed `.npy` files
   - Can be replaced with custom data following the same pipeline

2. **CNN Features Drive Pre-training but Are Optional for Inference**
   - Pre-computed and frozen during training
   - Pre-trained boundary CNN from RPLAN cannot be retrained
   - Can use zeros or custom CNN features during inference

3. **Three Distinct Workflows Available**
   - Workflow 1: Use pre-trained models for generation (no dataset needed)
   - Workflow 2: Train new models on custom dataset
   - Workflow 3: Fine-tune pre-trained models on custom dataset

---

## PART 1: RPLAN DATASET USAGE

### 1.1 Where Is RPLAN Actually Used?

**Location in Code:**

- `/home/user/GSDiff/datasets/rplang-v3-withsemantics/` (processed data)
- `/home/user/GSDiff/datasets/prerunning_cnn_featuremaps/` (CNN features)
- Referenced in: `datasets/rplan_edge_semantics_simplified_*.py`

**RPLAN Dataset Size:**

- Source: 80,788 PNG images (256×256 with semantic channel)
- After filtering: 71,763 valid floor plans
- Split: 65,763 train + 3,000 val + 3,000 test

### 1.2 Is RPLAN Required for Training?

**SHORT ANSWER:** YES, but only to create preprocessed `.npy` files

**HOW IT'S USED:**

```
Raw RPLAN PNGs (80,788 files)
    ↓ [10-stage preprocessing pipeline]
    ↓
.npy Files (71,763 files)
    └─ Corner coordinates (normalized to [-1,1])
    └─ Semantic labels (14-dim per corner)
    └─ Adjacency matrices (53×53)
    └─ Padding masks
    ↓
Training Script Loads .npy Files
    └─ Gets: (corners_feat, global_attn_matrix, padding_mask)
    └─ Plus pre-computed CNN features (1024×16×16)
```

**Code Reference:**

- `datasets/rplang_edge_semantics_simplified_81.py` loads these files
- Training loop reads `.npy` files, never touches original PNGs

### 1.3 Is RPLAN Required for Inference/Generation?

**SHORT ANSWER:** NO

**WHY:**

```python
# Test/inference script (test_main.py) does this:

1. Load pre-trained model weights
   model.load_state_dict(torch.load('outputs/structure-1/model.pt'))

2. Initialize noise for generation
   x_999 = torch.randn(batch_size, 53, 10)  # Pure random noise

3. Run reverse diffusion
   for t in range(999, -1, -1):
       x_t = model(x_t, mask, t, feat_16)  # Generate next step

4. Output generated floor plans
   # No dataset needed at any point
```

**CNN Features During Inference:**

- Can be zeros: `feat_16 = torch.zeros(bs, 1024, 16, 16)` → unconstrained generation
- Can be pre-computed: `feat_16 = load_features(boundary_image)` → boundary-constrained
- Can be custom: `feat_16 = your_custom_CNN(your_image)` → custom-constrained

### 1.4 What Parts of Code Depend on RPLAN?

**DEPENDS ON RPLAN:**

```
✓ trainval_main_*.py (training scripts)
  - Load dataset from rplang-v3-withsemantics/
  - Load pre-computed features
  - Required for: training Stage 1 and Stage 2

✓ prerunningCNN.py (feature extraction)
  - Reads rplang-v3-withsemantics files
  - Generates CNN features for entire dataset
  - ~500GB disk space required

✓ rplan-extract.py through rplan-process10.py
  - Transform raw RPLAN PNGs → .npy files
  - Required: once, to create dataset
```

**DOES NOT DEPEND ON RPLAN:**

```
✗ test_main.py, test_boun.py, test_topo.py (inference scripts)
  - Load pre-trained weights only
  - Initialize random noise
  - No .npy files or PNGs needed
  
✗ Model architecture (gsdiff/heterhouse_*.py, house_nn*.py)
  - Pure PyTorch, no data dependencies
```

---

## PART 2: TRAINING REQUIREMENTS

### 2.1 Can You Train Models Without RPLAN?

**ANSWER:** YES, if you have equivalently formatted data

**YOU NEED:**

1. **Graph Structure Files (.npy format):**
   - Corner coordinates: (53, 2) - normalized to [-1, 1]
   - Semantic labels: (53, 16) - 14-dim semantic vectors
   - Adjacency matrix: (53, 53) - binary edges
   - Padding mask: (53, 1) - valid/invalid corners
   - Attention matrix: (53, 53) - full connectivity for real corners

2. **Pre-computed CNN Features (Optional but Recommended):**
   - Size: (1, 1024, 16, 16) per sample
   - From frozen CNN encoder
   - Can use boundary images or custom images

3. **Data Split:**
   - Train: minimum ~60,000 samples (more is better)
   - Val: ~3,000 samples
   - Test: ~3,000 samples

### 2.2 Custom Dataset Creation Workflow

**STEP 1: Prepare Your Data**

```
Your custom floor plans
├─ Format: PNG/JPG images or vector formats
├─ Requirements:
│  └─ Semantic information per room
│  └─ Wall/boundary information
│  └─ Standard resolution (256×256 recommended)
└─ Minimum: ~5,000 samples (ideally 50,000+)
```

**STEP 2: Extract Graph Structure**

```
Your Data → [Custom Preprocessing Pipeline] → .npy Files
├─ Convert images to binary wall/room representation
├─ Detect corners (junctions in wall structure)
├─ Extract edges between corners
├─ Assign semantic labels to corners
├─ Normalize coordinates to [-1, 1]
└─ Pad to fixed size (can use 53 or modify)
```

**STEP 3: Create Dataset Loader**

```python
# Copy and modify datasets/rplang_edge_semantics_simplified_81.py
class CustomDataset(Dataset):
    def __init__(self, mode):  # mode: 'train', 'val', 'test'
        self.files = os.listdir(f'your_custom_data/{mode}')
        self.ftmps = []
        
        # Load pre-computed CNN features
        for fn in tqdm(self.files):
            self.ftmps.append(
                np.load(f'prerunning_cnn_featuremaps/{fn}', 
                       allow_pickle=True).item()[16][0]
            )
    
    def __getitem__(self, index):
        # Load .npy file
        graph = np.load(f'your_custom_data/{mode}/{file}', 
                       allow_pickle=True).item()
        
        # Extract components
        corners = graph['corner_list_np_normalized_padding_withsemantics']
        attention = graph['global_matrix_np_padding']
        padding_mask = graph['padding_mask']
        
        return (self.ftmps[index], corners, attention, padding_mask)
```

**STEP 4: Pre-compute CNN Features**

```python
# Modify scripts/prerunningCNN.py for your data:

1. Load pre-trained CNN (or train your own)
2. For each sample in your dataset:
   - Render boundary image from corners + edges
   - Pass through CNN
   - Extract 16×16 feature maps (1024 channels)
   - Save to disk

# Or use custom CNN:
your_cnn = YourCustomCNN()
your_cnn.load_state_dict(torch.load('your_cnn.pt'))
for your_data in dataset:
    features = your_cnn(boundary_image)
    save_features(features)
```

**STEP 5: Train Your Model**

```bash
# Modify trainval_main_unconstrained.py:
from datasets.your_custom_dataset import CustomDataset

dataset_train = CustomDataset('train')
dataset_val = CustomDataset('val')

# Run training
python trainval_main_unconstrained.py
```

### 2.3 Hardcoded RPLAN Assumptions

**Fixed Size Limitation (53 corners):**

```python
# In datasets/rplan_edge_semantics_simplified_81.py, line 89
global_attn_matrix = np.ones((53, 53), dtype=np.uint8)  # HARDCODED

# To use custom max corners (e.g., 100):
# 1. Modify all .npy file creation to pad to 100
# 2. Change model input size: 53 → 100
# 3. Modify loss computation masks
```

**Semantic Dimension (14 original, simplified to 7):**

```python
# Original: 14 room types from RPLAN
# Simplified: 7 categories (room type grouping)

# For custom data:
# - Can use 14 dimensions if you have same room types
# - Can use different number (e.g., 5, 10) - just modify:
#   └─ Dataset loader (column indexing)
#   └─ Model input size
#   └─ Loss computation
```

**Coordinate Normalization Range [-1, 1]:**

```python
# RPLAN: normalized from 256-pixel images
normalized = (pixel_coords - 128) / 128  # [-1, 1] range

# For custom data: adapt to your image resolution
# Use same [-1, 1] normalization to match training
```

**Pre-computed Feature Map Scale (16×16, 1024 channels):**

```python
# RPLAN: feat_16 shape is (1, 1024, 16, 16)
# This comes from specific CNN architecture

# For custom CNN:
# - Can use different architecture
# - Just ensure consistent feature extraction
# - Modify model's cross-attention projection if needed
```

### 2.4 Required Retraining Scenarios

**MUST RETRAIN when:**

1. Number of corners changes (requires changing padding size)
2. Semantic dimensions change (requires changing embedding)
3. Input image resolution changes significantly
4. Want different constraint types (topology, boundary, unconstrained)

**CAN FINE-TUNE when:**

1. Using same RPLAN-trained CNN features (just different data)
2. Have new training data but same format
3. Want to adapt to slightly different domain

**CAN USE PRE-TRAINED when:**

1. Dataset format matches RPLAN (53 corners, 7 semantics)
2. CNN features compatible
3. Only doing inference

---

## PART 3: INFERENCE / GENERATION

### 3.1 Can You Generate Without Any Dataset?

**ANSWER:** YES, completely

**WHAT YOU NEED:**

1. Pre-trained model weights (Stage 1 + Stage 2)
2. Nothing else! No data files, no CNN features

**HOW IT WORKS:**

```python
# Unconstrained generation - no data at all
import torch
from gsdiff.heterhouse_56_31 import HeterHouseModel
from gsdiff.house_nn2 import EdgeModel

# Load models
node_model = HeterHouseModel()
node_model.load_state_dict(torch.load('model_stage1.pt'))
node_model.eval()

edge_model = EdgeModel()
edge_model.load_state_dict(torch.load('model_stage2.pt'))
edge_model.eval()

# Generate in pure inference mode
with torch.no_grad():
    # Stage 1: Generate corners (1000 diffusion steps)
    x = torch.randn(batch_size, 53, 10)  # Pure random
    feat_16 = torch.zeros(batch_size, 1024, 16, 16)  # No features
    
    for t in range(999, -1, -1):
        x_pred = node_model(x, attention_mask, t, feat_16)
        x = posterior_sample(x_pred, x, t)  # Update
    
    # Stage 2: Generate edges from corners
    edges = edge_model(x[:, :, :2], attention_mask, x[:, :, 2:])
    
# Done! No dataset needed at any point
```

### 3.2 RPLAN Dataset During Inference

**RPLAN Used For What?**

```python
# In test_main.py, test_boun.py:

# Dataset is loaded ONLY for:
dataset_test = RPlanGEdgeSemanSimplified('test')

# Used ONLY for:
1. Getting GT data to render comparison images
   ├─ For visualization
   ├─ For computing evaluation metrics (FID/KID)
   └─ NOT used in actual generation

2. Getting CNN features (in boundary-constrained mode)
   ├─ feat_16 = precomputed_cnn_features[sample_id]
   └─ Used to guide generation
```

**The actual generation process:**

```python
# This part doesn't need dataset:
for t in range(999, -1, -1):
    output1, output2 = model(
        x_t,                      # Current noisy state
        global_attn_matrix,       # Fixed mask
        t,                        # Timestep
        feat_16                   # Pre-loaded features
    )
    x_t = sample_next(output1, output2, x_t, t)

# feat_16 can be:
# - Zeros (unconstrained)
# - Pre-computed from RPLAN (boundary-constrained)
# - From custom image (custom-constrained)
# - Any (1, 1024, 16, 16) tensor
```

### 3.3 Pre-trained Model Compatibility

**RPLAN-trained Models Work With:**

```
✓ RPLAN test data (exact match)
✓ Any other data with same format:
  ├─ 53 corners (padded)
  ├─ 7-dim semantics
  ├─ 53×53 adjacency matrix
  └─ 1024-dim CNN features (optional)

✓ No data (just random initialization)
  └─ Unconstrained generation

✗ Different corner counts (e.g., 100)
✗ Different semantic dimensions (e.g., 5)
✗ Different feature dimensions (e.g., 512)
```

### 3.4 Pre-trained vs Training From Scratch

**Pre-trained Advantages:**

```
✓ Instant inference ready
✓ 71,763 RPLAN samples worth of knowledge
✓ Optimized for residential floor plans
✓ No training time/compute needed
✓ Tested on RPLAN benchmark

✗ Limited to RPLAN domain
✗ May need fine-tuning for different building types
```

**Training From Scratch Advantages:**

```
✓ No RPLAN domain limitation
✓ Can customize architecture
✓ Can use your own data distribution
✓ Can optimize for your specific use case

✗ Requires 60,000+ training samples
✗ Requires ~500GB disk (CNN features)
✗ ~1 million training steps (weeks of GPU time)
✗ No prior knowledge transfer
```

**Fine-tuning Advantages:**

```
✓ Start from RPLAN knowledge
✓ Adapt to your data distribution
✓ Faster convergence than from scratch
✓ Need fewer of your samples

✗ Still requires 10,000-20,000+ samples
✗ Need 100GB+ disk space
✗ Need days/weeks of training
```

---

## PART 4: CUSTOM DATASET REQUIREMENTS

### 4.1 Data Format Requirements

**Input Data (You Provide):**

```
Floor plan images or vector representations
├─ Format: PNG, JPG, SVG, or vector
├─ Resolution: 256×256 (can adapt)
├─ Required Information:
│  ├─ Wall/boundary structure
│  ├─ Corner locations (or extractable from walls)
│  ├─ Room polygons
│  └─ Room semantic labels (type: living, bedroom, etc.)
└─ Minimum samples: 5,000 (recommended: 50,000+)
```

**Processing to Standard Format:**

```
Your Data
    ↓
Extract Wall Structure
    ├─ Binary image: walls=white, rooms=black
    ├─ Morphological operations (erosion, dilation)
    └─ Skeleton extraction (single-pixel width)

    ↓
Detect Corners
    ├─ Find junctions (L, T, X corners)
    ├─ Filter artifacts
    └─ List: [(x1,y1), (x2,y2), ...]

    ↓
Extract Edges
    ├─ Connect orthogonal corners
    ├─ Verify continuous walls
    └─ Build adjacency matrix

    ↓
Assign Semantics
    ├─ Find rooms (cycle basis in graph)
    ├─ Query pixel values
    ├─ Determine room type
    └─ Create 14-dim semantic vectors

    ↓
Normalize & Pad
    ├─ Normalize coords: [0,255] → [-1,1]
    ├─ Pad to 53 corners
    ├─ Create padding mask
    └─ Save .npy file
```

### 4.2 Preprocessing Scripts Needed

**You need to create (or adapt from RPLAN):**

```python
# 1. Binary wall extraction
def extract_walls(image, threshold=14):
    binary = (image >= threshold).astype(np.uint8) * 255
    return binary

# 2. Corner detection
def detect_corners(binary_img):
    corners_L, corners_T, corners_X = [], [], []
    for i, j in all_edge_pixels:
        neighbor_count = count_white_neighbors(binary_img, i, j)
        if neighbor_count == 2 and perpendicular:
            corners_L.append((i, j))
        elif neighbor_count == 3:
            corners_T.append((i, j))
        elif neighbor_count == 4:
            corners_X.append((i, j))
    return corners_L + corners_T + corners_X

# 3. Edge extraction
def extract_edges(binary_img, corners):
    edges = []
    for c1, c2 in corner_pairs:
        if orthogonal(c1, c2) and continuous_wall(binary_img, c1, c2):
            edges.append((c1, c2))
    return edges

# 4. Room detection via cycle basis
def find_rooms(corners, edges):
    G = nx.Graph()
    # Build graph...
    cycle_basis = nx.cycle_basis(G)
    return cycle_basis

# 5. Semantic assignment
def assign_semantics(rooms, semantic_img):
    semantics = []
    for room in rooms:
        pixels = get_room_pixels(room)
        values = [semantic_img[p] for p in pixels]
        label = max(Counter(values))
        semantics.append(label)
    return semantics

# 6. Create .npy file
def create_npy_file(corners, edges, semantics, filename):
    graph = {
        'corner_list_np_normalized_padding': pad_normalize(corners),
        'corner_list_np_normalized_padding_withsemantics': add_semantics(...),
        'adjacency_matrix_np_padding': create_adjacency(...),
        'global_matrix_np_padding': create_attention(...),
        'padding_mask': create_mask(...),
        'edges': flatten_edges(...),
        'edge_coords': create_edge_coords(...)
    }
    np.save(filename, graph)
```

**Reference:** See `/home/user/GSDiff/datasets/rplan-*.py` (1-10)

### 4.3 CNN Feature Extraction

**Option 1: Use RPLAN-trained CNN**

```python
# Load frozen RPLAN CNN
pretrained_encoder = BoundaryModel()
pretrained_encoder.load_state_dict(
    torch.load('outputs/structure-78-12/model_stage0_best_006700.pt')
)
pretrained_encoder.eval()
for param in pretrained_encoder.parameters():
    param.requires_grad = False

# Extract features for your data
for sample in your_dataset:
    boundary_img = render_boundary_image(sample.corners, sample.edges)
    feat_16 = pretrained_encoder(boundary_img.unsqueeze(0))
    save_features(feat_16, sample.id)
```

**Option 2: Train Custom CNN**

```python
# Modify scripts/train-CNN-autoe-final.py
# Use your preprocessed boundary images as training data
# Train boundary encoder from scratch or fine-tune RPLAN CNN

custom_cnn = BoundaryModel()  # or your architecture
# ... training loop ...
torch.save(custom_cnn.state_dict(), 'your_cnn.pt')

# Extract features
for sample in your_dataset:
    boundary_img = render_boundary_image(sample.corners, sample.edges)
    feat_16 = custom_cnn(boundary_img.unsqueeze(0))
    save_features(feat_16, sample.id)
```

**Option 3: Skip CNN Features (Unconstrained Only)**

```python
# Don't pre-compute features
# During training/inference, use zeros:
feat_16 = torch.zeros(batch_size, 1024, 16, 16)

# Pro: Saves ~500GB disk space
# Con: Only unconstrained generation (no boundary/topology guidance)
```

### 4.4 Dataset-Specific Parameters

**53 Corner Limit:**

```python
# This is a CONSTRAINT that can be changed:

# Current code (line 89 in rplang_edge_semantics_simplified_81.py):
global_attn_matrix = np.ones((53, 53), dtype=np.uint8)

# To support custom max corners:
# 1. When creating .npy files:
MAX_CORNERS = 100  # or whatever your max is
padding_mask = np.zeros((MAX_CORNERS, 1))
adjacency = np.zeros((MAX_CORNERS, MAX_CORNERS))
attention = np.ones((MAX_CORNERS, MAX_CORNERS))
# ...

# 2. Update model input size:
# In heterhouse_56_31.py or your model:
class YourModel(nn.Module):
    def forward(self, corners):  # corners: (bs, NUM_CORNERS, features)
        # NUM_CORNERS = 100 instead of 53

# 3. Update training/inference:
# In test_main.py:
corners_withsemantics_0_test_batch = torch.zeros(bs, NUM_CORNERS, 10)
```

**Semantic Dimensions:**

```python
# RPLAN: 14 dimensions (room types)
# Simplified: 7 dimensions (grouping)

# For your data, options:
# - Use 14 if you have 14 room types
# - Use 7 if you group rooms
# - Use 5 if you have: living, bedroom, kitchen, bathroom, other
# - Use custom number

# Implementation:
# 1. In preprocessing:
semantic_vec = np.zeros(YOUR_NUM_SEMANTICS)
for room in rooms_containing_corner:
    semantic_vec[room.type] += 1

# 2. In dataset loader:
# Modify column indexing to use YOUR_NUM_SEMANTICS

# 3. In model:
# Semantic embedding dimension matches YOUR_NUM_SEMANTICS
```

---

## PART 5: PRACTICAL WORKFLOWS

### WORKFLOW 1: Use Pre-trained Models (No Dataset)

**Time:** 5 minutes  
**Disk:** 200MB  
**GPU:** ~4GB VRAM

```python
import torch
from gsdiff.heterhouse_56_31 import HeterHouseModel
from gsdiff.house_nn2 import EdgeModel

# 1. Load pre-trained weights (download from README)
node_model = HeterHouseModel().cuda()
node_model.load_state_dict(torch.load('outputs/structure-1/model.pt'))
node_model.eval()

edge_model = EdgeModel().cuda()
edge_model.load_state_dict(torch.load('outputs/structure-56-16/model.pt'))
edge_model.eval()

# 2. Generate 100 floor plans (unconstrained)
batch_size = 100
with torch.no_grad():
    # Stage 1: Corners
    x = torch.randn(batch_size, 53, 10).cuda()
    feat_16 = torch.zeros(batch_size, 1024, 16, 16).cuda()
    
    for t in range(999, -1, -1):
        # Run diffusion step (pseudocode)
        output = node_model(x, attention_mask, t, feat_16)
        x = update_with_posterior_sample(output, x, t)
    
    # Stage 2: Edges
    edges = edge_model(x[:, :, :2], attention_mask, x[:, :, 2:])

# 3. Post-process and save
for i in range(batch_size):
    corners = x[i, :, :2].cpu().numpy()
    semantics = x[i, :, 2:9].cpu().numpy()
    edges_pred = edges[i].cpu().numpy()
    
    # Render or save
    render_floor_plan(corners, edges_pred, semantics, f'output_{i}.png')
```

**Limitations:**

- RPLAN data distribution only
- Unconstrained generation
- Cannot control room types/layout

### WORKFLOW 2: Train on Custom Dataset

**Time:** 2-4 weeks  
**Disk:** 500GB (CNN features) + dataset  
**GPU:** 8x A100 or equivalent

**Step 1: Prepare Data**

```bash
# Create folders
mkdir -p datasets/your_data/{train,val,test}
mkdir -p datasets/your_data_features/{train,val,test}

# Preprocess your floor plans
python your_preprocessing_script.py
# Output: 60,000+ .npy files in datasets/your_data/

# Extract CNN features (takes 1-2 weeks!)
python scripts/prerunningCNN.py
# Output: 60,000+ feature files in datasets/your_data_features/
```

**Step 2: Create Dataset Loader**

```python
# Copy datasets/rplang_edge_semantics_simplified_81.py
# Modify paths to your_data instead of rplang-v3-withsemantics
# Save as datasets/your_data_dataset.py
```

**Step 3: Train**

```bash
# Modify scripts/trainval_main_unconstrained.py:
from datasets.your_data_dataset import YourDataset

# Run training
python scripts/trainval_main_unconstrained.py
# Checkpoints saved to outputs/structure-YOUR_NAME/

# Monitor with tensorboard
tensorboard --logdir=outputs/structure-YOUR_NAME/
```

**Step 4: Train Edge Model**

```bash
python scripts/trainval_simplified_edge_unconstrained.py
# More edge model checkpoints
```

**Step 5: Generate & Evaluate**

```bash
# Modify scripts/test_main.py with your model paths
python scripts/test_main.py
# Generates 3,000 samples, computes FID/KID metrics
```

**Result:**

- Custom trained models
- Tailored to your data distribution
- Can now control room types/semantics

### WORKFLOW 3: Fine-tune Pre-trained Models

**Time:** 3-7 days  
**Disk:** 50GB (CNN features) + dataset  
**GPU:** 2x A100 or equivalent

**Step 1: Prepare Data (10,000-20,000 samples)**

```bash
# Preprocess your data
python your_preprocessing_script.py

# Extract CNN features (1-2 days)
python scripts/prerunningCNN.py
```

**Step 2: Create Dataset Loader**

```python
# Same as Workflow 2
```

**Step 3: Fine-tune**

```python
# Modified trainval_main_unconstrained.py:

# Load pre-trained weights
model = HeterHouseModel()
model.load_state_dict(torch.load('outputs/structure-1/model.pt'))
# Don't reinitialize - keep existing weights

# Optional: Lower learning rate
lr = 1e-5  # Instead of 1e-4

# Train on your data
# ... training loop ...
```

**Result:**

- Faster convergence
- Better results with less data
- Retains RPLAN knowledge while adapting to your domain

---

## PART 6: COMPARISON TABLE

| Aspect | Pre-trained Only | Train From Scratch | Fine-tune |
|--------|-----------------|-------------------|-----------|
| **Time** | 5 min | 2-4 weeks | 3-7 days |
| **GPU Needed** | Yes (4GB) | Yes (80GB) | Yes (40GB) |
| **Disk** | 200MB | 500GB+ | 50GB+ |
| **Data Needed** | None | 60,000+ samples | 10,000+ samples |
| **Domain** | RPLAN residential | Custom any type | Hybrid |
| **Customization** | None | Full | Partial |
| **Results Quality** | Good | Excellent | Very good |
| **Use Case** | Demo, quick test | Production, custom data | Custom data, fast |

---

## PART 7: TROUBLESHOOTING

### "My custom dataset has 100 corners but model expects 53"

**Solution 1: Simplify your floor plans**

```python
# Remove redundant corners
def simplify_corners(corners, edges):
    # Remove corners with only 2 neighbors (straight edges)
    simplified = []
    for corner in corners:
        neighbors = count_neighbors(corner, edges)
        if neighbors > 2:  # Keep only junctions
            simplified.append(corner)
    return simplified
```

**Solution 2: Modify model to support 100 corners**

```python
# In preprocessing:
MAX_CORNERS = 100
padding_mask = np.zeros((100, 1))  # instead of 53

# In model architecture:
corners: (bs, 100, 10)  # instead of 53, 10
adjacency: (100, 100)   # instead of 53, 53
```

### "CNN feature extraction takes forever"

**Solution 1: Parallel processing**

```python
# scripts/prerunningCNN.py
# Add multiprocessing.Pool or torch.nn.parallel.DataParallel
# Process multiple samples simultaneously
```

**Solution 2: Skip CNN features**

```python
# In training:
feat_16 = torch.zeros(batch_size, 1024, 16, 16)
# Then use unconstrained generation
```

### "Different corner counts between samples"

**Current:** Pad to fixed 53  
**Solution:** Use dynamic padding masks (already supported!)

```python
# The padding_mask handles variable sizes:
padding_mask = np.zeros((53, 1))
padding_mask[:n_real_corners] = 1

# Loss computation only uses real corners
loss = loss * padding_mask
```

### "How to handle custom room types?"

**If you have 5 room types instead of 14:**

```python
# In preprocessing:
semantic_vec = np.zeros(5)  # 5 instead of 14
for room in rooms_with_corner:
    semantic_vec[room.type_index] += 1

# In dataset loader:
# Change column indexing from 14 to 5
# corners_simplified[:, 2:7] = semantics[:, :5]

# In model:
# Semantic embedding takes input_dim=5 instead of 14
```

---

## CONCLUSION

**RPLAN Dataset Role:**

- Essential for preprocessing pipeline (10 stages)
- Creates training data (.npy files)
- NOT needed for inference/generation
- Replaceable with any custom dataset of equivalent size

**Custom Dataset Support:**

- Fully supported with proper preprocessing
- Need: graph structure + CNN features
- Can train from scratch or fine-tune
- Requires proper format (53 corners, padding masks)

**Practical Path:**

1. **Start:** Use pre-trained models (no data needed)
2. **Test:** Fine-tune on small custom dataset (10K samples)
3. **Production:** Train from scratch on large custom dataset (50K+ samples)

---
