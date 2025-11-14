# GSDiff Training Guide: Complete Documentation

## Table of Contents

1. [Training Pipeline Overview](#training-pipeline-overview)
2. [Training Scripts Location](#training-scripts-location)
3. [The 53 Corner Limit: Complete Analysis](#the-53-corner-limit-complete-analysis)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [How to Retrain on New Data](#how-to-retrain-on-new-data)
6. [How to Increase Corner Capacity](#how-to-increase-corner-capacity)
7. [Model Architecture Files](#model-architecture-files)
8. [Key Configuration Parameters](#key-configuration-parameters)

---

## Training Pipeline Overview

GSDiff uses a **multi-stage training pipeline** with three constraint variants:

### Training Stages

```
Stage 1: Node Generation (Corner Prediction)
    ↓
Stage 2: Edge Prediction (Wall Connectivity)
    ↓
Stage 3 (Optional): CNN Boundary Encoder
```

### Three Constraint Variants

1. **Unconstrained**: Pure diffusion-based generation
2. **Topology-Constrained**: Uses bubble diagram topology as conditioning
3. **Boundary-Constrained**: Uses CNN-extracted boundary features as conditioning

---

## Training Scripts Location

### Stage 1: Node Generation Training Scripts

| Script | Path | Purpose | Model Used |
|--------|------|---------|------------|
| **Unconstrained** | `/home/user/GSDiff/scripts/trainval_main_unconstrained.py` | Train node generation without constraints | `HeterHouseModel` (gsdiff/house_nn1.py) |
| **Topology** | `/home/user/GSDiff/scripts/trainval_main_topo.py` | Train with bubble diagram topology | `TopoHeterHouseModel` (gsdiff/heterhouse_80_106_2.py) |
| **Boundary** | `/home/user/GSDiff/scripts/trainval_main_boun.py` | Train with boundary image conditioning | `BoundHeterHouseModel` (gsdiff/heterhouse_81_106_3.py) |

**Key Parameters:**

- Batch size: 256
- Optimizer: AdamW (lr=1e-4, weight_decay=0 or 1e-7)
- Diffusion steps: 1000
- Training steps: 1,000,000
- Device: CUDA (cuda:0)

### Stage 2: Edge Prediction Training Scripts

| Script | Path | Purpose |
|--------|------|---------|
| **Simple Unconstrained** | `/home/user/GSDiff/scripts/trainval_simplified_edge_unconstrained.py` | Basic edge prediction for validation |
| **Main Unconstrained** | `/home/user/GSDiff/scripts/trainval_main_edge_unconstrained.py` | Advanced edge prediction (unconstrained) |
| **Simple Topology** | `/home/user/GSDiff/scripts/trainval_simplified_edge_topo.py` | Basic edge prediction for topology variant |
| **Main Topology** | `/home/user/GSDiff/scripts/trainval_main_edge_topo.py` | Advanced edge prediction (topology) |
| **Simple Boundary** | `/home/user/GSDiff/scripts/trainval_simplified_edge_boun.py` | Basic edge prediction for boundary variant |
| **Main Boundary** | `/home/user/GSDiff/scripts/trainval_main_edge_boun.py` | Advanced edge prediction with CNN features |

**Key Parameters:**

- Batch size: 4-8
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Validation interval: 100-1000 steps
- Early stopping: patience=20

### Stage 3: CNN Boundary Encoder Training

| Script | Path | Purpose |
|--------|------|---------|
| **Stage 1** | `/home/user/GSDiff/scripts/train-CNN-autoe.py` | Initial CNN autoencoder training |
| **Stage 2** | `/home/user/GSDiff/scripts/train-CNN-autoe2.py` | Intermediate CNN training |
| **Stage 3** | `/home/user/GSDiff/scripts/train-CNN-autoe-final.py` | Fine-tuning on real RPLAN data |

**Key Parameters:**

- Batch size: 16
- Optimizer: AdamW (lr=1e-5, weight_decay=0)
- Loss: L1 loss (MAE)
- Gradient clipping: max_norm=0.1

### Auxiliary Training Scripts

| Script | Path | Purpose |
|--------|------|---------|
| **Topology Transformer Stage 1** | `/home/user/GSDiff/scripts/train-TopoTransformer-autoe.py` | Train bubble diagram encoder |
| **Topology Transformer Final** | `/home/user/GSDiff/scripts/train-TopoTransformer-autoe-final.py` | Fine-tune on RPLAN bubble diagrams |
| **CNN Feature Extraction** | `/home/user/GSDiff/scripts/prerunningCNN.py` | Pre-compute CNN features (saves GPU memory) |

### LIFULL Dataset Variants (Japanese Floor Plans)

| Script | Path | Purpose |
|--------|------|---------|
| **LIFULL 56-34** | `/home/user/GSDiff/scripts/image_train_56-34-lifull.py` | Train on LIFULL dataset (variant 1) |
| **LIFULL 77-106** | `/home/user/GSDiff/scripts/image_train_77-106-lifull.py` | Train on LIFULL dataset (variant 2) |

---

## The 53 Corner Limit: Complete Analysis

### Why 53?

From `/home/user/GSDiff/datasets/rplan-process4.py` (Lines 914-916):

```python
'''padding and attn mask generating.
   1 means compute and 0 means padding.
   padding to 53 because max corner number is 53; (this is rational!! more paddings make no sense.)
   we don't use 100 because 100*100 edges are too large, about 4 times to 53*53'''
padding_to_number = 53
```

**Rationale:**

- 53 is the maximum corner count observed in the RPLAN dataset
- Using 100 would create `100×100 = 10,000` edge computations vs `53×53 = 2,809`
- Memory efficiency: 4× reduction in edge matrix size

### Where 53 is Hardcoded (50+ locations)

#### 1. Primary Definition

**File:** `/home/user/GSDiff/datasets/rplan-process4.py`

```python
# Line 916
padding_to_number = 53

# Lines 919-950: All padding arrays
corner_list_np_normalized_padding = np.zeros((53, 2), dtype=np.float64)
padding_mask = np.zeros((53, 1), dtype=np.uint8)
global_matrix_np_padding = np.zeros((53, 53), dtype=np.uint8)
adjacency_matrix_np_padding = np.zeros((53, 53), dtype=np.uint8)
```

#### 2. Dataset Classes

All dataset loaders expect pre-padded data with shape (53, ...):

**Files:**

- `/home/user/GSDiff/datasets/rplang_edge_semantics_simplified.py` (Line 58)
- `/home/user/GSDiff/datasets/rplang_edge_semantics_simplified_81.py` (Line 89)
- `/home/user/GSDiff/datasets/rplang_edge_semantics_simplified_80.py` (Line 109)
- `/home/user/GSDiff/datasets/lifull.py` (Lines 90, 110, 112, 124)
- `/home/user/GSDiff/datasets/lifull_55_100.py` (Lines 77, 94-95, 106)

**Example from lifull.py:**

```python
corners_with_semantics_pad = np.zeros((53 - len(corners_with_semantics), 15))
adjacency_matrix = np.zeros((53, 53), dtype=np.uint8)
edges = np.zeros((53, 53), dtype=np.uint8)
pdm = np.zeros((53, 1), dtype=np.uint8)
```

#### 3. Model Architecture Expectations

All models expect input tensors with shape `(batch_size, 53, features)`:

**Files:**

- `/home/user/GSDiff/gsdiff/heterhouse_80_106_2.py` (Lines 112-115)
- `/home/user/GSDiff/gsdiff/house_nn2.py` (Line 127)
- `/home/user/GSDiff/gsdiff/house_nn3.py` (Line 118)
- `/home/user/GSDiff/gsdiff/heterhouse_56_11.py` (Line 118)
- `/home/user/GSDiff/gsdiff/heterhouse_56_31.py` (Line 136)
- `/home/user/GSDiff/gsdiff/heterhouse_56_32.py` (Line 184)

**Example comment from heterhouse_80_106_2.py:**

```python
'''corners: (batch size, 53(含padding), 512(角点空间上的维数))
  global_attn_matrix: (batch size, 53, 53) 非padding（左上角）为True，padding为False'''
```

#### 4. Training Scripts

All training scripts create tensors with hardcoded 53:

**Files:**

- `/home/user/GSDiff/scripts/trainval_main_unconstrained.py` (Lines 393-394, 430, 615-627)
- `/home/user/GSDiff/scripts/trainval_main_topo.py` (Lines 419-420, 460, 657-669)
- `/home/user/GSDiff/scripts/trainval_main_boun.py` (Lines 672-684)
- `/home/user/GSDiff/scripts/test_main.py` (Lines 317, 322, 327, 330)
- `/home/user/GSDiff/scripts/test_boun.py` (Lines 345, 350, 355, 358)

**Example from trainval_main_unconstrained.py:**

```python
# Line 430
corner_number = torch.ones((batch_size,), device=device) * 53

# Lines 393-394: Diagonal masking
torch.eye(53).expand(bs, 53, 53)

# Lines 615-627: Stage 2 validation
corners_0_2 = torch.zeros((1, 53, 10), device=device)
global_attn_matrix_2 = torch.zeros((1, 53, 53), device=device)
corners_padding_mask_2 = torch.zeros((1, 53, 1), device=device)
```

#### 5. Utility Functions

**File:** `/home/user/GSDiff/gsdiff/utils.py`

```python
# Lines 275-277: Edge removal loops
for _ in range(53):
    result_edge_unpaddinged.append(
        np.array(l.cpu())[:, _ * 53:_ * 53 + results_corners_numbers[j], :]
    )

# Lines 286-288: Similar pattern
for _ in range(53):
    # Edge processing logic
```

### Summary: 53 Limit Locations

| Category | File Count | Key Locations |
|----------|------------|---------------|
| **Dataset Generation** | 1 | rplan-process4.py (line 916) |
| **Dataset Loaders** | 10+ | All rplang_*.py, lifull*.py files |
| **Model Architectures** | 13 | All heterhouse_*.py, house_nn*.py files |
| **Training Scripts** | 9 | All trainval_*.py, image_train_*.py files |
| **Testing Scripts** | 4 | test_*.py files |
| **Utility Functions** | 2 | utils.py, utils_lifull.py |
| **Evaluation Scripts** | 5 | evalmetric-*.py files |

**Total: 50+ files with hardcoded 53**

---

## Data Preprocessing Pipeline

### Complete Pipeline (Sequential Steps)

```
Raw RPLAN Images (256×256 PNGs)
    ↓
Step 1: rplan-extract.py → Corner & structure extraction
    ↓
Step 2: rplan-process1-2.py → Format conversion (adjacency list → matrix)
    ↓
Step 3: rplan-process4.py → Normalization + semantic labeling
    ↓
Step 4: rplan-process5-7.py → Add boundary features (train/val/test)
    ↓
Step 5: rplan-process8-10.py → Extract bubble diagrams (train/val/test)
    ↓
Step 6: prerunningCNN.py → Extract CNN features (optional, saves GPU memory)
    ↓
Dataset Loaders: rplang_edge_semantics_simplified_*.py
```

### Step-by-Step Preprocessing Scripts

#### Step 1: Initial Extraction

**File:** `/home/user/GSDiff/datasets/rplan-extract.py` (407 lines)

**Purpose:** Extract corners and structure from RPLAN images

**Processes:**

- Reads 256×256 PNG floor plan images
- Extracts semantic channels (14 room types)
- Generates binary images via thresholding
- Applies morphological operations (erosion)
- Detects corners:
  - **L-corner**: 2 adjacent walls (sum = 254)
  - **T-corner**: 3 adjacent walls (sum = 253)
  - **X-corner**: 4 adjacent walls (sum = 252)
- Builds adjacency lists from corner connectivity
- **Output:** `structure_graphs.npy`

**Key Functions:**

- Corner detection using 4-neighbor analysis
- Graph building from wall connectivity
- Validation and filtering of invalid floor plans

#### Step 2: Format Conversion

**Files:**

- `/home/user/GSDiff/datasets/rplan-process1.py` (31 lines)
- `/home/user/GSDiff/datasets/rplan-process2.py` (29 lines)

**Purpose:** Convert adjacency lists to matrices and reindex

**Processes:**

- Transforms corner-adjacency dictionaries to matrix form
- Converts coordinate-based adjacency to index-based

#### Step 3: Main Normalization and Semantic Labeling

**File:** `/home/user/GSDiff/datasets/rplan-process4.py` (1087 lines)

**Purpose:** Core data preprocessing - normalization and semantic assignment

**Key Functions:**

```python
# Line 54-61: Edge extraction
def extract_edges(adjacency_matrix):
    """Convert adjacency matrix to edge list"""

# Line 63-109: Semantic label extraction
def get_label(image, polygon):
    """Extract dominant semantic label from polygon region"""

# Line 111-158: Pixel extraction from polygon
def get_points_and_pixel_values_inside_polygon(polygon, image):
    """Extract all pixels within room boundary"""

# Line 160-209: Connected component counting
def count_connected_components(arr):
    """Count number of disconnected regions"""

# Line 211-260: Quadrant computation
def get_quadrant(coordinates):
    """Compute directional angle coverage for semantics"""
```

**Normalization Formula:**

```python
# Line ~920: Coordinate normalization
normalized_x = (x - 128) / 128  # Maps 0-256 to [-1, 1]
normalized_y = (y - 128) / 128
```

**Padding to 53:**

```python
# Lines 914-950
padding_to_number = 53

corner_list_np_normalized_padding = np.zeros((53, 2), dtype=np.float64)
corner_list_np_normalized_padding[:len(corners)] = normalized_corners

padding_mask = np.zeros((53, 1), dtype=np.uint8)
padding_mask[:len(corners)] = 1  # 1 = valid, 0 = padding

global_matrix_np_padding = np.zeros((53, 53), dtype=np.uint8)
global_matrix_np_padding[:len(corners), :len(corners)] = 1  # All-to-all attention

adjacency_matrix_np_padding = np.zeros((53, 53), dtype=np.uint8)
adjacency_matrix_np_padding[:len(corners), :len(corners)] = adjacency_matrix
```

**Output Dictionary Format:**

```python
{
    'file_id': int,
    'corners': list,  # Original coordinates
    'adjacency_matrix': np.array,  # Original size
    'adjacency_list': dict,
    'corner_list_np_normalized': np.array,  # Shape: (n, 2), values in [-1, 1]
    'corner_list_np_normalized_padding': np.array,  # Shape: (53, 2)
    'padding_mask': np.array,  # Shape: (53, 1)
    'global_matrix_np_padding': np.array,  # Shape: (53, 53)
    'adjacency_matrix_np_padding': np.array,  # Shape: (53, 53)
    'edges': np.array,  # Shape: (2809, 1) - flattened 53×53
    'semantics': np.array,  # Room type one-hot encodings
    'corner_list_np_normalized_padding_withsemantics': np.array  # Shape: (53, 16)
}
```

**Saved to:** `datasets/rplang-v3-withsemantics/train/`, `val/`, `test/`

#### Step 4: Add Boundary Features

**Files:**

- `/home/user/GSDiff/datasets/rplan-process5.py` (276 lines) - Training set
- `/home/user/GSDiff/datasets/rplan-process6.py` (276 lines) - Validation set
- `/home/user/GSDiff/datasets/rplan-process7.py` (277 lines) - Test set

**Purpose:** Augment graphs with boundary/wall feature maps

**Saved to:** `datasets/rplang-v3-withsemantics-withboundary/`

#### Step 5: Generate Bubble Diagrams

**Files:**

- `/home/user/GSDiff/datasets/rplan-process8.py` (226 lines) - Training set
- `/home/user/GSDiff/datasets/rplan-process9.py` (226 lines) - Validation set
- `/home/user/GSDiff/datasets/rplan-process10.py` (226 lines) - Test set

**Purpose:** Create abstract bubble diagram topology

**Output Format:**

```python
{
    'file_id': int,
    'polygons': list,  # Room boundary polygons
    'centroids': list,  # Room centers (computed via Shoelace formula)
    'semantics': np.array,  # Room type labels
    'adjacency_matrix': np.array  # Room-to-room connectivity
}
```

**Saved to:** `datasets/rplang-v3-bubble-diagram/`

#### Step 6: CNN Feature Extraction (Optional)

**File:** `/home/user/GSDiff/scripts/prerunningCNN.py` (100 lines)

**Purpose:** Pre-compute CNN boundary features to save GPU memory during training

**Process:**

```python
# Loads pre-trained BoundaryModel from outputs/structure-78-12
model = BoundaryModel()
model.load_state_dict(torch.load('pretrained_cnn.pt'))

# Extract features at multiple resolutions
for batch in dataloader:
    e3, e4, e5 = model.encoder(boundary_image)
    feat = {
        64: e3,  # 256 channels, 64×64
        32: e4,  # 512 channels, 32×32
        16: e5   # 1024 channels, 16×16
    }
    np.save(f'datasets/prerunning_cnn_featuremaps/{file_id}.npy', feat)
```

**Disk Requirement:** ~500GB for full RPLAN dataset

**Saved to:** `datasets/prerunning_cnn_featuremaps/`

### Data Normalization Functions

**File:** `/home/user/GSDiff/gsdiff/utils.py`

#### Forward Normalization

```python
# Applied in rplan-process4.py
normalized_coord = (pixel_coord - 128) / 128  # [0, 256] → [-1, 1]
```

#### Inverse Normalization Functions

```python
# Line 265-280: Basic inverse with padding removal
def inverse_normalize_remove_padding(result, results_corners_numbers, corners_padding_mask):
    """
    result: (batch, 53, 53, 2) normalized coordinates
    Returns: List of denormalized coordinates with padding removed
    """
    pixel_coord = result * 128 + 128  # [-1, 1] → [0, 256]

# Line 293-305: With semantic labels
def inverse_normalize_remove_padding_51(result, results_corners_numbers, corners_padding_mask):

# Line 307-319: Combined denormalization
def inverse_normalize_and_remove_padding(result, results_corners_numbers, corners_padding_mask):

# Line 339-351: Variable resolution support
def inverse_normalize_and_remove_padding_4testing(result, results_corners_numbers,
                                                   corners_padding_mask, resolution=256):
    """
    Supports arbitrary output resolution
    """
    pixel_coord = result * (resolution // 2) + (resolution // 2)
```

### Semantic Label Processing

#### RPLAN Semantics (7 classes)

**Bubble Diagram:** 7-class system (0-6)

- Used in topology-constrained variant
- One-hot encoding per room

#### Semantic Simplification

**File:** `/home/user/GSDiff/datasets/rplang_edge_semantics_simplified.py` (Lines 60-74)

```python
# Original: 16-dimensional semantic vector
# Simplified: 9-dimensional by aggregating similar room types

corners_withsemantics_simplified[:, 0:2] = corners[:, 0:2]  # Coordinates
corners_withsemantics_simplified[:, 2] = sum([columns 2, 6, 12])  # Wall types
corners_withsemantics_simplified[:, 3] = sum([columns 3, 7, 8, 9, 10])  # Room types
corners_withsemantics_simplified[:, 4] = sum([columns 13, 14])  # Other
# ... additional aggregations
```

#### LIFULL Semantics (12 classes)

**Files:** `lifull.py`, `lifull_55_100.py`

```python
semantic_dict = {
    'living_room': 1, 'kitchen': 2, 'bedroom': 3, 'bathroom': 4,
    'restroom': 5, 'balcony': 6, 'closet': 7, 'corridor': 8,
    'washing_room': 9, 'PS': 10, 'outside': 11, 'wall': 12,
    'no_type': 0
}
```

**Multi-hot encoding:** Each corner can have multiple semantic labels

### Data Augmentation

**File:** `/home/user/GSDiff/datasets/rplang_bubble_diagram.py`

#### Semantic Augmentation

```python
# Distribution learned from training data
q_seman = {
    0: 0.1512,  # Probability distribution
    1: 0.3833,
    2: 0.0970,
    3: 0.1106,
    4: 0.1386,
    5: 0.1193
}

# Apply during training
if self.randomize_data:
    semantics_augmented = np.dot(semantics, q_seman)
    semantics = np.random.choice(7, p=semantics_augmented)
```

#### Edge Augmentation

```python
# Edge probability distribution
q_edge = [0.6489, 0.3511]  # [no_edge, edge]

if self.randomize_data:
    edge_prob = np.dot(adjacency_matrix, q_edge)
    adjacency_matrix = np.random.choice(2, p=edge_prob)
```

---

## How to Retrain on New Data

### Prerequisites

1. **Data Format:** Floor plan images (256×256 PNG) with semantic annotations
2. **Semantic Classes:** Define room type labels
3. **Hardware:** NVIDIA GPU with CUDA support
4. **Disk Space:** ~500GB for CNN features (optional)

### Training Workflow

#### Option 1: Complete Pipeline (Recommended for New Data)

```bash
# Step 1: Extract corners from raw images
python datasets/rplan-extract.py

# Step 2: Convert to matrix format
python datasets/rplan-process1.py
python datasets/rplan-process2.py

# Step 3: Normalize and add semantics
python datasets/rplan-process4.py

# Step 4: Add boundary features
python datasets/rplan-process5.py  # Training set
python datasets/rplan-process6.py  # Validation set
python datasets/rplan-process7.py  # Test set

# Step 5: Generate bubble diagrams (if using topology variant)
python datasets/rplan-process8.py   # Training set
python datasets/rplan-process9.py   # Validation set
python datasets/rplan-process10.py  # Test set

# Step 6 (Optional): Train CNN boundary encoder
python scripts/train-CNN-autoe.py
python scripts/train-CNN-autoe2.py
python scripts/train-CNN-autoe-final.py

# Step 7 (Optional): Pre-compute CNN features
python scripts/prerunningCNN.py

# Step 8 (Optional): Train topology transformer (if using topology variant)
python scripts/train-TopoTransformer-autoe.py
python scripts/train-TopoTransformer-autoe-final.py

# Step 9: Train Stage 1 - Node Generation
# Choose one variant:
python scripts/trainval_main_unconstrained.py    # Unconstrained
# OR
python scripts/trainval_main_topo.py             # Topology-constrained
# OR
python scripts/trainval_main_boun.py             # Boundary-constrained

# Step 10: Train Stage 2 - Edge Prediction
# Choose corresponding variant:
python scripts/trainval_main_edge_unconstrained.py
# OR
python scripts/trainval_main_edge_topo.py
# OR
python scripts/trainval_main_edge_boun.py
```

#### Option 2: Quick Start (Using Existing Preprocessed Data)

If you already have NPY files in the correct format:

```bash
# Just train the models directly:

# Stage 1: Node Generation
python scripts/trainval_main_unconstrained.py

# Stage 2: Edge Prediction
python scripts/trainval_main_edge_unconstrained.py
```

### Configuration Changes Needed

#### 1. Update Dataset Paths

**In all training scripts**, modify:

```python
# Change this:
dataset_train = RPlanGEdgeSemanSimplified('train')

# To point to your data:
dataset_train = RPlanGEdgeSemanSimplified('train', data_root='/path/to/your/data')
```

#### 2. Update Device Settings

**In all scripts:**

```python
# Change from:
device = 'cuda:0'

# To your device:
device = 'cuda:0'  # or 'cuda:1', 'cpu', etc.
```

#### 3. Adjust Batch Size for Your GPU

```python
# Default:
batch_size = 256  # Requires ~24GB VRAM

# For smaller GPUs:
batch_size = 128  # ~12GB VRAM
batch_size = 64   # ~6GB VRAM
```

#### 4. Set Output Directory

**In all training scripts:**

```python
# Change this:
output_dir = os.path.join('outputs', 'structure-56-36-interval1000')

# To your preferred location:
output_dir = os.path.join('outputs', 'my_experiment_name')
```

### Monitoring Training

All scripts save:

- **Checkpoints:** `output_dir/model_stage1_XXXXXX.pt`
- **Best model:** `output_dir/model_stage1_best_XXXXXX.pt`
- **Logs:** TensorBoard logs in `output_dir/`

View logs:

```bash
tensorboard --logdir outputs/your_experiment_name
```

---

## How to Increase Corner Capacity

### Challenge: The 53 Limit is Hardcoded in 50+ Locations

To increase the maximum corners from 53 to a new value (e.g., 100), you need to:

### Step 1: Choose New Corner Limit

**Considerations:**

- Memory usage scales as O(N²) for edge matrices
- 53 → 100: Memory increase = (100²/53²) = 3.56×
- 53 → 150: Memory increase = (150²/53²) = 8.0×

**Edge matrix sizes:**

- 53 corners: 2,809 edges
- 100 corners: 10,000 edges (3.56× larger)
- 150 corners: 22,500 edges (8.0× larger)

### Step 2: Regenerate All Training Data

**Critical:** You must regenerate all NPY files with new padding size.

**File:** `/home/user/GSDiff/datasets/rplan-process4.py`

```python
# Line 916: CHANGE THIS
padding_to_number = 100  # Changed from 53

# Lines 919-950: All arrays will automatically use new size
corner_list_np_normalized_padding = np.zeros((padding_to_number, 2))
padding_mask = np.zeros((padding_to_number, 1))
global_matrix_np_padding = np.zeros((padding_to_number, padding_to_number))
adjacency_matrix_np_padding = np.zeros((padding_to_number, padding_to_number))
```

**Then regenerate:**

```bash
python datasets/rplan-process4.py   # Regenerate all train/val/test
python datasets/rplan-process5.py
python datasets/rplan-process6.py
python datasets/rplan-process7.py
```

### Step 3: Update All Dataset Loaders

**Files to modify (10+ files):**

#### `/home/user/GSDiff/datasets/lifull.py`

```python
# Lines 90, 110, 112, 124: Change from 53 to 100
corners_with_semantics_pad = np.zeros((100 - len(corners_with_semantics), 15))
adjacency_matrix = np.zeros((100, 100), dtype=np.uint8)
edges = np.zeros((100, 100), dtype=np.uint8)
pdm = np.zeros((100, 1), dtype=np.uint8)
```

#### `/home/user/GSDiff/datasets/lifull_55_100.py`

```python
# Lines 77, 94-95, 106: Same changes
```

#### All `/home/user/GSDiff/datasets/rplang_edge_semantics_simplified*.py` files

```python
# Update shape comments from (53, 16) to (100, 16)
# Example from line 58:
'''coords_withsemantics, (100, 16)'''  # Changed from 53
```

### Step 4: Update All Model Architecture Files

**Files to modify (13 files):**

All model files expect input shape `(batch, 53, ...)` - change to `(batch, 100, ...)`

#### Example: `/home/user/GSDiff/gsdiff/heterhouse_80_106_2.py`

```python
# Lines 112-115: Update comments
'''corners: (batch size, 100(含padding), 512(角点空间上的维数))
  global_attn_matrix: (batch size, 100, 100) 非padding（左上角）为True，padding为False'''
```

**Files to update:**

- `gsdiff/house_nn2.py`
- `gsdiff/house_nn3.py`
- `gsdiff/heterhouse_56_11.py`
- `gsdiff/heterhouse_56_31.py`
- `gsdiff/heterhouse_56_32.py`
- `gsdiff/heterhouse_56_13_lifull.py`
- And all other model files

### Step 5: Update All Training Scripts

**Files to modify (9+ files):**

#### Example: `/home/user/GSDiff/scripts/trainval_main_unconstrained.py`

```python
# Line 393-394: Change torch.eye(53) to torch.eye(100)
diagonal_mask = torch.eye(100).expand(bs, 100, 100)

# Line 430: Change 53 to 100
corner_number = torch.ones((batch_size,), device=device) * 100

# Lines 615-627: Update all tensor shapes
corners_0_2 = torch.zeros((1, 100, 10), device=device)
global_attn_matrix_2 = torch.zeros((1, 100, 100), device=device)
corners_padding_mask_2 = torch.zeros((1, 100, 1), device=device)
```

**Apply same changes to:**

- `scripts/trainval_main_topo.py`
- `scripts/trainval_main_boun.py`
- `scripts/test_main.py`
- `scripts/test_boun.py`
- `scripts/test_topo.py`
- `scripts/image_train_56-34-lifull.py`
- `scripts/image_train_77-106-lifull.py`

### Step 6: Update Utility Functions

**File:** `/home/user/GSDiff/gsdiff/utils.py`

```python
# Lines 275-277: Change range(53) to range(100)
for _ in range(100):
    result_edge_unpaddinged.append(
        np.array(l.cpu())[:, _ * 100:_ * 100 + results_corners_numbers[j], :]
    )

# Lines 286-288: Same change
for _ in range(100):
    # Edge processing logic
```

### Step 7: Update Evaluation Scripts

**Files to modify (5 files):**

#### Example: `/home/user/GSDiff/evalmetric-no-constrain-align-metrics.py`

```python
# Line 32: Update shape
pred_xstart_cs = torch.zeros((1, 100, 10))

# Lines 61, 63: Update eye matrix
torch.eye(100).expand(bs, 100, 100)
```

### Step 8: Search and Replace

**Automated approach using grep + sed:**

```bash
# Find all occurrences of "53" in Python files
grep -r "53" --include="*.py" | grep -v ".git" | grep -v "__pycache__"

# Carefully review each occurrence and determine if it should be changed

# Example: Replace in a specific file
sed -i 's/padding_to_number = 53/padding_to_number = 100/g' datasets/rplan-process4.py
```

**⚠️ WARNING:** Not all "53" should be changed! Carefully review each occurrence.

### Complete File List Requiring Changes

#### Critical Files (Must Change)

1. **Data Generation:**
   - `datasets/rplan-process4.py` (line 916)

2. **Dataset Loaders (10+ files):**
   - `datasets/rplang_edge_semantics_simplified.py`
   - `datasets/rplang_edge_semantics_simplified_81.py`
   - `datasets/rplang_edge_semantics_simplified_80.py`
   - `datasets/rplang_edge_semantics_simplified_56_32.py`
   - `datasets/rplang_edge_semantics_simplified_55_106.py`
   - `datasets/rplang_edge_semantics_simplified_78_10.py`
   - `datasets/rplang_edge_semantics_simplified_78_11.py`
   - `datasets/lifull.py`
   - `datasets/lifull_55_100.py`

3. **Model Architectures (13 files):**
   - `gsdiff/house_nn1.py`
   - `gsdiff/house_nn2.py`
   - `gsdiff/house_nn3.py`
   - `gsdiff/heterhouse_80_106_2.py`
   - `gsdiff/heterhouse_81_106_3.py`
   - `gsdiff/heterhouse_56_11.py`
   - `gsdiff/heterhouse_56_31.py`
   - `gsdiff/heterhouse_56_32.py`
   - `gsdiff/heterhouse_56_13_lifull.py`
   - `gsdiff/heterhouse_56_11_lifull.py`
   - `gsdiff/heterhouse_75_106_lifull.py`
   - `gsdiff/boundary_78_10.py`
   - `gsdiff/bubble_diagram_57_9.py`

4. **Training Scripts (9 files):**
   - `scripts/trainval_main_unconstrained.py`
   - `scripts/trainval_main_topo.py`
   - `scripts/trainval_main_boun.py`
   - `scripts/trainval_main_edge_unconstrained.py`
   - `scripts/trainval_main_edge_topo.py`
   - `scripts/trainval_main_edge_boun.py`
   - `scripts/trainval_simplified_edge_*.py` (3 files)

5. **Testing Scripts (4 files):**
   - `scripts/test_main.py`
   - `scripts/test_boun.py`
   - `scripts/test_topo.py`
   - `scripts/test-final-lifull1.py`

6. **Utility Functions (2 files):**
   - `gsdiff/utils.py`
   - `gsdiff/utils_lifull.py`

7. **Evaluation Scripts (5 files):**
   - `evalmetric-no-constrain-align-metrics.py`
   - `evalmetric-boun-constrain-fid-kid.py`
   - `evalmetric-no-constrain-fid-kid.py`
   - `evalmetric-no-constrain-geometry-topological-metrics.py`
   - `evalmetric-topoconstrain-ged-roomnumber.py`

**Total: 50+ files**

### Verification Checklist

After making changes:

```bash
# 1. Search for any remaining "53" that should be changed
grep -r "\b53\b" --include="*.py" gsdiff/ datasets/ scripts/

# 2. Check tensor shape comments
grep -r "53," --include="*.py"

# 3. Verify data regeneration worked
python -c "import numpy as np; d=np.load('datasets/rplang-v3-withsemantics/train/0.npy', allow_pickle=True).item(); print(d['padding_mask'].shape)"
# Should output: (100, 1)

# 4. Run a small training test
python scripts/trainval_main_unconstrained.py  # Will error if shapes are inconsistent
```

---

## Model Architecture Files

### Node Generation Models

| Model File | Class | Purpose | Input Shape |
|------------|-------|---------|-------------|
| `gsdiff/house_nn1.py` | `HeterHouseModel` | Unconstrained node generation | (batch, 53, 256) |
| `gsdiff/heterhouse_80_106_2.py` | `TopoHeterHouseModel` | Topology-constrained node generation | (batch, 53, 256) |
| `gsdiff/heterhouse_81_106_3.py` | `BoundHeterHouseModel` | Boundary-constrained node generation | (batch, 53, 256) |
| `gsdiff/heterhouse_75_106_lifull.py` | `HeterHouseModel` | LIFULL dataset variant | (batch, 53, 256) |

**Architecture Components:**

- **d_model**: 256 (embedding dimension)
- **Transformer layers**: 6
- **Attention heads**: 4
- **Time embedding**: Sinusoidal for diffusion timestep
- **Semantic embedding**: 8 → 256 dimensions
- **Output**: Node coordinates (x, y) + semantics

### Edge Prediction Models

| Model File | Class | Purpose | Input Shape |
|------------|-------|---------|-------------|
| `gsdiff/house_nn3.py` | `EdgeModel` | Basic edge prediction | (batch, 53, features) |
| `gsdiff/heterhouse_56_11.py` | `EdgeModel` | Advanced edge prediction | (batch, 53, features) |
| `gsdiff/heterhouse_56_32.py` | `BoundEdgeModel` | Edge prediction with CNN features | (batch, 53, features) |
| `gsdiff/heterhouse_56_11_lifull.py` | `EdgeModel` | LIFULL edge prediction | (batch, 53, features) |

**Architecture Components:**

- **MultiHeadAttention**: Global attention mechanism
- **Adaptive sampling**: Smart edge candidate selection
- **Padding mask support**: Handles variable-length sequences
- **Output**: Binary edge prediction (53, 53) matrix

### Boundary Encoder Models

| Model File | Class | Purpose |
|------------|-------|---------|
| `gsdiff/boundary_78_10.py` | `BoundaryModel` | CNN autoencoder for boundary images |

**Architecture:**

- **Encoder**: ResNet-style with skip connections
- **Feature scales**: 64×64 (256ch), 32×32 (512ch), 16×16 (1024ch)
- **Decoder**: Upsampling with Conv2DTranspose
- **Output**: Reconstructed 256×256 boundary image

### Topology Models

| Model File | Class | Purpose |
|------------|-------|---------|
| `gsdiff/bubble_diagram_57_9.py` | `TopoGraphModel` | Bubble diagram transformer |

**Architecture:**

- **Node encoding**: Graph nodes representing rooms
- **Edge encoding**: Room adjacency
- **Output**: Semantic labels + adjacency predictions

---

## Key Configuration Parameters

### Universal Parameters (All Training Scripts)

```python
# Device
device = 'cuda:0'  # or 'cuda:1', 'cpu'

# Diffusion settings
diffusion_steps = 1000  # Number of denoising steps
batch_size = 256        # Stage 1 training
batch_size_edge = 8     # Stage 2 training

# Optimizer
optimizer = AdamW
learning_rate = 1e-4    # Stage 1
learning_rate = 1e-4    # Stage 2
weight_decay = 0        # Stage 1 (or 1e-7 for boundary variant)
weight_decay = 1e-5     # Stage 2

# Training duration
total_steps = 1000000   # Stage 1
interval = 1000         # Validation/checkpoint interval (Stage 1)
interval = 100          # Validation interval (Stage 2)

# Early stopping (Stage 2 only)
lr_reduce_patience = 5
stop_patience = 20

# Gradient clipping
max_norm = 0.1  # CNN
max_norm = 1.0  # Transformers

# Data augmentation
merge_points = True     # Merge very close points
clamp_trick_training = True  # Clamp coordinates during training
align_points = True     # Align points for generation
```

### Stage 1: Node Generation

**Unconstrained:**

```python
# trainval_main_unconstrained.py
batch_size = 256
lr = 1e-4
weight_decay = 0
merge_points = False
```

**Topology:**

```python
# trainval_main_topo.py
batch_size = 256
lr = 1e-4
weight_decay = 0
merge_points = True
# Additional: Loads bubble diagram data
```

**Boundary:**

```python
# trainval_main_boun.py
batch_size = 256
lr = 1e-4
weight_decay = 1e-7  # Note: Small weight decay for boundary variant
merge_points = True
# Additional: Loads CNN features
```

### Stage 2: Edge Prediction

```python
# All edge prediction scripts
batch_size = 4  # or 8
lr = 1e-4
weight_decay = 1e-5
interval = 1000  # Main variants
interval = 100   # Simplified variants
```

### CNN Training

```python
# train-CNN-autoe*.py
batch_size = 16
lr = 1e-5
weight_decay = 0
loss = L1Loss  # MAE
gradient_clip = 0.1
```

### Transformer Training (Topology)

```python
# train-TopoTransformer-autoe*.py
batch_size = 256
lr = 1e-4
weight_decay = 0
num_workers = 8
gradient_clip = 1.0
# Dual losses: semantics + edges
```

### Dataset Paths

All datasets expect this structure:

```
datasets/
├── rplang-v3-withsemantics/
│   ├── train/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── val/
│   └── test/
├── rplang-v3-withsemantics-withboundary/
│   ├── train/
│   ├── val/
│   └── test/
├── rplang-v3-bubble-diagram/
│   ├── train/
│   ├── val/
│   └── test/
└── prerunning_cnn_featuremaps/
    ├── 0.npy
    ├── 1.npy
    └── ...
```

### Output Directory Structure

```
outputs/
└── experiment_name/
    ├── model_stage1_000000.pt
    ├── model_stage1_001000.pt
    ├── ...
    ├── model_stage1_best_065000.pt
    └── tensorboard_logs/
```

---

## Summary

### To Retrain on New Data

1. **Prepare data:** Run preprocessing pipeline (rplan-extract.py → rplan-process*.py)
2. **Choose variant:** Unconstrained, Topology, or Boundary
3. **Train Stage 1:** Node generation (trainval_main_*.py)
4. **Train Stage 2:** Edge prediction (trainval_main_edge_*.py)

### To Change Corner Capacity

1. **Update rplan-process4.py:** Change `padding_to_number = 53` to new value
2. **Regenerate all data:** Run rplan-process4.py through rplan-process10.py
3. **Update 50+ files:** Change all hardcoded "53" to new value
4. **Verify:** Check all tensor shapes and run tests

### Key Files to Remember

- **Main training:** `scripts/trainval_main_*.py`
- **Data generation:** `datasets/rplan-process4.py` (defines padding size)
- **Model architectures:** `gsdiff/heterhouse_*.py`
- **Utilities:** `gsdiff/utils.py`

---

**All file paths are absolute for reference.**

**Created:** 2025-11-13
**Repository:** GSDiff Floor Plan Generation
