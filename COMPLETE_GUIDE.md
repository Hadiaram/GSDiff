# GSDiff: Complete Guide to Working with the Repository

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Repository Structure](#repository-structure)
4. [Understanding Data Formats](#understanding-data-formats)
5. [Dataset Requirements](#dataset-requirements)
6. [Model Architectures](#model-architectures)
7. [Training Pipeline](#training-pipeline)
8. [Testing and Evaluation](#testing-and-evaluation)
9. [Working with Custom Data](#working-with-custom-data)
10. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
11. [Branches and Development](#branches-and-development)
12. [Advanced Topics](#advanced-topics)

---

## Project Overview

**GSDiff** is the official implementation of the AAAI 2025 paper: "GSDiff: Synthesizing Vector Floor Plans via Geometry-enhanced Structural Graph Generation"

### What Does This Project Do?

GSDiff generates vector-based floor plans (architectural drawings) using a diffusion-based generative model. It can operate in three modes:

1. **Unconstrained Generation**: Freely generates floor plans from random noise
2. **Topology-Constrained Generation**: Generates plans respecting room adjacency relationships (bubble diagrams)
3. **Boundary-Constrained Generation**: Generates plans within specified building boundaries

### Key Features

- **Two-Stage Generation**:
  - Stage 1: Generate corners (vertices) with coordinates and room semantics
  - Stage 2: Generate edges connecting corners to form complete floor plans
- **Diffusion-Based**: Uses 1000-step diffusion process with cosine scheduling
- **Transformer Architecture**: 24-layer transformer with 4 attention heads
- **Graph Representation**: Floor plans as structural graphs (nodes = corners, edges = walls)

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (32GB+ recommended for training)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd GSDiff
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt
```

### Core Dependencies

```
torch==2.0.1
torchvision==0.15.2
numpy==1.26.0
opencv-python==4.9.0.80
pillow==10.0.1
matplotlib==3.9.2
networkx==3.1
pytorch-fid==0.3.0
scikit-image==0.24.0
scikit-learn==1.4.1.post1
scipy==1.10.1
shapely==2.0.6
tqdm==4.65.0
tensorboardx==2.6.2.2
```

### Step 3: Set Up Directory Structure

```bash
mkdir -p datasets
mkdir -p scripts/outputs
mkdir -p scripts/test_outputs
```

---

## Repository Structure

```
GSDiff/
├── datasets/                       # Data loaders and preprocessing
│   ├── __init__.py
│   ├── path_utils.py              # Path utility functions
│   ├── rplang_edge_semantics_simplified.py          # Main RPLAN loader
│   ├── rplang_edge_semantics_simplified_81.py       # RPLAN with CNN features
│   ├── rplang_bubble_diagram.py                     # Topology loader
│   ├── lifull.py                                    # LIFULL dataset loader
│   ├── rplan-extract.py                             # Data extraction
│   ├── rplan-process[1-10].py                       # Data processing pipeline
│   ├── rplang-v3-withsemantics/                     # Processed graph data
│   │   ├── train/                 # 65,763 .npy files
│   │   ├── val/                   # 3,000 .npy files
│   │   └── test/                  # 3,000 .npy files
│   ├── rplang-v3-withsemantics-withboundary/        # Graph + boundary
│   ├── rplang-v3-withsemantics-withboundary-v2/     # Boundary adjacency
│   ├── rplang-v3-bubble-diagram/                    # Topology graphs
│   └── prerunning_cnn_featuremaps/                  # Pre-computed CNN features
│
├── gsdiff/                         # Model architectures
│   ├── __init__.py
│   ├── house_nn1.py               # Node generation model (HeterHouseModel)
│   ├── house_nn2.py               # Edge generation model
│   ├── house_nn3.py               # Basic transformer
│   ├── heterhouse_56_11.py        # EdgeModel architecture
│   ├── heterhouse_56_31.py        # Variant models
│   ├── heterhouse_56_32.py        # BoundEdgeModel
│   ├── heterhouse_81_106_3.py     # BoundHeterHouseModel (large variant)
│   ├── boundary_78_10.py          # Boundary CNN autoencoder
│   ├── bubble_diagram_57_9.py     # Topology transformer (TopoGraphModel)
│   └── utils.py                   # Utility functions (113KB+)
│
├── scripts/                        # Training and testing scripts
│   ├── __init__.py
│   ├── trainval_main_unconstrained.py              # Train without constraints
│   ├── trainval_main_topo.py                       # Train with topology
│   ├── trainval_main_boun.py                       # Train with boundary
│   ├── trainval_main_edge_*.py                     # Edge model training
│   ├── train-CNN-autoe-final.py                    # Boundary CNN training
│   ├── train-TopoTransformer-autoe-final.py        # Topology transformer training
│   ├── test_main.py                                # Test unconstrained
│   ├── test_topo.py                                # Test topology-constrained
│   ├── test_boun.py                                # Test boundary-constrained
│   ├── outputs/                   # Training outputs and model checkpoints
│   ├── test_outputs/              # Test results and rendered images
│   └── metrics/                   # Evaluation metrics
│       ├── fid.py                 # Frechet Inception Distance
│       └── kid.py                 # Kernel Inception Distance
│
├── evalmetric-*.py                # Evaluation scripts (at root)
│   ├── evalmetric-no-constrain-fid-kid.py
│   ├── evalmetric-no-constrain-geometry-topological-metrics.py
│   ├── evalmetric-topoconstrain-ged-roomnumber.py
│   ├── evalmetric-boun-constrain-fid-kid.py
│   └── evalmetric-no-constrain-align-metrics.py
│
├── requirements.txt               # Core dependencies
├── requirements_full.txt          # Full conda environment
├── README.md                      # Basic setup instructions
├── PATH_FIXES_SUMMARY.md          # Path configuration documentation
└── COMPLETE_GUIDE.md              # This file
```

---

## Understanding Data Formats

### The Core Problem: What Are We Working With?

Floor plans in this project are represented as **structural graphs**, not images:

- **Nodes (Corners)**: Junction points where walls meet
- **Edges (Walls)**: Connections between corners
- **Semantics**: Room types associated with corners/faces
- **Faces (Rooms)**: Closed polygons formed by edges

### Data Flow Overview

```
Raw RPLAN Images
    ↓ (rplan-extract.py)
Semantic Segmentation + Corner Detection
    ↓ (rplan-process1-4.py)
Structural Graphs (.npy files)
    ↓ (Data Loaders)
Padded Tensors (53 corners max)
    ↓ (Model Training/Testing)
Generated Floor Plans
    ↓ (Cycle Basis Extraction + Rendering)
Final Images (512×512 RGB)
```

---

## Dataset Requirements

### Critical: Two Types of Data Required

**IMPORTANT**: The test scripts require **TWO** types of data files:

1. **Graph Data Files** - Structural graph information (corners, edges, semantics)
2. **CNN Feature Map Files** - Pre-computed visual features from images

Both must exist with matching filenames for the model to work!

### 1. Graph Data Format (`.npy` files in `rplang-v3-withsemantics/`)

Each `.npy` file is a Python dictionary saved with `numpy`:

```python
# Load example:
data = np.load('datasets/rplang-v3-withsemantics/test/0.npy', allow_pickle=True).item()

# Expected structure:
{
    'corner_list_np_normalized_padding_withsemantics': ndarray, shape (53, 16),
    'global_matrix_np_padding': ndarray, shape (53, 53), dtype=bool,
    'padding_mask': ndarray, shape (53, 1), dtype=uint8,
    'edges': ndarray, shape (2809, 1), dtype=int/float
}
```

#### Field Descriptions:

**`corner_list_np_normalized_padding_withsemantics` (53, 16):**
- **Columns 0-1**: Normalized X, Y coordinates
  - Range: [-1, 1] (normalized from 0-256 pixel space)
  - Formula: `(pixel - 128) / 128`
- **Columns 2-15**: Semantic one-hot encoding (14 channels)
  - Room type indicators for each corner
  - Aggregated to 7 channels in simplified loaders:
    - Channels [2,6,12] → wall features
    - Channels [3,7,8,9,10] → room features
    - Channels [13,14] → other features
    - Channels [4,5,11,15] → direct semantic classes
- **Padding**: Unused corners filled with zeros, identified by padding_mask

**`global_matrix_np_padding` (53, 53):**
- Boolean adjacency matrix
- `True` for all valid corner pairs (used for attention mask)
- `False` for padded corners

**`padding_mask` (53, 1):**
- `1` for valid corners
- `0` for padded (unused) corners
- Most floor plans have 15-35 corners, padded to 53

**`edges` (2809, 1):**
- Flattened adjacency matrix (53 × 53 = 2809)
- Binary values: `1` = edge exists, `0` = no edge
- Represents wall connections between corners

### 2. CNN Feature Map Format (`.npy` files in `prerunning_cnn_featuremaps/`)

**Critical for Testing**: These files MUST exist with matching names!

```python
# Load example:
features = np.load('datasets/prerunning_cnn_featuremaps/0.npy', allow_pickle=True).item()

# Expected structure:
{
    16: [ndarray, shape (1024, 16, 16), dtype=float32],
    # Optionally:
    32: [ndarray, shape (512, 32, 32)],
    64: [ndarray, shape (256, 64, 64)]
}
```

#### What Are These?

These are **pre-computed CNN features** extracted from floor plan images using a pre-trained ResNet or similar CNN. They serve as **conditioning information** to guide the diffusion model during generation.

**Why are they needed?**
- Line 230 in `test_boun.py`: Model takes `feat_16_test_batch` as input
- Line 363: EdgeModel also requires CNN features
- They provide visual/spatial context from the original images

#### How to Create Them:

**Option 1: Use Pre-existing Features** (if available)
```python
# Copy from existing dataset
cp datasets/prerunning_cnn_featuremaps/0.npy your_target_directory/
```

**Option 2: Create Dummy Features** (for testing without real images)
```python
import numpy as np

# Create dummy CNN feature map
dummy_features = {
    16: [np.zeros((1024, 16, 16), dtype=np.float32)]
}

# Save it
np.save('datasets/prerunning_cnn_featuremaps/0.npy', dummy_features)
```

**Option 3: Extract from Real Images** (if you have floor plan images)
- Use `train-CNN-autoe-final.py` to train boundary CNN
- Extract features using the encoder part of the trained CNN
- Save features in the required format

### 3. Boundary Data Format (for boundary-constrained generation)

**File**: `rplang-v3-withsemantics-withboundary/`

Adds one additional key to the standard format:

```python
{
    # ... all standard keys from section 1 ...
    'boundary_vertex_indices': ndarray, shape (53, 2), dtype=uint8
    # Values: [1, 1] if corner is on building boundary
    #         [0, 0] if corner is internal
}
```

**File**: `rplang-v3-withsemantics-withboundary-v2/`

```python
{
    'boundary_adjacency_matrix': ndarray, shape (53, 53), dtype=bool
    # Adjacency matrix specifically for boundary polygon edges
}
```

### 4. Topology Data Format (for topology-constrained generation)

**File**: `rplang-v3-bubble-diagram/`

Different structure for room-level topology:

```python
{
    'semantics': ndarray, shape (n,), dtype=int,  # Room type indices (0-6)
    'adjacency_matrix': ndarray, shape (n, n), dtype=int  # Room adjacency
}
```

Where `n` = number of rooms (typically 4-8, padded to 8 in loader)

**Room Semantic Classes:**
- 0: LivingRoom
- 1: Bedroom
- 2: Kitchen
- 3: Bathroom
- 4: Balcony
- 5: Others
- 6: Wall/Boundary

---

## Model Architectures

### Stage 1: Node Generation

**File**: `gsdiff/house_nn1.py` - Class: `HeterHouseModel`

**Purpose**: Generate corner coordinates and semantics from noise

**Architecture Overview**:
```
Input: noisy corners + attention mask + timestep
    ↓
Sinusoidal Embeddings (coordinates, time)
    ↓
24 × Transformer Layers
    ↓
Output Heads: coordinates (2D) + semantics (7D)
```

**Key Parameters**:
- `d_model = 256`: Embedding dimension
- `num_heads = 4`: Multi-head attention
- `num_layers = 24`: Transformer depth
- `hidden_dim = 1024`: Feedforward inner dimension

**Input Shape**: `(batch_size, 53, 9)`
- 2 channels: X, Y coordinates (normalized)
- 7 channels: Room semantic encoding

**Output Shape**:
- `output_corners`: `(batch_size, 53, 2)` - Predicted coordinates
- `output_semantics`: `(batch_size, 53, 7)` - Predicted semantics

### Stage 2: Edge Generation

**File**: `gsdiff/heterhouse_56_11.py` - Class: `EdgeModel`

**Purpose**: Predict which corners should be connected by walls

**Architecture Overview**:
```
Input: corners + semantics + attention mask
    ↓
Edge Pair Embeddings
    ↓
Transformer with Adjacency Attention
    ↓
Output: Edge probabilities for all pairs
```

**Input Shape**:
- `corners`: `(batch_size, 53, 2)`
- `semantics`: `(batch_size, 53, 7)`

**Output Shape**: `(batch_size, 2809, 2)` - Binary classification for each edge
- 2809 = 53 × 53 (all possible corner pairs)
- 2 classes: [no_edge, edge]

### Auxiliary Models

**Boundary CNN** (`gsdiff/boundary_78_10.py`)
- ResNet-style encoder-decoder
- Input: 256×256 RGB boundary images
- Output: Compressed boundary representations

**Topology Transformer** (`gsdiff/bubble_diagram_57_9.py`)
- Operates on room-level graphs
- Input: Room semantics + adjacency
- Output: Generated topology constraints

---

## Training Pipeline

### Training Configuration

**Typical Hyperparameters** (from `trainval_main_unconstrained.py`):

```python
diffusion_steps = 1000          # Number of diffusion steps
lr = 1e-4                       # Learning rate
weight_decay = 0                # No weight decay
total_steps = 1000000           # Total training steps
batch_size = 256                # Training batch size
batch_size_val = 3000           # Validation batch size
device = 'cuda:0'               # GPU device
```

### Diffusion Schedule

**Cosine Beta Schedule**:

```python
alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
betas = []
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
```

This creates a smooth noise schedule from clean data (t=0) to pure noise (t=999).

### Training Process

1. **Load Data**:
   ```python
   dataset_train = RPlanGEdgeSemanSimplified('train')
   dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
   ```

2. **Forward Diffusion** (add noise):
   ```python
   # Sample random timestep
   t = torch.randint(0, diffusion_steps, (batch_size,))

   # Add noise according to schedule
   noise = torch.randn_like(corners_withsemantics)
   noisy_corners = sqrt_alphas_cumprod[t] * corners_withsemantics + \
                   sqrt_one_minus_alphas_cumprod[t] * noise
   ```

3. **Model Prediction**:
   ```python
   predicted_noise = model(noisy_corners, attention_mask, t)
   ```

4. **Loss Calculation**:
   ```python
   loss = F.mse_loss(predicted_noise, noise)
   ```

5. **Backpropagation**:
   ```python
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   ```

### Validation

Every N steps, generate samples and compute FID/KID:

1. Run full reverse diffusion (1000 steps)
2. Render generated floor plans
3. Compare with ground truth using FID/KID metrics

---

## Testing and Evaluation

### Running Tests

**Unconstrained Generation**:
```bash
cd scripts
python test_main.py
```

**Topology-Constrained**:
```bash
python test_topo.py
```

**Boundary-Constrained**:
```bash
python test_boun.py
```

### What Tests Do

1. **Load Test Dataset**:
   ```python
   dataset_test = RPlanGEdgeSemanSimplified_81('test')
   # Loads 3000 samples (or your custom test size)
   ```

2. **Render Ground Truth**:
   - Extract cycle basis (closed polygons)
   - Color rooms by semantic type
   - Draw walls and corners
   - Save to `test_outputs/test_gt/`

3. **Generate Samples**:
   - Start from random noise
   - Run reverse diffusion (999→0)
   - Apply post-processing (merge points, alignment)
   - Run edge model for final connectivity

4. **Render Predictions**:
   - Same rendering pipeline as ground truth
   - Save to `test_outputs/test_<model_name>/`

5. **Compute Metrics**:
   - **FID**: Frechet Inception Distance (image quality)
   - **KID**: Kernel Inception Distance (image quality)
   - **GED**: Graph Edit Distance (structural similarity)
   - Room-type statistics

### Output Structure

```
scripts/test_outputs/
└── AP-1/                          # Run name
    ├── test_gt/                   # Ground truth renderings
    │   ├── test_gt_0.png
    │   ├── test_gt_1.png
    │   └── ...
    ├── test_model1000000/         # Predictions from model checkpoint
    │   ├── test_pred_0.png
    │   ├── test_pred_1.png
    │   └── ...
    ├── test_corner_step0_model1000000/  # Corner visualizations
    └── test_metrics.npy           # Saved metrics
```

### Evaluation Metrics

**FID (Frechet Inception Distance)**:
- Measures distributional similarity of generated vs. real images
- Lower is better
- Typical range: 10-50 for good results

**KID (Kernel Inception Distance)**:
- Alternative to FID using kernel methods
- More stable for small sample sizes
- Lower is better

**GED (Graph Edit Distance)**:
- Measures structural similarity of graphs
- Counts node/edge insertions, deletions, substitutions
- Lower is better

Run evaluation scripts:
```bash
# Average FID/KID across multiple runs
python evalmetric-no-constrain-fid-kid.py

# Geometric and topological metrics
python evalmetric-no-constrain-geometry-topological-metrics.py

# Topology-constrained GED
python evalmetric-topoconstrain-ged-roomnumber.py
```

---

## Working with Custom Data

### Scenario: Converting JSON Floor Plans to Compatible Format

This section addresses the common use case of having floor plan data in JSON format and needing to convert it for use with GSDiff.

#### Step 1: Understand Your JSON Structure

Typical JSON structure:
```json
{
  "images": [
    {"id": 0, "width": 256, "height": 256}
  ],
  "annotations": [
    {
      "image_id": 0,
      "point": [128, 64],
      "semantic": ["living_room", "corridor"]
    }
  ]
}
```

#### Step 2: Convert to Required NumPy Format

Create a conversion script:

```python
import json
import numpy as np

def convert_json_to_npy(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for image in data['images']:
        image_id = image['id']

        # Extract annotations for this image
        annotations = [a for a in data['annotations'] if a['image_id'] == image_id]

        # Initialize arrays
        num_corners = len(annotations)
        corners = np.zeros((53, 16))  # Padded to 53
        padding_mask = np.zeros((53, 1), dtype=np.uint8)

        # Process each corner
        for i, ann in enumerate(annotations):
            # Normalize coordinates to [-1, 1]
            x_norm = (ann['point'][0] - 128) / 128
            y_norm = (ann['point'][1] - 128) / 128
            corners[i, 0] = x_norm
            corners[i, 1] = y_norm

            # Convert semantics to multi-hot encoding
            # (implement based on your semantic classes)
            semantic_vector = convert_semantics(ann['semantic'])
            corners[i, 2:16] = semantic_vector

            padding_mask[i, 0] = 1

        # Create edge adjacency (if available in JSON)
        edges = np.zeros((2809, 1))
        # ... populate edges from your data ...

        # Create global attention matrix
        global_matrix = np.zeros((53, 53), dtype=bool)
        global_matrix[:num_corners, :num_corners] = True

        # Save as numpy file
        output_data = {
            'corner_list_np_normalized_padding_withsemantics': corners,
            'global_matrix_np_padding': global_matrix,
            'padding_mask': padding_mask,
            'edges': edges
        }

        np.save(f'{output_dir}/{image_id}.npy', output_data)

# Usage
convert_json_to_npy('instances_train.json', 'datasets/rplang-v3-withsemantics/train/')
```

#### Step 3: Create Matching CNN Feature Maps

**Option A: Dummy Features for Testing**

```python
import numpy as np
import os

def create_dummy_cnn_features(num_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_files):
        dummy_features = {
            16: [np.random.randn(1024, 16, 16).astype(np.float32)]
        }
        np.save(f'{output_dir}/{i}.npy', dummy_features)

# Create dummy features for your test files
create_dummy_cnn_features(5, 'datasets/prerunning_cnn_featuremaps/')
```

**Option B: Extract from Images** (if you have images)

```python
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

def extract_cnn_features(image_path, output_path):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    # Remove final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.eval()

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = model(img_tensor)  # Shape: (1, 2048, 8, 8)

        # Resize to 16x16 with 1024 channels (as expected)
        features_16 = torch.nn.functional.interpolate(
            features, size=(16, 16), mode='bilinear'
        )
        features_16 = features_16[:, :1024, :, :]  # Take first 1024 channels

    # Save in required format
    feature_dict = {
        16: [features_16.squeeze(0).numpy()]
    }
    np.save(output_path, feature_dict)

# Usage
for i in range(5):
    extract_cnn_features(f'images/{i}.png',
                        f'datasets/prerunning_cnn_featuremaps/{i}.npy')
```

#### Step 4: Place Files in Correct Directories

```
datasets/
├── rplang-v3-withsemantics/
│   └── test/
│       ├── 0.npy          # Your converted graph data
│       ├── 1.npy
│       ├── 2.npy
│       ├── 3.npy
│       └── 4.npy
├── rplang-v3-withsemantics-withboundary/
│   └── test/
│       ├── 0.npy          # Copy of the same files (or with boundary data)
│       ├── 1.npy
│       ├── 2.npy
│       ├── 3.npy
│       └── 4.npy
└── prerunning_cnn_featuremaps/
    ├── 0.npy              # CNN features (real or dummy)
    ├── 1.npy
    ├── 2.npy
    ├── 3.npy
    └── 4.npy
```

**Critical**: Files must have matching names across all three directories!

#### Step 5: Verify Your Data

```python
import numpy as np

# Check graph data
graph_data = np.load('datasets/rplang-v3-withsemantics/test/0.npy',
                     allow_pickle=True).item()
print("Graph data keys:", graph_data.keys())
print("Corners shape:", graph_data['corner_list_np_normalized_padding_withsemantics'].shape)
print("Padding mask sum:", graph_data['padding_mask'].sum())

# Check CNN features
cnn_data = np.load('datasets/prerunning_cnn_featuremaps/0.npy',
                   allow_pickle=True).item()
print("CNN feature keys:", cnn_data.keys())
print("Feature 16 shape:", cnn_data[16][0].shape)
```

Expected output:
```
Graph data keys: dict_keys(['corner_list_np_normalized_padding_withsemantics', 'global_matrix_np_padding', 'padding_mask', 'edges'])
Corners shape: (53, 16)
Padding mask sum: 25  # (number of actual corners)
CNN feature keys: dict_keys([16])
Feature 16 shape: (1024, 16, 16)
```

---

## Common Issues and Troubleshooting

### Issue 1: `KeyError: 16` when loading CNN features

**Symptom**:
```
KeyError: 16
  File "datasets/rplang_edge_semantics_simplified_81.py", line 30
    self.ftmps.append(np.load(...).item()[16][0])
```

**Cause**: Missing or incorrectly formatted CNN feature files

**Solution**:
1. Check if files exist: `ls datasets/prerunning_cnn_featuremaps/`
2. Verify structure:
   ```python
   data = np.load('datasets/prerunning_cnn_featuremaps/0.npy', allow_pickle=True).item()
   print(data.keys())  # Should include 16
   print(data[16][0].shape)  # Should be (1024, 16, 16)
   ```
3. Create dummy features if needed (see "Working with Custom Data")

### Issue 2: `FileNotFoundError` when loading data

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: '../datasets/rplang-v3-withsemantics/test/0.npy'
```

**Cause**: Incorrect working directory or missing path_utils

**Solution**:
1. Use absolute paths via `path_utils.py`:
   ```python
   from datasets.path_utils import get_data_path
   file_path = get_data_path('rplang-v3-withsemantics', 'test', '0.npy')
   ```
2. Or run scripts from correct directory:
   ```bash
   cd /path/to/GSDiff
   python scripts/test_boun.py
   ```

### Issue 3: Shape mismatch errors

**Symptom**:
```
RuntimeError: shape mismatch: expected (53, 16), got (53, 9)
```

**Cause**: Data file has wrong shape

**Solution**:
- Verify your conversion script outputs correct shapes
- Check dataset loader is using correct files
- Ensure padding is applied correctly (53 corners, 16 semantic channels)

### Issue 4: Out of memory during training

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size:
   ```python
   batch_size = 128  # Instead of 256
   ```
2. Use gradient accumulation:
   ```python
   accumulation_steps = 2
   loss = loss / accumulation_steps
   loss.backward()
   if (step + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```
3. Use mixed precision training:
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

### Issue 5: Model weights not loading

**Symptom**:
```
RuntimeError: Error(s) in loading state_dict
```

**Cause**: Model architecture mismatch or wrong checkpoint file

**Solution**:
1. Verify model class matches checkpoint:
   ```python
   checkpoint = torch.load(model_path, map_location='cpu')
   print(checkpoint.keys())  # Check if it's just state_dict or has other keys
   ```
2. Load with strict=False to see missing/extra keys:
   ```python
   model.load_state_dict(checkpoint, strict=False)
   ```
3. Ensure correct model variant (e.g., HeterHouseModel vs BoundHeterHouseModel)

### Issue 6: Poor generation quality

**Symptoms**:
- Disconnected corners
- Invalid room shapes
- Overlapping walls

**Possible Causes and Solutions**:

1. **Insufficient training**:
   - Train for full 1M steps
   - Monitor validation FID/KID

2. **Wrong diffusion schedule**:
   - Verify beta schedule is cosine
   - Check alpha_bar computation

3. **Post-processing issues**:
   - Enable point merging: `merge_points = True`
   - Enable alignment: `align_points = True`
   - Adjust thresholds:
     ```python
     merge_threshold = resolution * 0.01  # 1% of resolution
     align_threshold = resolution * 0.01
     ```

4. **Dataset quality**:
   - Verify ground truth data is correct
   - Check coordinate normalization
   - Ensure semantic encoding is consistent

---

## Branches and Development

### Active Branches

```bash
# List all branches
git branch -a
```

**Main Branches**:

1. **`main`** / **`master`** (if exists): Stable release version
   - Contains tested, working code
   - Use for production/replication

2. **`claude/json-numpy-conversion-011CV3XNhPtDbzGu297ubosj`**: Data conversion development
   - Work on JSON to NumPy conversion utilities
   - Path fixes and data loading improvements

3. **`claude/review-report-accuracy-011CUtcrg4bHqzi28deEkE72`**: Report review and accuracy checks
   - Documentation improvements
   - Accuracy verification

### Development Workflow

**Creating a new branch**:
```bash
git checkout -b feature/my-new-feature
```

**Committing changes**:
```bash
git add .
git commit -m "Descriptive commit message"
git push -u origin feature/my-new-feature
```

**Merging branches**:
```bash
git checkout main
git merge feature/my-new-feature
```

---

## Advanced Topics

### Custom Model Architectures

To modify the model architecture:

1. **Edit transformer depth**:
   ```python
   # In gsdiff/house_nn1.py
   self.num_layers = 32  # Instead of 24
   ```

2. **Change embedding dimension**:
   ```python
   self.d_model = 512  # Instead of 256
   ```

3. **Add attention heads**:
   ```python
   self.num_heads = 8  # Instead of 4
   ```

Remember to retrain from scratch if changing architecture!

### Custom Semantic Classes

To add new room types:

1. **Update semantic dictionary** in dataset loader:
   ```python
   self.semantics_dict = {
       'living_room': 1,
       'kitchen': 2,
       'bedroom': 3,
       'bathroom': 4,
       'study': 5,  # New class
       'garage': 6,  # New class
       'wall': 7,
       'no_type': 0
   }
   ```

2. **Update semantic vector size**:
   - Change from 7 to 9 channels (or appropriate number)
   - Update all model input/output dimensions

3. **Update colors for rendering**:
   ```python
   colors = {
       0: (244, 241, 222),  # Living room
       1: (234, 182, 159),  # Kitchen
       # ... add new colors ...
       7: (0, 0, 0)  # Wall (always black)
   }
   ```

### Multi-GPU Training

```python
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

# Wrap model
model = HeterHouseModel()
model = model.to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])

# Use DistributedSampler for data loading
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset_train)
dataloader = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size)
```

### Custom Diffusion Schedules

**Linear schedule**:
```python
betas = np.linspace(0.0001, 0.02, diffusion_steps)
```

**Quadratic schedule**:
```python
betas = np.linspace(0.0001**0.5, 0.02**0.5, diffusion_steps) ** 2
```

**Sigmoid schedule**:
```python
betas = np.array([
    min(1 - (1 - 0.02) * (1 / (1 + np.exp(-10 * (t/diffusion_steps - 0.5)))), 0.999)
    for t in range(diffusion_steps)
])
```

### Conditional Generation

To add custom conditioning (e.g., room count, total area):

1. **Add conditioning input to model**:
   ```python
   def forward(self, x, attention_mask, t, condition):
       # Embed condition
       cond_embed = self.condition_embedding(condition)

       # Add to time embedding
       t_embed = self.time_embedding(t) + cond_embed

       # Rest of forward pass...
   ```

2. **Modify dataset to return conditions**:
   ```python
   def __getitem__(self, idx):
       # ... load data ...
       room_count = torch.tensor([self.get_room_count(idx)])
       return corners, attention_mask, padding_mask, edges, room_count
   ```

3. **Update training loop**:
   ```python
   for corners, mask, padding, edges, conditions in dataloader:
       output = model(noisy_corners, mask, t, conditions)
   ```

---

## Summary Checklist

### To Use Pre-trained Models:

- [ ] Install dependencies from `requirements.txt`
- [ ] Download model weights from Google Drive links in README
- [ ] Place weights in `scripts/outputs/`
- [ ] Download RPLAN dataset or prepare your own data
- [ ] Ensure data is in correct format (see "Dataset Requirements")
- [ ] Run test scripts: `python scripts/test_*.py`
- [ ] Check results in `scripts/test_outputs/`

### To Train Your Own Models:

- [ ] Prepare full RPLAN dataset (71,763 files)
- [ ] Process data using `rplan-extract.py` and `rplan-process*.py`
- [ ] Verify data format matches requirements
- [ ] Configure hyperparameters in training script
- [ ] Run training: `python scripts/trainval_*.py`
- [ ] Monitor validation metrics (FID/KID)
- [ ] Save checkpoints regularly
- [ ] Evaluate on test set after training

### To Use Custom Data:

- [ ] Convert your data to required NumPy format
- [ ] Create or extract CNN feature maps
- [ ] Place files in correct directories with matching names
- [ ] Verify data shapes and types
- [ ] Test with small batch first
- [ ] Run full test pipeline

---

## References and Resources

### Official Links

- **Paper**: AAAI 2025 - "GSDiff: Synthesizing Vector Floor Plans via Geometry-enhanced Structural Graph Generation"
- **RPLAN Dataset**: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
- **LIFULL Dataset**: https://github.com/SizheHu/Raster-to-Graph

### Related Work

- **HouseDiffusion**: Diffusion models for floor plan generation
- **House-GAN++**: GAN-based floor plan generation
- **Graph2Plan**: Graph-to-image floor plan generation
- **Raster-to-Graph**: Converting raster floor plans to vector graphs

### Citation

If you use this code, please cite:
```bibtex
@inproceedings{gsdiff2025,
  title={GSDiff: Synthesizing Vector Floor Plans via Geometry-enhanced Structural Graph Generation},
  author={[Authors]},
  booktitle={AAAI},
  year={2025}
}
```

---

## Contact and Support

For issues, questions, or contributions:

1. **Open an issue** on the GitHub repository
2. **Check existing issues** for similar problems
3. **Provide details**: Error messages, data samples, configurations
4. **Include environment**: OS, Python version, CUDA version

---

## Appendix: Quick Command Reference

### Data Preparation
```bash
# Extract RPLAN data
python datasets/rplan-extract.py
python datasets/rplan-process1.py
python datasets/rplan-process2.py
python datasets/rplan-process3.py
python datasets/rplan-process4.py

# Add boundary constraints
python datasets/rplan-process5.py
python datasets/rplan-process6.py
python datasets/rplan-process7.py

# Add topology constraints
python datasets/rplan-process8.py
python datasets/rplan-process9.py
python datasets/rplan-process10.py

# Move to final location
python datasets/move.py
```

### Training
```bash
# Unconstrained
python scripts/trainval_main_unconstrained.py

# Topology-constrained
python scripts/trainval_main_topo.py

# Boundary-constrained
python scripts/trainval_main_boun.py

# Edge models
python scripts/trainval_main_edge_unconstrained.py
```

### Testing
```bash
# Basic tests
python scripts/test_main.py
python scripts/test_topo.py
python scripts/test_boun.py

# Evaluation
python evalmetric-no-constrain-fid-kid.py
python evalmetric-topoconstrain-ged-roomnumber.py
python evalmetric-boun-constrain-fid-kid.py
```

### Utility Commands
```bash
# Check data format
python -c "import numpy as np; print(np.load('datasets/rplang-v3-withsemantics/test/0.npy', allow_pickle=True).item().keys())"

# Count files
find datasets/rplang-v3-withsemantics/train -name "*.npy" | wc -l

# Check GPU usage
nvidia-smi

# Monitor training
tensorboard --logdir=scripts/outputs/
```

---

**End of Complete Guide**

This guide should provide everything you need to understand, use, and modify the GSDiff repository. For specific implementation details, refer to the source code files referenced throughout this document.
