# GSDiff Data Generation Pipeline

**Report Date:** 2025-11-10

**Purpose:** Complete documentation of how the `.npy` input files are generated from the original RPLAN dataset

---

## Executive Summary

This report documents the complete data preprocessing pipeline that transforms raw RPLAN floor plan images into structured graph representations stored as `.npy` files. The pipeline consists of 10 sequential preprocessing scripts that perform corner detection, edge extraction, semantic labeling, graph construction, and feature generation.

**Key Pipeline Stages:**

1. **Edge Detection & Filtering** - Extract wall structures from PNG images
2. **Corner Detection** - Identify junction points (L, T, X corners)
3. **Graph Construction** - Build adjacency relationships
4. **Semantic Extraction** - Assign room type labels to corners
5. **Normalization & Padding** - Standardize to fixed 53-corner format
6. **CNN Feature Generation** - Pre-compute boundary image embeddings

**Input:** 80,788 PNG images from RPLAN dataset
**Output:** 71,763 valid `.npy` graph files (65,763 train + 3,000 val + 3,000 test)

---

## Table of Contents

1. [Original RPLAN Dataset Format](#1-original-rplan-dataset-format)
2. [Complete Preprocessing Pipeline](#2-complete-preprocessing-pipeline)
3. [Stage 1: Edge Detection & Filtering](#3-stage-1-edge-detection--filtering)
4. [Stage 2: Corner Detection](#4-stage-2-corner-detection)
5. [Stage 3: Edge Extraction](#5-stage-3-edge-extraction)
6. [Stage 4: Graph Construction & Normalization](#6-stage-4-graph-construction--normalization)
7. [Stage 5: Semantic Label Assignment](#7-stage-5-semantic-label-assignment)
8. [Stage 6: Padding to Fixed Size](#8-stage-6-padding-to-fixed-size)
9. [Stage 7: Attention Matrix Creation](#9-stage-7-attention-matrix-creation)
10. [Stage 8: CNN Feature Extraction](#10-stage-8-cnn-feature-extraction)
11. [Final .npy File Structure](#11-final-npy-file-structure)
12. [Data Flow Visualization](#12-data-flow-visualization)
13. [Code Reference Guide](#13-code-reference-guide)

---

## 1. Original RPLAN Dataset Format

### 1.1 Dataset Source

**Origin:** RPLAN dataset from USTC
**URL:** <http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html>
**Size:** 80,788 residential floor plan images
**License:** Academic use

### 1.2 File Format

**Type:** PNG images with 4 channels (RGBA)
**Resolution:** 256×256 pixels
**Storage Location:** `datasets/rplandata/Data/floorplan_dataset/`
**Naming:** Sequential integers: `0.png`, `1.png`, ..., `80787.png`

### 1.3 Channel Encoding

**Channel 0 (Red):** Reserved/unused
**Channel 1 (Green):** **Primary semantic information** (used in preprocessing)
**Channel 2 (Blue):** Reserved/unused
**Channel 3 (Alpha):** Opacity

### 1.4 Semantic Label Encoding (Channel 1)

Pixel values in Channel 1 encode room types and boundaries:

| Value Range | Meaning |
|-------------|---------|
| 0 | Living Room / Dining Room / Entrance |
| 1 | Master Bedroom |
| 2 | Kitchen |
| 3 | Bathroom |
| 4 | Dining Room (specific) |
| 5 | Child Room / Kids Room |
| 6 | Study Room |
| 7 | Second Bedroom |
| 8 | Guest Room |
| 9 | Balcony |
| 10 | Entrance (specific) |
| 11 | Storage / Storeroom |
| 12 | Walk-in closet |
| 13 | External area |
| ≥14 | **Walls / Boundaries** |

**Key Insight:** Pixels with values ≥14 represent wall structures, while values 0-13 represent interior room spaces.

### 1.5 Example File Reading

**File:** `datasets/rplan-extract.py:43`

```python
# Read original PNG image and extract semantic channel
origin_img = cv2.imread(
    'rplandata/Data/floorplan_dataset/' + str(id) + '.png',
    -1  # Read all channels including alpha
)[:, :, 1]  # Extract Channel 1 (Green) - semantic information
```

---

## 2. Complete Preprocessing Pipeline

The preprocessing consists of 10 sequential Python scripts that progressively refine the raw images into structured graph data.

### 2.1 Pipeline Overview

```
Raw RPLAN PNGs (80,788 files)
    ↓
[1] rplan-extract.py: Edge detection, corner extraction, filtering
    ↓ (71,763 valid files)
[2] rplan-process1.py: Convert adjacency list → adjacency matrix
    ↓
[3] rplan-process2.py: Convert coordinate-based → index-based adjacency
    ↓
[4] rplan-process3.py: Filter to dataset intersection
    ↓
[5] rplan-process4.py: Semantic extraction, normalization, padding
    ↓ (65,763 train + 3,000 val + 3,000 test)
[6] rplan-process5.py: Boundary polygon extraction
    ↓
[7] rplan-process6-7.py: Create boundary-conditioned variants
    ↓
[8] rplan-process8-10.py: Topology/bubble diagram extraction
    ↓
Final .npy Files
```

### 2.2 File Locations

**Preprocessing Scripts:** All located in `/datasets/`

- `rplan-extract.py` (407 lines)
- `rplan-process1.py` through `rplan-process10.py`
- Main processing: `rplan-process4.py` (1,600+ lines)

**Output Directories:**

- `rplang-v3-withsemantics/` - Primary output
- `rplang-v3-withsemantics-withboundary/` - With boundary info
- `rplang-v3-bubble-diagram/` - Topology graphs

---

## 3. Stage 1: Edge Detection & Filtering

**Script:** `datasets/rplan-extract.py`
**Lines:** 1-407

### 3.1 Binary Wall Image Creation

**Purpose:** Convert semantic image to binary wall/room representation

**Implementation:** Lines 56-62

```python
# Create binary image: walls=255 (white), rooms=0 (black)
binary_img = np.zeros((256, 256), dtype=np.uint8)

for i in range(256):
    for j in range(256):
        if origin_img[i, j] >= 14:  # Wall pixels
            binary_img[i, j] = 255
        else:  # Room pixels (0-13)
            binary_img[i, j] = 0
```

**Result:** Binary image where:

- White (255) = walls/boundaries
- Black (0) = interior room space

### 3.2 Morphological Processing

**Operations:** Lines 125-215

1. **Erosion:** Thin thick walls to single-pixel width

   ```python
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
   eroded = cv2.erode(binary_img, kernel, iterations=1)
   ```

2. **Skeletonization:** Extract wall centerlines

   ```python
   skeleton = cv2.ximgproc.thinning(binary_img)
   ```

3. **Dilation & Smoothing:** Clean up noise

   ```python
   dilated = cv2.dilate(skeleton, kernel, iterations=1)
   smoothed = cv2.GaussianBlur(dilated, (3, 3), 0)
   ```

**Result:** Clean single-pixel-width wall representation

### 3.3 Data Quality Filtering

**Three-stage filtering process:**

#### Filter 1: Remove Invalid Patterns (Lines 218-269)

**Issue:** Some floor plans contain 2×2 solid white blocks (artifacts)

```python
# Detect 2×2 white blocks
for i in range(254):
    for j in range(254):
        if (img[i,j] == 255 and img[i+1,j] == 255 and
            img[i,j+1] == 255 and img[i+1,j+1] == 255):
            # Mark as invalid
            valid_files.remove(file_id)
```

**Result:** 80,788 → 75,350 files

#### Filter 2: Remove Topological Errors (Lines 270-300)

**Issue:** Floor plans with dead-end corridors or disconnected components

```python
# Check connectivity
components = cv2.connectedComponents(binary_img)
if components[0] > 1:  # Multiple disconnected components
    valid_files.remove(file_id)
```

**Result:** 75,350 → 71,814 files

#### Filter 3: Validate Against Reference (Lines 301-320)

**Issue:** Ensure processed graphs match original semantic connectivity

```python
# Compare edge count and corner count with original
if not validate_graph_structure(processed_graph, original_img):
    valid_files.remove(file_id)
```

**Result:** 71,814 → 71,763 files (final valid set)

---

## 4. Stage 2: Corner Detection

**Script:** `datasets/rplan-extract.py`
**Lines:** 316-334

### 4.1 Corner Detection Algorithm

**Principle:** Corners are pixels where walls meet at angles

**Classification:**

- **L-corner:** 2 perpendicular wall segments
- **T-corner:** 3 wall segments (junction)
- **X-corner:** 4 wall segments (cross junction)

### 4.2 Implementation

```python
corners_L = []  # L-shaped corners
corners_T = []  # T-shaped junctions
corners_X = []  # X-shaped junctions

for i in range(1, 255):
    for j in range(1, 255):
        if img[i, j] == 255:  # White pixel (wall)
            # Count white neighbors in 4 directions
            up = img[i-1, j]
            down = img[i+1, j]
            left = img[i, j-1]
            right = img[i, j+1]

            neighbor_sum = up + down + left + right

            # L-corner: 2 neighbors perpendicular
            # Must be (up+down OR left+right) = 255
            if neighbor_sum == 510:  # 2 × 255
                if (up + down == 255) or (left + right == 255):
                    corners_L.append((j, i))  # Store as (x, y)

            # T-corner: 3 neighbors
            elif neighbor_sum == 765:  # 3 × 255
                corners_T.append((j, i))

            # X-corner: 4 neighbors
            elif neighbor_sum == 1020:  # 4 × 255
                corners_X.append((j, i))

# Combine all corner types
all_corners = corners_L + corners_T + corners_X
```

### 4.3 Corner Coordinates

**Format:** List of (x, y) tuples
**Range:** [0, 255] in pixel coordinates
**Example:**

```python
corners = [(45, 67), (120, 67), (120, 180), (45, 180), ...]
```

**Later Processing:** Normalized to [-1, 1] range in `rplan-process4.py`

---

## 5. Stage 3: Edge Extraction

**Script:** `datasets/rplan-extract.py`
**Lines:** 344-399

### 5.1 Edge Detection Algorithm

**Principle:** Edges are straight wall segments connecting two corners

**Requirements:**

1. Corners must be on same row OR same column (orthogonal walls)
2. All pixels between corners must be white (wall pixels)
3. No intermediate corners allowed (direct connection only)

### 5.2 Implementation

```python
edges = []

for corner1 in all_corners:
    for corner2 in all_corners:
        if corner1 == corner2:
            continue

        x1, y1 = corner1
        x2, y2 = corner2

        # Check if orthogonal (same row or column)
        if (x1 == x2) or (y1 == y2):
            # Get all pixels between corners
            if x1 == x2:  # Vertical edge
                coords = [(x1, y) for y in range(min(y1,y2), max(y1,y2)+1)]
            else:  # Horizontal edge
                coords = [(x, y1) for x in range(min(x1,x2), max(x1,x2)+1)]

            # Validate edge
            valid = True
            for coord in coords:
                # Must be white (wall)
                if img[coord[1], coord[0]] != 255:
                    valid = False
                    break

                # No intermediate corners
                if (coord in all_corners and
                    coord != corner1 and coord != corner2):
                    valid = False
                    break

            if valid:
                edges.append((corner1, corner2))
```

### 5.3 Structure Graph Format

**Output:** Dictionary mapping corners to adjacent corners with directional info

```python
structure_graph = {
    (x, y): [up_neighbor, left_neighbor, down_neighbor, right_neighbor]
}
```

**Example:**

```python
structure_graph = {
    (45, 67): [(-1, -1), (12, 67), (45, 180), (-1, -1)],
    #          up        left       down        right
    #          (none)    (corner)   (corner)    (none)
}
```

**Saved as:** `structure_graphs.npy` - dictionary mapping file_id → structure_graph

---

## 6. Stage 4: Graph Construction & Normalization

**Script:** `datasets/rplan-process4.py`
**Lines:** 1-1600+

This is the main processing script that performs semantic extraction, normalization, and padding.

### 6.1 Coordinate Normalization

**Purpose:** Convert pixel coordinates [0, 255] to normalized range [-1, 1]

**Implementation:** Lines ~1250

```python
# Load corner coordinates in pixel space
corners_np = np.array(corners_list)  # Shape: (n, 2)

# Normalize to [-1, 1]
corner_list_np_normalized = (corners_np - 128.0) / 128.0

# Example:
# Pixel (0, 0) → (-1, -1)
# Pixel (128, 128) → (0, 0)
# Pixel (255, 255) → (0.9922, 0.9922)
```

**Rationale:**

- Neural networks prefer normalized inputs
- [-1, 1] range is standard for coordinate data
- Center at (128, 128) → (0, 0)

### 6.2 Adjacency Matrix Construction

**Purpose:** Convert edge list to matrix representation

**Implementation:** Lines ~1310

```python
n_corners = len(corners)
adjacency_matrix = np.zeros((n_corners, n_corners), dtype=np.uint8)

for (corner1, corner2) in edges:
    i = corners.index(corner1)
    j = corners.index(corner2)

    # Undirected graph: symmetric matrix
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1
```

**Result:** Binary matrix where `[i,j] = 1` indicates edge between corners i and j

---

## 7. Stage 5: Semantic Label Assignment

**Script:** `datasets/rplan-process4.py`
**Lines:** 194-450, 1350-1420

### 7.1 Room Detection via Cycle Basis

**Algorithm:** Use graph cycles to identify rooms

**Implementation:** Lines 194-450

```python
import networkx as nx

# Create graph from corners and edges
G = nx.Graph()
for i, corner in enumerate(corners):
    G.add_node(i, pos=corner)

for i, j in edge_pairs:
    G.add_edge(i, j)

# Find all fundamental cycles (rooms)
cycle_basis = nx.cycle_basis(G)

# Each cycle represents one room
rooms = []
for cycle in cycle_basis:
    # Get corner coordinates for this room
    room_corners = [corners[idx] for idx in cycle]
    rooms.append(room_corners)
```

**Result:** List of room polygons, each defined by corner vertices

### 7.2 Semantic Label Extraction

**Purpose:** Determine room type by querying original image pixels

**Implementation:** Lines 1350-1420

```python
def get_points_and_pixel_values_inside_polygon(semantic_img, polygon_vertices):
    """Extract all pixels inside polygon and their semantic values"""

    # Create mask for polygon
    mask = np.zeros(semantic_img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_vertices)], 1)

    # Get all points inside polygon
    points = np.argwhere(mask == 1)

    # Get semantic value for each point
    pixel_values = [semantic_img[y, x] for (y, x) in points]

    return list(zip(points, pixel_values))

def get_room_semantic_label(semantic_img, room_polygon):
    """Determine room type by majority vote"""

    points_and_values = get_points_and_pixel_values_inside_polygon(
        semantic_img, room_polygon
    )

    # Count semantic labels
    from collections import Counter
    label_counts = Counter([val for (pt, val) in points_and_values])

    # Return most common label
    room_semantic = max(label_counts, key=label_counts.get)
    return room_semantic
```

### 7.3 Corner Semantic Vector Construction

**Purpose:** Assign semantic labels to each corner based on adjacent rooms

**Implementation:** Lines ~1360-1400

```python
# Initialize semantic vectors for all corners
corner_semantics = {corner: [0]*14 for corner in corners}

# For each room and its semantic label
for room_polygon, room_semantic in zip(rooms, room_semantics):
    # Mark all corners of this room with this semantic
    for corner in room_polygon.vertices:
        corner_semantics[corner][room_semantic] += 1

# Result: Each corner has 14-dimensional semantic vector
# Vector[i] = count of adjacent rooms with semantic type i
```

**Example:**

```python
# Corner at (45, 67) is adjacent to:
# - Living room (type 0): count = 1
# - Kitchen (type 2): count = 1
# Semantic vector: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### 7.4 Combine Coordinates and Semantics

**Implementation:** Lines ~1420-1450

```python
# Create 16-dimensional corner representation
corner_list_with_semantics = np.zeros((n_corners, 16))

for i, corner in enumerate(corners):
    # Columns 0-1: Normalized coordinates
    corner_list_with_semantics[i, 0:2] = corner_normalized

    # Columns 2-15: Semantic labels (14 dimensions)
    corner_list_with_semantics[i, 2:16] = corner_semantics[corner]
```

**Result:** Each corner represented as (x, y, s0, s1, ..., s13) where si is count of adjacent rooms of type i

---

## 8. Stage 6: Padding to Fixed Size

**Script:** `datasets/rplan-process4.py`
**Lines:** ~1265-1300

### 8.1 Padding Strategy

**Problem:** Floor plans have variable numbers of corners (typically 10-50)
**Solution:** Pad all to fixed size (53 corners) for batching

**Why 53?** Maximum corner count observed in RPLAN dataset

### 8.2 Implementation

```python
padding_to_number = 53  # Maximum corners in dataset
n_actual_corners = len(corners)  # e.g., 23 for a typical floor plan

# Pad corner coordinates
corner_list_np_normalized_padding = np.zeros((53, 2))
corner_list_np_normalized_padding[:n_actual_corners] = corner_list_normalized
# Remaining 30 corners are (0, 0)

# Pad corner + semantic data
corner_list_with_semantics_padding = np.zeros((53, 16))
corner_list_with_semantics_padding[:n_actual_corners] = corner_list_with_semantics
# Remaining 30 corners are all zeros

# Create padding mask
padding_mask = np.zeros((53, 1))
padding_mask[:n_actual_corners] = 1  # 1 = real corner
# Remaining 30 are 0 = padding corner

# Pad adjacency matrix
adjacency_matrix_padding = np.zeros((53, 53))
adjacency_matrix_padding[:n_actual_corners, :n_actual_corners] = adjacency_matrix
# Padding corners have no edges
```

### 8.3 Padding Mask Usage

**During Training/Inference:**

```python
# Compute loss only on real corners
loss = criterion(predictions, targets)
loss = loss * padding_mask  # Zero out padding positions
loss = loss.sum() / padding_mask.sum()  # Average over real corners only
```

**During Attention:**

```python
# Prevent attention to padding corners
attention_scores.masked_fill_(~global_attn_matrix, float('-inf'))
```

---

## 9. Stage 7: Attention Matrix Creation

**Script:** `datasets/rplan-process4.py`
**Lines:** ~1298-1305

### 9.1 Global Attention Matrix

**Purpose:** Define which corners can attend to each other in transformer

**Implementation:**

```python
# All real corners attend to all other real corners
global_matrix_np_padding = np.zeros((53, 53), dtype=np.uint8)
global_matrix_np_padding[:n_real_corners, :n_real_corners] = 1

# Example for floor plan with 23 corners:
# global_matrix[0:23, 0:23] = 1  (all real corners can attend)
# global_matrix[23:53, :] = 0     (padding can't attend)
# global_matrix[:, 23:53] = 0     (can't attend to padding)
```

**Usage in Model:**

```python
# In transformer attention layer
attention_weights = softmax(Q @ K.T / sqrt(d_k))
attention_weights = attention_weights.masked_fill(~global_attn_matrix, 0)
output = attention_weights @ V
```

### 9.2 Edge Coordinates and Binary Edge Vector

**Purpose:** Create all pairwise edge representations

**Implementation:** Lines ~1320

```python
# Create edge coordinate pairs for all (i,j) combinations
edge_coord1 = np.repeat(
    corner_list_normalized_padding[:, np.newaxis, :],
    53, axis=1
)  # (53, 53, 2)

edge_coord2 = np.repeat(
    corner_list_normalized_padding[np.newaxis, :, :],
    53, axis=0
)  # (53, 53, 2)

# Concatenate to get (x1, y1, x2, y2) for each pair
edge_coords = np.concatenate([edge_coord1, edge_coord2], axis=2)  # (53, 53, 4)
edge_coords = edge_coords.reshape(-1, 4)  # (2809, 4) where 2809 = 53*53

# Binary edge existence vector
edges = adjacency_matrix_padding.reshape(-1, 1)  # (2809, 1)
# edges[i*53 + j] = 1 if edge exists between corner i and j, else 0
```

---

## 10. Stage 8: CNN Feature Extraction

**Script:** `scripts/prerunningCNN.py`
**Lines:** 1-214

### 10.1 Purpose

Pre-compute CNN features for boundary/wall images to avoid redundant computation during training.

### 10.2 Boundary Image Rendering

**Process:** Lines 50-100

```python
def render_boundary_image(corners, edges, image_size=256):
    """Render floor plan boundary as binary image"""

    # Create blank image
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    # Convert normalized coords [-1,1] back to pixel coords [0,255]
    corners_pixel = ((corners + 1) * 128).astype(np.int32)

    # Draw edges as white lines
    for i, j in edge_pairs:
        pt1 = tuple(corners_pixel[i])
        pt2 = tuple(corners_pixel[j])
        cv2.line(img, pt1, pt2, 255, thickness=2)

    # Convert to 3-channel RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img_rgb
```

### 10.3 CNN Encoder Architecture

**Model:** BoundaryModel (ResNet-like)

**Implementation:** Lines 120-180

```python
class BoundaryModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(2, 2)  # 256→128

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128→64

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64→32

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 32→16

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        # Output: (1024, 16, 16)

    def forward(self, x):
        # Input: (BS, 3, 256, 256)
        e1 = F.relu(self.conv1(x))          # (BS, 64, 256, 256)
        e1 = self.pool1(e1)                  # (BS, 64, 128, 128)

        e2 = F.relu(self.conv2(e1))         # (BS, 128, 128, 128)
        e2 = self.pool2(e2)                  # (BS, 128, 64, 64)

        e3 = F.relu(self.conv3(e2))         # (BS, 256, 64, 64)
        e3 = self.pool3(e3)                  # (BS, 256, 32, 32)

        e4 = F.relu(self.conv4(e3))         # (BS, 512, 32, 32)
        e4 = self.pool4(e4)                  # (BS, 512, 16, 16)

        e5 = F.relu(self.conv5(e4))         # (BS, 1024, 16, 16)

        return e5, e4, e3  # Multi-scale features
```

### 10.4 Feature Extraction Process

**Implementation:** Lines 185-210

```python
model = BoundaryModel().cuda()
model.eval()

for file_id in file_list:
    # Load graph data
    graph = np.load(f'rplang-v3-withsemantics/train/{file_id}.npy',
                    allow_pickle=True).item()

    # Render boundary image
    corners = graph['corner_list_np_normalized_padding'][:, 0:2]
    edges = graph['adjacency_matrix_np_padding']
    boundary_img = render_boundary_image(corners, edges)

    # Convert to tensor
    img_tensor = torch.from_numpy(boundary_img).float().permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).cuda()  # (1, 3, 256, 256)

    # Extract features
    with torch.no_grad():
        feat_16, feat_32, feat_64 = model(img_tensor)

    # Save features
    features = {
        'feat_16': feat_16.cpu().numpy(),  # (1, 1024, 16, 16)
        'feat_32': feat_32.cpu().numpy(),  # (1, 512, 32, 32)
        'feat_64': feat_64.cpu().numpy()   # (1, 256, 64, 64)
    }

    np.save(f'prerunning_cnn_featuremaps/{file_id}_feat16.npy',
            features['feat_16'])
```

**Output:** Pre-computed feature maps stored in `datasets/rplang-v3-withsemantics-prerunCNN-16/`

---

## 11. Final .npy File Structure

### 11.1 Complete Dictionary Keys

Each `.npy` file contains a Python dictionary with the following structure:

```python
graph_dict = {
    # ===== Raw Graph Data =====
    'corners': list,
        # List of (x, y) tuples in original pixel coordinates [0, 255]

    'corners_np': ndarray(n, 2),
        # NumPy array of corner coordinates in pixel space

    'adjacency_list': list,
        # List of lists: adjacency_list[i] = [indices of neighbors of corner i]

    'adjacency_matrix': ndarray(n, n),
        # Binary adjacency matrix (n = actual number of corners)

    # ===== Normalized Data =====
    'corner_list_np_normalized': ndarray(n, 2),
        # Normalized coordinates in [-1, 1] range
        # Formula: (pixel_coord - 128) / 128

    # ===== Padded Data (Fixed Size 53) =====
    'corner_list_np_normalized_padding': ndarray(53, 2),
        # Padded normalized coordinates
        # First n rows: real corners
        # Remaining (53-n) rows: zeros (padding)

    'corner_list_np_normalized_padding_withsemantics': ndarray(53, 16),
        # Padded corners with semantic labels
        # Columns 0-1: normalized (x, y)
        # Columns 2-15: semantic labels (14 dimensions)

    'adjacency_matrix_np_padding': ndarray(53, 53),
        # Padded adjacency matrix
        # Top-left (n×n) block: real edges
        # Rest: zeros (no edges to/from padding)

    'global_matrix_np_padding': ndarray(53, 53),
        # Global attention mask
        # Top-left (n×n) block: all 1s (full attention among real corners)
        # Rest: zeros (no attention to/from padding)

    'padding_mask': ndarray(53, 1),
        # Binary mask: 1 = real corner, 0 = padding
        # Used to filter padding in loss calculations

    # ===== Edge Data =====
    'edge_coords': ndarray(2809, 4),
        # All pairwise edge coordinates (2809 = 53*53)
        # Each row: [x1, y1, x2, y2]
        # edge_coords[i*53 + j] = coordinates from corner i to corner j

    'edges': ndarray(2809, 1),
        # Binary edge existence (flattened adjacency matrix)
        # edges[i*53 + j] = 1 if edge exists, 0 otherwise

    # ===== Semantic Data =====
    'semantics': dict,
        # Dictionary mapping corner coordinates → semantic vector
        # Key: (x_normalized, y_normalized) tuple
        # Value: [count_semantic_0, ..., count_semantic_13]

    # ===== Optional: Boundary Data (in withboundary variants) =====
    'boundary_vertex_indices': ndarray(53, 2),
        # Binary mask: 1 if corner is on boundary, 0 otherwise

    'boundary_adjacency_matrix': ndarray(53, 53),
        # Adjacency matrix for boundary edges only

    'boundary_vertex_coords_4cvae': list,
        # List of boundary corner coordinates (for CVAE training)
}
```

### 11.2 Semantic Label Encoding (14 Dimensions)

The 14-dimensional semantic vector in columns 2-15:

| Index | Column | Room Type |
|-------|--------|-----------|
| 0 | 2 | Living Room / Dining Room / Entrance |
| 1 | 3 | Master Bedroom |
| 2 | 4 | Kitchen |
| 3 | 5 | Bathroom |
| 4 | 6 | Dining Room (specific) |
| 5 | 7 | Child Room / Kids Room |
| 6 | 8 | Study Room |
| 7 | 9 | Second Bedroom |
| 8 | 10 | Guest Room |
| 9 | 11 | Balcony |
| 10 | 12 | Entrance (specific) |
| 11 | 13 | Storage / Storeroom |
| 12 | 14 | Walk-in Closet |
| 13 | 15 | External Area |

**Note:** These 14 dimensions are later simplified to 9 dimensions during dataset loading (see `rplang_edge_semantics_simplified_81.py`).

### 11.3 Example Data Structure

```python
# Load example file
graph = np.load('rplang-v3-withsemantics/train/0.npy', allow_pickle=True).item()

# Print shapes
print("Corner coordinates (padded):", graph['corner_list_np_normalized_padding'].shape)
# Output: (53, 2)

print("Corners with semantics:", graph['corner_list_np_normalized_padding_withsemantics'].shape)
# Output: (53, 16)

print("Adjacency matrix:", graph['adjacency_matrix_np_padding'].shape)
# Output: (53, 53)

print("Padding mask:", graph['padding_mask'].shape)
# Output: (53, 1)

print("Edge coordinates:", graph['edge_coords'].shape)
# Output: (2809, 4)

# Check actual number of corners
n_real_corners = int(graph['padding_mask'].sum())
print(f"Real corners: {n_real_corners}, Padding: {53 - n_real_corners}")
# Example: Real corners: 23, Padding: 30
```

---

## 12. Data Flow Visualization

### 12.1 Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RPLAN PNG Image (256×256×4)                  │
│                   Channel 1: Semantic Labels                    │
│              Pixels <14: Rooms, Pixels ≥14: Walls               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Step 1: Binary Wall Extraction                 │
│                    [rplan-extract.py:56-62]                     │
│                                                                 │
│  For each pixel: value ≥14 → 255 (white), else → 0 (black)    │
│  Output: 256×256 binary image (walls=white, rooms=black)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│               Step 2: Morphological Processing                  │
│                   [rplan-extract.py:125-215]                    │
│                                                                 │
│  Erosion → Skeletonization → Dilation → Smoothing              │
│  Output: Single-pixel-width wall structure                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Step 3: Quality Filtering                     │
│                   [rplan-extract.py:218-320]                    │
│                                                                 │
│  Remove: 2×2 blocks, disconnected components, invalid topology │
│  80,788 → 75,350 → 71,814 → 71,763 valid files                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Step 4: Corner Detection                     │
│                   [rplan-extract.py:316-334]                    │
│                                                                 │
│  Detect L-corners (2 neighbors), T-corners (3), X-corners (4)  │
│  Output: List of (x,y) pixel coordinates [0,255]               │
│  Example: [(45,67), (120,67), (120,180), (45,180), ...]        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Step 5: Edge Extraction                     │
│                   [rplan-extract.py:344-399]                    │
│                                                                 │
│  For each corner pair:                                          │
│    - Check orthogonality (same row/column)                      │
│    - Verify continuous wall pixels                             │
│    - Ensure no intermediate corners                            │
│  Output: List of (corner1, corner2) edge pairs                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                Step 6: Graph Construction                       │
│                  [rplan-process1-3.py]                          │
│                                                                 │
│  process1: adjacency list → adjacency matrix                   │
│  process2: coordinate-based → index-based adjacency            │
│  process3: filter to dataset intersection                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│             Step 7: Room Detection (Cycle Basis)                │
│                  [rplan-process4.py:194-450]                    │
│                                                                 │
│  Use NetworkX to find fundamental cycles                        │
│  Each cycle = one room polygon                                  │
│  Output: List of room polygons with vertices                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              Step 8: Semantic Label Assignment                  │
│                 [rplan-process4.py:1350-1420]                   │
│                                                                 │
│  For each room polygon:                                         │
│    1. Extract all interior pixels                              │
│    2. Query original image for semantic values                 │
│    3. Determine room type by majority vote                     │
│    4. Assign semantic to all corners of room                   │
│  Output: 14-dim semantic vector per corner                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Step 9: Coordinate Normalization                │
│                 [rplan-process4.py:~1250]                       │
│                                                                 │
│  Formula: (pixel_coord - 128) / 128                            │
│  Range: [0, 255] → [-1, 1]                                     │
│  Example: (0,0)→(-1,-1), (128,128)→(0,0), (255,255)→(0.99,0.99)│
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│            Step 10: Padding to Fixed Size (53)                  │
│                [rplan-process4.py:1265-1300]                    │
│                                                                 │
│  Pad coordinates: (n,2) → (53,2) with zeros                    │
│  Pad semantics: (n,16) → (53,16) with zeros                    │
│  Create padding mask: (53,1) - first n are 1, rest are 0      │
│  Pad adjacency: (n,n) → (53,53) with zeros                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│           Step 11: Attention Matrix Creation                    │
│                [rplan-process4.py:1298-1305]                    │
│                                                                 │
│  Global attention: top-left (n×n) block = 1, rest = 0         │
│  Allows real corners to attend to each other, not padding     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              Step 12: Edge Coordinate Generation                │
│                [rplan-process4.py:~1320]                        │
│                                                                 │
│  Generate all (53×53) edge coordinate pairs                    │
│  edge_coords[i*53+j] = [x_i, y_i, x_j, y_j]                   │
│  edges[i*53+j] = 1 if edge exists, 0 otherwise                │
│  Output: (2809,4) coordinates + (2809,1) binary                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Step 13: Save .npy File                      │
│                  [rplan-process4.py:~1450]                      │
│                                                                 │
│  Save dictionary with all data:                                │
│  - corner_list_np_normalized_padding: (53,2)                   │
│  - corner_list_np_normalized_padding_withsemantics: (53,16)    │
│  - adjacency_matrix_np_padding: (53,53)                        │
│  - global_matrix_np_padding: (53,53)                           │
│  - padding_mask: (53,1)                                        │
│  - edge_coords: (2809,4)                                       │
│  - edges: (2809,1)                                             │
│  Location: rplang-v3-withsemantics/{train,val,test}/*.npy      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│            Step 14: CNN Feature Pre-computation                 │
│                  [scripts/prerunningCNN.py]                     │
│                                                                 │
│  1. Render boundary image from corners + edges                 │
│  2. Pass through ResNet-like CNN encoder                       │
│  3. Extract features at 16×16 resolution                       │
│  4. Save as *_feat16.npy: (1,1024,16,16)                       │
│  Location: rplang-v3-withsemantics-prerunCNN-16/{train,val,test}/│
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FINAL OUTPUT FILES                         │
│                                                                 │
│  ✓ Graph data: 71,763 .npy files                               │
│  ✓ CNN features: 71,763 _feat16.npy files                      │
│  ✓ Split: 65,763 train + 3,000 val + 3,000 test               │
│  ✓ Ready for model training/inference                          │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Data Shape Transformations

```
Original PNG Image
    256 × 256 × 4 (RGBA)
    ↓
Binary Wall Image
    256 × 256 × 1 (grayscale)
    ↓
Corner Detection
    n corners (variable, typically 10-50)
    List of (x, y) tuples
    ↓
Adjacency Matrix
    n × n (binary)
    ↓
Normalize Coordinates
    (n, 2) - range [-1, 1]
    ↓
Combine with Semantics
    (n, 16) - [x, y, s0, ..., s13]
    ↓
Pad to Fixed Size
    (53, 16) - first n real, rest padding
    ↓
Create Attention Matrices
    (53, 53) - global attention
    (53, 53) - adjacency
    ↓
Generate Edge Data
    (2809, 4) - edge coordinates
    (2809, 1) - edge existence
    ↓
Add Padding Mask
    (53, 1) - binary indicator
    ↓
Pre-compute CNN Features
    (1, 1024, 16, 16) - boundary embeddings
```

---

## 13. Code Reference Guide

### 13.1 Preprocessing Scripts

**Location:** `/home/user/GSDiff/datasets/`

| Script | Lines | Purpose | Key Operations |
|--------|-------|---------|----------------|
| `rplan-extract.py` | 407 | Edge/corner detection | Binary conversion, morphology, filtering, corner/edge extraction |
| `rplan-process1.py` | ~30 | Adjacency list → matrix | Convert list representation to matrix |
| `rplan-process2.py` | ~30 | Coordinate → index | Map coordinates to indices |
| `rplan-process3.py` | ~50 | Dataset filtering | Filter to valid intersection |
| `rplan-process4.py` | 1600+ | **Main processing** | Cycles, semantics, normalization, padding |
| `rplan-process5.py` | ~200 | Boundary extraction | Extract boundary polygons |
| `rplan-process6-7.py` | ~150 | Boundary variants | Create withboundary datasets |
| `rplan-process8-10.py` | ~250 | Topology extraction | Create bubble diagrams |

### 13.2 Key Functions in rplan-process4.py

```python
# Line 10-17: Get semantic label for polygon
def get_label(semantic_img, polygon_vertices):
    """Query image pixels to determine room type"""

# Line 20-42: Extract pixels inside polygon
def get_points_and_pixel_values_inside_polygon(semantic_img, polygon):
    """Get all interior pixels and their semantic values"""

# Line 194-450: Room detection via cycle basis
def get_cycle_basis_and_semantic(graph, corners, semantic_img):
    """Find all rooms (cycles) and assign semantic labels"""

# Line ~1250: Coordinate normalization
corner_list_normalized = (corners_np - 128) / 128

# Line ~1265-1300: Padding implementation
corner_list_padding = np.zeros((53, dimension))
corner_list_padding[:n_real] = corner_list_real

# Line ~1350-1420: Semantic vector construction
for room in rooms:
    for corner in room.vertices:
        semantic_vector[room.label] += 1
```

### 13.3 CNN Feature Extraction Script

**Location:** `/home/user/GSDiff/scripts/prerunningCNN.py`

```python
# Line 120-180: BoundaryModel architecture
class BoundaryModel(nn.Module):
    """ResNet-like encoder for boundary images"""

# Line 50-100: Render boundary image
def render_boundary_image(corners, edges):
    """Convert graph to binary image"""

# Line 185-210: Feature extraction loop
for file_id in dataset:
    boundary_img = render_boundary_image(graph)
    features = model(boundary_img)
    np.save(f'{file_id}_feat16.npy', features)
```

### 13.4 Output Directory Structure

```
datasets/
├── rplandata/
│   └── Data/
│       └── floorplan_dataset/
│           ├── 0.png          # Original RPLAN images
│           ├── 1.png
│           └── ...
│
├── rplang-v3-withsemantics/
│   ├── train/
│   │   ├── 0.npy             # Processed graph files
│   │   ├── 1.npy
│   │   └── ... (65,763 files)
│   ├── val/
│   │   └── ... (3,000 files)
│   └── test/
│       └── ... (3,000 files)
│
├── rplang-v3-withsemantics-prerunCNN-16/
│   ├── train/
│   │   ├── 0_feat16.npy      # Pre-computed CNN features
│   │   ├── 1_feat16.npy
│   │   └── ... (65,763 files)
│   ├── val/
│   │   └── ... (3,000 files)
│   └── test/
│       └── ... (3,000 files)
│
├── rplang-v3-withsemantics-withboundary/
│   └── ... (boundary-conditioned variants)
│
└── rplang-v3-bubble-diagram/
    └── ... (topology-only graphs)
```

---

## Summary

This data generation pipeline transforms raw RPLAN floor plan images into structured graph representations suitable for deep learning:

1. **Input:** 80,788 PNG images with semantic channel encoding
2. **Processing:** 10-stage pipeline with filtering, detection, extraction, normalization
3. **Output:** 71,763 `.npy` files with graph structure + semantics + CNN features
4. **Key Innovation:** Fixed-size padding (53 corners) enables batching variable-size graphs
5. **Optimization:** Pre-computed CNN features avoid redundant computation

The resulting `.npy` files contain complete graph representations with:

- Normalized corner coordinates (53, 2)
- Semantic labels (14 dimensions per corner)
- Adjacency matrices (53, 53)
- Attention masks (53, 53)
- Edge data (2809 pairs)
- Padding masks (53, 1)
- Pre-computed CNN features (1024, 16, 16)

This structured format enables efficient training of the GSDiff diffusion model for floor plan generation.

---

**End of Report**
