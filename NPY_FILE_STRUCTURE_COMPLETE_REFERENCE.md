# GSDiff: Numpy File (.npy) Creation and Loading - Complete Reference

**Date:** 2025-11-10
**Purpose:** Understand exactly where .npy files are created, read, and their structure

---

## Quick Answer

| Question | Answer | File Location |
|----------|--------|---------------|
| **Where are .npy files CREATED?** | `datasets/rplan-process4.py:1079-1083` | Preprocessing stage |
| **Where are .npy files READ?** | `datasets/rplang_edge_semantics_simplified_81.py:58-64` | Dataset loader |
| **What's the structure?** | Python dictionary with 15+ keys | See detailed structure below |

---

## Part 1: Where .npy Files Are CREATED

### 1.1 File Location

**Script:** `/home/user/GSDiff/datasets/rplan-process4.py`

**Lines:** 1079-1083

### 1.2 Exact Code Creating .npy Files

```python
# From rplan-process4.py, lines 1072-1085

# Create the dictionary structure
new_graph = copy.deepcopy(graph)
new_graph['semantics'] = normalized_seman_d
new_graph['corner_list_np_normalized_padding_withsemantics'] = result

# Save to .npy files
if file_id in train_fnids:
    np.save('./rplandata/Data/rplang-v3-withsemantics/train/' + str(file_id) + '.npy', new_graph)
elif file_id in val_fnids:
    np.save('./rplandata/Data/rplang-v3-withsemantics/val/' + str(file_id) + '.npy', new_graph)
elif file_id in test_fnids:
    np.save('./rplandata/Data/rplang-v3-withsemantics/test/' + str(file_id) + '.npy', new_graph)
else:
    assert 0
```

### 1.3 What Is Being Saved

The variable `new_graph` is a Python dictionary containing all the processed floor plan data.

**Output Directories:**
- `./rplandata/Data/rplang-v3-withsemantics/train/*.npy` (65,763 files)
- `./rplandata/Data/rplang-v3-withsemantics/val/*.npy` (3,000 files)
- `./rplandata/Data/rplang-v3-withsemantics/test/*.npy` (3,000 files)

---

## Part 2: Building The Dictionary Structure

### 2.1 Dictionary Construction Process

The dictionary `g` (later copied to `new_graph`) is built step by step in `rplan-process4.py`.

**Location of main loop:** Lines 871-1085

```python
# Line 871: Main loop starts
for file_id, structure_graph in tqdm(structure_graphs.items()):
    g = {}  # Line 874: Initialize empty dictionary

    # Lines 876-955: Build the dictionary with 15+ keys
    # Lines 1072-1075: Add semantic information
    # Lines 1079-1083: Save to .npy file
```

### 2.2 Complete Dictionary Structure (Step by Step)

Here's the **exact order** in which keys are added to the dictionary:

```python
# From rplan-process4.py, showing line numbers and code

# ===== Line 876: Basic identification =====
g['file_id'] = file_id  # Integer ID of the floor plan

# ===== Lines 879-889: Original data from structure_graphs =====
g['corners'] = corners  # List of (x,y) tuples in pixel coords
g['adjacency_matrix'] = adjacency_matrix  # n×n list of lists
g['adjacency_list'] = adjacency_list  # List of neighbor indices

# ===== Lines 892-902: Convert to numpy arrays =====
g['corners_np'] = corners_np  # shape: (n, 2)
g['adjacency_matrix_np'] = adjacency_matrix_np  # shape: (n, n)
g['adjacency_list_np'] = adjacency_list_np  # shape: variable

# ===== Line 910: Normalize coordinates =====
g['corner_list_np_normalized'] = corner_list_np_normalized
# shape: (n, 2), range: [-1, 1]
# Formula: (corners_np - 128) / 128

# ===== Lines 922-941: Padding to fixed size (53) =====
g['corner_list_np_normalized_padding'] = corner_list_np_normalized_padding
# shape: (53, 2), padded with zeros

g['padding_mask'] = padding_mask
# shape: (53, 1), 1=real corner, 0=padding

g['global_matrix_np_padding'] = global_matrix_np_padding
# shape: (53, 53), boolean attention mask

g['adjacency_matrix_np_padding'] = adjacency_matrix_np_padding
# shape: (53, 53), padded adjacency

# ===== Lines 950-955: Edge data =====
g['edge_coords'] = edge_coords
# shape: (2809, 4) where 2809 = 53×53
# Each row: [x1, y1, x2, y2] for corner pair

g['edges'] = edges
# shape: (2809, 1), binary edge existence

# ===== Lines 1074-1075: Add semantic information =====
new_graph = copy.deepcopy(g)
new_graph['semantics'] = normalized_seman_d
# Dictionary mapping normalized coords → 14-dim semantic vector

new_graph['corner_list_np_normalized_padding_withsemantics'] = result
# shape: (53, 16)
# Columns 0-1: x, y coordinates
# Columns 2-15: 14-dimensional semantic labels
```

---

## Part 3: Complete .npy File Structure

### 3.1 All Dictionary Keys

When you load a .npy file, you get a dictionary with these keys:

```python
# Load example
graph = np.load('rplang-v3-withsemantics/train/0.npy', allow_pickle=True).item()

# Available keys:
graph.keys() = [
    'file_id',                                              # int
    'corners',                                              # list of tuples
    'adjacency_matrix',                                     # list of lists
    'adjacency_list',                                       # list
    'corners_np',                                           # ndarray(n, 2)
    'adjacency_matrix_np',                                  # ndarray(n, n)
    'adjacency_list_np',                                    # ndarray
    'corner_list_np_normalized',                            # ndarray(n, 2)
    'corner_list_np_normalized_padding',                    # ndarray(53, 2)
    'padding_mask',                                         # ndarray(53, 1)
    'global_matrix_np_padding',                             # ndarray(53, 53)
    'adjacency_matrix_np_padding',                          # ndarray(53, 53)
    'edge_coords',                                          # ndarray(2809, 4)
    'edges',                                                # ndarray(2809, 1)
    'semantics',                                            # dict
    'corner_list_np_normalized_padding_withsemantics'       # ndarray(53, 16)
]
```

### 3.2 Detailed Structure with Types and Shapes

```python
{
    # ===== Identification =====
    'file_id': int
        # Example: 12345
        # Original RPLAN image ID

    # ===== Original Corner Data (Variable Size) =====
    'corners': list
        # Example: [(45, 67), (120, 67), (120, 180), ...]
        # List of (x, y) tuples in pixel coordinates [0, 255]
        # Length: n (variable, typically 10-50)

    'corners_np': ndarray(n, 2)
        # Same as 'corners' but as numpy array
        # dtype: float64
        # Example shape: (23, 2) for floor plan with 23 corners

    # ===== Original Adjacency Data (Variable Size) =====
    'adjacency_matrix': list[list[int]]
        # Example: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        # n×n matrix, 1=edge exists, 0=no edge

    'adjacency_matrix_np': ndarray(n, n)
        # Same as 'adjacency_matrix' but as numpy array
        # dtype: uint8
        # Example shape: (23, 23)

    'adjacency_list': list
        # Example: [[1, 5], [0, 2, 7], [1], ...]
        # For each corner, list of connected corner indices

    'adjacency_list_np': ndarray
        # Same as 'adjacency_list' but as numpy array
        # dtype: uint8

    # ===== Normalized Coordinates (Variable Size) =====
    'corner_list_np_normalized': ndarray(n, 2)
        # Coordinates normalized to [-1, 1] range
        # Formula: (pixel_coord - 128) / 128
        # dtype: float64
        # Example shape: (23, 2)
        # Example values: [[-0.64, -0.48], [0.12, -0.48], ...]

    # ===== Padded Data (Fixed Size: 53) =====
    'corner_list_np_normalized_padding': ndarray(53, 2)
        # Padded version of corner_list_np_normalized
        # First n rows: real corners
        # Remaining (53-n) rows: zeros (padding)
        # dtype: float64
        # Shape: always (53, 2)

    'padding_mask': ndarray(53, 1)
        # Binary mask indicating valid corners
        # 1 = real corner, 0 = padding corner
        # dtype: uint8
        # Shape: (53, 1)
        # Example: [[1], [1], ..., [1], [0], [0], ...]
        #          ^-- n ones --^  ^-- (53-n) zeros --^

    'global_matrix_np_padding': ndarray(53, 53)
        # Global attention mask (all-to-all for real corners)
        # Top-left (n×n) block: all 1s
        # Rest: all 0s (padding doesn't attend)
        # dtype: uint8
        # Shape: (53, 53)
        # Used for: Self-attention masking in Transformer

    'adjacency_matrix_np_padding': ndarray(53, 53)
        # Padded adjacency matrix
        # Top-left (n×n) block: real adjacency
        # Rest: all 0s (no edges to/from padding)
        # dtype: uint8
        # Shape: (53, 53)

    # ===== Edge Data (Fixed Size: 2809 = 53×53) =====
    'edge_coords': ndarray(2809, 4)
        # All pairwise edge coordinates
        # Each row: [x1, y1, x2, y2] for corners (i, j)
        # Row index = i*53 + j
        # dtype: float64
        # Shape: (2809, 4)
        # Example row: [-0.64, -0.48, 0.12, -0.48]
        #              ^-- corner i --^  ^-- corner j --^

    'edges': ndarray(2809, 1)
        # Binary edge existence for all pairs
        # edges[i*53 + j] = 1 if edge exists between corners i and j
        # Flattened version of adjacency_matrix_np_padding
        # dtype: uint8
        # Shape: (2809, 1)

    # ===== Semantic Data =====
    'semantics': dict
        # Maps normalized coordinates → 14-dim semantic vector
        # Key: (x_normalized, y_normalized) tuple
        # Value: [count_type_0, count_type_1, ..., count_type_13]
        # Example: {
        #     (-0.64, -0.48): [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     (0.12, -0.48): [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # }
        # Interpretation: Corner at (-0.64, -0.48) is adjacent to:
        #   - 1 living room (type 0)
        #   - 1 kitchen (type 2)

    'corner_list_np_normalized_padding_withsemantics': ndarray(53, 16)
        # Combines coordinates + semantics in single array
        # Columns 0-1: normalized x, y coordinates
        # Columns 2-15: 14-dimensional semantic vector
        # dtype: float64
        # Shape: (53, 16)
        # This is the MAIN array used during training!
}
```

### 3.3 The 14 Semantic Dimensions (Columns 2-15)

```python
# Column index in 'corner_list_np_normalized_padding_withsemantics'
columns = {
    2:  'Living room / Dining room / Entrance (merged)',
    3:  'Master bedroom',
    4:  'Kitchen',
    5:  'Bathroom',
    6:  'Dining room (specific)',
    7:  'Child room / Kids room',
    8:  'Study room',
    9:  'Second bedroom',
    10: 'Guest room',
    11: 'Balcony',
    12: 'Entrance (specific)',
    13: 'Storage / Storeroom',
    14: 'Walk-in closet',
    15: 'External area'
}
```

**Important:** These are NOT one-hot encoded! Each value is a **count** of how many adjacent rooms have that type.

**Example:**
```python
corner_semantics = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#                   ^     ^
#                   |     |
#                   |     +-- 1 kitchen adjacent
#                   +-------- 1 living room adjacent

# This corner is at the junction between living room and kitchen
```

---

## Part 4: Where .npy Files Are READ

### 4.1 File Location

**Script:** `/home/user/GSDiff/datasets/rplang_edge_semantics_simplified_81.py`

**Lines:** 58-64 (loading), 67-102 (processing and returning)

### 4.2 Exact Code Reading .npy Files

```python
# From rplang_edge_semantics_simplified_81.py, lines 57-64

if self.mode == 'train':
    graph = np.load(get_data_path('rplang-v3-withsemantics', 'train', self.files[index]),
                    allow_pickle=True).item()
elif self.mode == 'val':
    graph = np.load(get_data_path('rplang-v3-withsemantics', 'val', self.files[index]),
                    allow_pickle=True).item()
elif self.mode == 'test':
    graph = np.load(get_data_path('rplang-v3-withsemantics', 'test', self.files[index]),
                    allow_pickle=True).item()
else:
    assert 0, 'mode error'
```

### 4.3 What Data Is Extracted

```python
# Line 67: Extract the main semantic array
corners_withsemantics = graph['corner_list_np_normalized_padding_withsemantics']
# Shape: (53, 16)

# Lines 69-82: Simplify semantics from 14 dimensions to 7
corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))

# Copy coordinates (columns 0-1)
corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]

# Merge semantic categories:
# New col 2: Living (sum of cols 2, 6, 12)
corners_withsemantics_simplified[:, 2] = corners_withsemantics[:, [2, 6, 12]].sum(axis=1)

# New col 3: Bedrooms (sum of cols 3, 7, 8, 9, 10)
corners_withsemantics_simplified[:, 3] = corners_withsemantics[:, [3, 7, 8, 9, 10]].sum(axis=1)

# New col 4: Storage (sum of cols 13, 14)
corners_withsemantics_simplified[:, 4] = corners_withsemantics[:, [13, 14]].sum(axis=1)

# Copy remaining categories directly:
corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]   # Kitchen
corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]   # Bathroom
corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]  # Balcony
corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]  # External

# Result shape: (53, 9)
# Columns: [x, y, living, bedrooms, storage, kitchen, bathroom, balcony, external]
```

### 4.4 Other Data Extracted

```python
# Line 89: Create or load global attention matrix
global_attn_matrix = np.ones((53, 53), dtype=np.uint8)
# Note: This code creates all-ones, but the saved file has proper masking

# Line 91: Load padding mask
corners_padding_mask = graph['padding_mask']  # shape: (53, 1)

# Line 93: Load pre-computed CNN features (from separate file)
featmap_16 = self.ftmps[index]  # shape: (1024, 16, 16)
# These are loaded in __init__ from 'prerunning_cnn_featuremaps' directory
```

### 4.5 What's Returned to Training

```python
# Line 102: Return tuple of 4 items
return (
    featmap_16,                        # (1024, 16, 16) - CNN features
    corners_withsemantics_simplified,  # (53, 9) - coords + 7 semantics
    global_attn_matrix,                # (53, 53) - attention mask
    corners_padding_mask               # (53, 1) - valid corner indicator
)
```

---

## Part 5: Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW RPLAN PNG IMAGE                          │
│                      (256×256×4 RGBA)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│              PREPROCESSING (rplan-process1-3.py)                │
│  - Extract corners and edges                                    │
│  - Build adjacency matrix                                       │
│  - Save to structure_graphs.npy                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│            MAIN PROCESSING (rplan-process4.py)                  │
│                                                                 │
│  Line 852: Load structure_graphs.npy                           │
│  Line 874: Initialize empty dict g = {}                        │
│            ↓                                                    │
│  Lines 876-955: Build dictionary with 14 keys:                 │
│    ├─ corners, adjacency_matrix, adjacency_list                │
│    ├─ corners_np, adjacency_matrix_np                          │
│    ├─ corner_list_np_normalized                                │
│    ├─ corner_list_np_normalized_padding (53, 2)                │
│    ├─ padding_mask (53, 1)                                     │
│    ├─ global_matrix_np_padding (53, 53)                        │
│    ├─ adjacency_matrix_np_padding (53, 53)                     │
│    ├─ edge_coords (2809, 4)                                    │
│    └─ edges (2809, 1)                                          │
│            ↓                                                    │
│  Lines 962-1041: Extract semantic labels                       │
│    ├─ Read PNG image channel 1                                 │
│    ├─ Detect room polygons (cycles in graph)                   │
│    ├─ Query pixels to get room types                           │
│    └─ Assign 14-dim semantic vector to each corner             │
│            ↓                                                    │
│  Lines 1048-1075: Create final arrays                          │
│    ├─ semantics: dict mapping coords → 14-dim vector           │
│    └─ corner_list_np_normalized_padding_withsemantics (53,16)  │
│            ↓                                                    │
│  Lines 1079-1083: SAVE TO .NPY FILE ✓                          │
│    └─ np.save('rplang-v3-withsemantics/{train|val|test}/*.npy')│
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: .npy FILES ON DISK                         │
│  - 65,763 train/*.npy files                                     │
│  - 3,000 val/*.npy files                                        │
│  - 3,000 test/*.npy files                                       │
│  Each file: Python dict with 16 keys                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│         DATASET LOADER (rplang_edge_semantics_simplified_81.py) │
│                                                                 │
│  Lines 58-64: LOAD .NPY FILE ✓                                 │
│    graph = np.load(path).item()                                │
│            ↓                                                    │
│  Line 67: Extract main array                                   │
│    corners_withsemantics = graph['corner_list_np_norm...']     │
│    Shape: (53, 16)                                             │
│            ↓                                                    │
│  Lines 69-82: Simplify semantics 14→7 dimensions               │
│    Create new array (53, 9)                                    │
│    Merge categories: living, bedrooms, storage, etc.           │
│            ↓                                                    │
│  Lines 89-93: Load other data                                  │
│    ├─ global_attn_matrix (53, 53)                              │
│    ├─ padding_mask (53, 1)                                     │
│    └─ CNN features (1024, 16, 16) from separate file           │
│            ↓                                                    │
│  Line 102: RETURN TO TRAINING ✓                                │
│    return (cnn_features, corners_simplified, attn, mask)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING SCRIPT                               │
│  Receives batch of:                                             │
│    - CNN features: (batch_size, 1024, 16, 16)                  │
│    - Corners + semantics: (batch_size, 53, 9)                  │
│    - Attention matrix: (batch_size, 53, 53)                    │
│    - Padding mask: (batch_size, 53, 1)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Practical Code Examples

### 6.1 Loading and Inspecting a .npy File

```python
import numpy as np

# Load a single floor plan
graph = np.load('/path/to/rplang-v3-withsemantics/train/0.npy', allow_pickle=True).item()

# Check what keys are available
print("Available keys:", list(graph.keys()))

# Inspect file_id
print("File ID:", graph['file_id'])

# Check number of real corners
n_real_corners = int(graph['padding_mask'].sum())
print(f"Real corners: {n_real_corners}, Padding: {53 - n_real_corners}")

# Print corner coordinates with semantics
corners_with_sem = graph['corner_list_np_normalized_padding_withsemantics']
print("\nFirst 5 corners (with semantics):")
print("Format: [x, y, living, bedroom, kitchen, bathroom, dining, child, study, second, guest, balcony, entrance, storage, walkin, external]")
for i in range(min(5, n_real_corners)):
    print(f"Corner {i}: {corners_with_sem[i]}")

# Check adjacency
adj_matrix = graph['adjacency_matrix_np_padding']
n_edges = int(adj_matrix[:n_real_corners, :n_real_corners].sum() // 2)
print(f"\nNumber of edges: {n_edges}")

# Examine semantic dictionary
print("\nSemantic dictionary (first 3 corners):")
for i, (coord, sem_vector) in enumerate(list(graph['semantics'].items())[:3]):
    print(f"  {coord}: {sem_vector}")
```

### 6.2 Creating Your Own .npy File (Custom Data)

```python
import numpy as np

# Example: Create a simple 4-corner rectangular floor plan
g = {}

# File ID
g['file_id'] = 99999

# Original corners (4 corners forming a rectangle)
g['corners'] = [(50, 50), (200, 50), (200, 150), (50, 150)]
g['corners_np'] = np.array([[50, 50], [200, 50], [200, 150], [50, 150]], dtype=np.float64)

# Adjacency (rectangle: 0-1-2-3-0)
g['adjacency_matrix'] = [[0, 1, 0, 1],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [1, 0, 1, 0]]
g['adjacency_matrix_np'] = np.array(g['adjacency_matrix'], dtype=np.uint8)

g['adjacency_list'] = [[1, 3], [0, 2], [1, 3], [2, 0]]
g['adjacency_list_np'] = np.array(g['adjacency_list'], dtype=np.uint8)

# Normalize coordinates
g['corner_list_np_normalized'] = (g['corners_np'] - 128) / 128

# Pad to 53
g['corner_list_np_normalized_padding'] = np.zeros((53, 2), dtype=np.float64)
g['corner_list_np_normalized_padding'][:4] = g['corner_list_np_normalized']

# Padding mask
g['padding_mask'] = np.zeros((53, 1), dtype=np.uint8)
g['padding_mask'][:4] = 1

# Global attention matrix
g['global_matrix_np_padding'] = np.zeros((53, 53), dtype=np.uint8)
g['global_matrix_np_padding'][:4, :4] = 1

# Adjacency matrix padded
g['adjacency_matrix_np_padding'] = np.zeros((53, 53), dtype=np.uint8)
g['adjacency_matrix_np_padding'][:4, :4] = g['adjacency_matrix_np']

# Edge coordinates
edge_coord1 = np.repeat(g['corner_list_np_normalized_padding'][:, None, :], 53, axis=1)
edge_coord2 = np.repeat(g['corner_list_np_normalized_padding'][None, :, :], 53, axis=0)
g['edge_coords'] = np.concatenate((edge_coord1, edge_coord2), axis=2).reshape(-1, 4)

# Edges
g['edges'] = g['adjacency_matrix_np_padding'].reshape(-1, 1)

# Semantics (simple example: all living room)
g['semantics'] = {}
for corner in g['corner_list_np_normalized'][:4]:
    coord_tuple = (corner[0], corner[1])
    g['semantics'][coord_tuple] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Living room

# Corner with semantics
result = np.zeros((53, 16), dtype=np.float64)
for idx in range(4):
    coord = g['corner_list_np_normalized_padding'][idx]
    coord_tuple = (coord[0], coord[1])
    if coord_tuple in g['semantics']:
        vector = g['semantics'][coord_tuple]
    else:
        vector = [0] * 14
    result[idx] = np.concatenate((coord, vector))
g['corner_list_np_normalized_padding_withsemantics'] = result

# Save
np.save('custom_floor_plan.npy', g)
print("Saved custom floor plan to custom_floor_plan.npy")
```

### 6.3 Loading in Dataset Class

```python
from torch.utils.data import Dataset
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load .npy file
        graph = np.load(os.path.join(self.data_dir, self.files[index]),
                        allow_pickle=True).item()

        # Extract data
        corners_with_semantics = graph['corner_list_np_normalized_padding_withsemantics']
        padding_mask = graph['padding_mask']
        global_attn_matrix = np.ones((53, 53), dtype=np.uint8)  # Or load from graph

        # Simplify semantics (16→9)
        corners_simplified = self._simplify_semantics(corners_with_semantics)

        # Convert to tensors
        corners_simplified = torch.from_numpy(corners_simplified).float()
        padding_mask = torch.from_numpy(padding_mask).float()
        global_attn_matrix = torch.from_numpy(global_attn_matrix).bool()

        # Create dummy CNN features (or load actual features)
        feat_16 = torch.zeros(1024, 16, 16).float()

        return feat_16, corners_simplified, global_attn_matrix, padding_mask

    def _simplify_semantics(self, corners_withsemantics):
        # Same as in rplang_edge_semantics_simplified_81.py lines 69-82
        simplified = np.zeros((53, 9))
        simplified[:, 0:2] = corners_withsemantics[:, 0:2]
        simplified[:, 2] = corners_withsemantics[:, [2, 6, 12]].sum(axis=1)
        simplified[:, 3] = corners_withsemantics[:, [3, 7, 8, 9, 10]].sum(axis=1)
        simplified[:, 4] = corners_withsemantics[:, [13, 14]].sum(axis=1)
        simplified[:, 5] = corners_withsemantics[:, 4]
        simplified[:, 6] = corners_withsemantics[:, 5]
        simplified[:, 7] = corners_withsemantics[:, 11]
        simplified[:, 8] = corners_withsemantics[:, 15]
        return simplified
```

---

## Part 7: Summary & Quick Reference

### 7.1 Key Locations

| Operation | File | Lines |
|-----------|------|-------|
| **CREATE .npy** | `datasets/rplan-process4.py` | 1079-1083 |
| **BUILD dictionary** | `datasets/rplan-process4.py` | 874-1075 |
| **LOAD .npy** | `datasets/rplang_edge_semantics_simplified_81.py` | 58-64 |
| **SIMPLIFY semantics** | `datasets/rplang_edge_semantics_simplified_81.py` | 69-82 |
| **RETURN to training** | `datasets/rplang_edge_semantics_simplified_81.py` | 102 |

### 7.2 Critical Arrays

| Array Name | Shape | Purpose |
|------------|-------|---------|
| `corner_list_np_normalized_padding_withsemantics` | (53, 16) | **Main array:** coords + 14-dim semantics |
| `padding_mask` | (53, 1) | Valid corner indicator |
| `global_matrix_np_padding` | (53, 53) | Attention mask |
| `adjacency_matrix_np_padding` | (53, 53) | Edge connectivity |
| `edge_coords` | (2809, 4) | All pairwise edge coordinates |
| `edges` | (2809, 1) | Binary edge existence |

### 7.3 Data Type Reference

```python
# Integer
file_id: int

# Lists (Python native)
corners: list[tuple]
adjacency_matrix: list[list[int]]
adjacency_list: list[list[int]]

# Numpy arrays
corners_np: ndarray(n, 2), dtype=float64
adjacency_matrix_np: ndarray(n, n), dtype=uint8
corner_list_np_normalized: ndarray(n, 2), dtype=float64
corner_list_np_normalized_padding: ndarray(53, 2), dtype=float64
padding_mask: ndarray(53, 1), dtype=uint8
global_matrix_np_padding: ndarray(53, 53), dtype=uint8
adjacency_matrix_np_padding: ndarray(53, 53), dtype=uint8
edge_coords: ndarray(2809, 4), dtype=float64
edges: ndarray(2809, 1), dtype=uint8
corner_list_np_normalized_padding_withsemantics: ndarray(53, 16), dtype=float64

# Dictionary
semantics: dict[tuple[float,float], list[int]]
```

---

## Part 8: Common Questions

### Q: Why are there so many similar arrays (corners, corners_np, corner_list_np_normalized, etc.)?

**A:** Different processing stages require different formats:
- `corners` → Original list format
- `corners_np` → Numpy array format (for computation)
- `corner_list_np_normalized` → Normalized to [-1, 1]
- `corner_list_np_normalized_padding` → Padded to fixed size (53)
- `corner_list_np_normalized_padding_withsemantics` → With semantic labels added

### Q: Why pad to 53?

**A:** Neural networks need fixed-size inputs for batching. 53 is the maximum number of corners observed in the RPLAN dataset.

### Q: Why are semantics stored twice (in dictionary and array)?

**A:**
- Dictionary `semantics`: Easy lookup by coordinates during preprocessing
- Array `corner_list_np_normalized_padding_withsemantics`: Efficient for neural network input

### Q: Can I create my own .npy files?

**A:** Yes! Follow the structure shown in Section 6.2. The minimum required keys are:
- `corner_list_np_normalized_padding_withsemantics` (53, 16)
- `padding_mask` (53, 1)
- `global_matrix_np_padding` (53, 53)

### Q: What if my floor plans have more than 53 corners?

**A:** You need to:
1. Change padding size in preprocessing (e.g., 100 instead of 53)
2. Modify all arrays to match new size
3. Update model architecture to accept new size
4. Retrain from scratch

---

**End of Report**

---

## Quick Command Reference

```bash
# Find where .npy files are created
grep -n "np.save" datasets/rplan-process4.py

# Find where .npy files are loaded
grep -n "np.load" datasets/rplang_edge_semantics_simplified_81.py

# Check structure of a .npy file
python3 -c "import numpy as np; g = np.load('train/0.npy', allow_pickle=True).item(); print(list(g.keys()))"

# Verify all .npy files have same structure
python3 -c "import numpy as np, os; files = os.listdir('train/')[:10]; [print(f'{f}: {list(np.load(f\"train/{f}\", allow_pickle=True).item().keys())}') for f in files]"
```
