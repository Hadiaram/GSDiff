# NetworkX Graph to GSDiff .npy Conversion Guide

**Script:** `gpickle_to_npy_converter.py`

**Purpose:** Convert NetworkX graphs (saved as `.gpickle` files) into GSDiff-compatible `.npy` format for training or inference.

---

## Quick Start

```bash
# Convert all .gpickle files in a directory
python gpickle_to_npy_converter.py \
    --input_dir data/my_graphs \
    --output_dir data/my_npy_files

# Convert a single file
python gpickle_to_npy_converter.py \
    --input data/graph_0.gpickle \
    --output data/graph_0.npy
```

---

## NetworkX Graph Requirements

Your NetworkX graphs MUST have:

### Required Node Attributes

1. **Position coordinates** (one of):
   - `x` and `y` attributes (preferred)
   - OR `pos` attribute as tuple `(x, y)`

### Optional Node Attributes

2. **Semantic labels** (recommended):
   - `semantic` attribute with integer value 0-13
   - Represents room type (living room, bedroom, kitchen, etc.)
   - If not provided, defaults to 0 (living room)

### Required Graph Structure

3. **Edges** represent walls/connections between corners
4. **Nodes** represent corners (junctions where walls meet)

---

## Example: Creating Compatible NetworkX Graphs

### Example 1: Simple Rectangular Floor Plan

```python
import networkx as nx
import numpy as np

# Create graph
G = nx.Graph()

# Add 4 corners (rectangle)
G.add_node(0, x=50, y=50, semantic=0)    # Living room corner
G.add_node(1, x=200, y=50, semantic=0)   # Living room corner
G.add_node(2, x=200, y=150, semantic=2)  # Kitchen corner
G.add_node(3, x=50, y=150, semantic=2)   # Kitchen corner

# Add edges (walls)
G.add_edge(0, 1)  # Top wall
G.add_edge(1, 2)  # Right wall
G.add_edge(2, 3)  # Bottom wall
G.add_edge(3, 0)  # Left wall

# Save to gpickle
nx.write_gpickle(G, 'floor_plan_0.gpickle')
```

### Example 2: Using `pos` Attribute

```python
import networkx as nx

G = nx.Graph()

# Add nodes with 'pos' tuple
G.add_node(0, pos=(50, 50), semantic=0)
G.add_node(1, pos=(200, 50), semantic=0)
G.add_node(2, pos=(200, 150), semantic=2)
G.add_node(3, pos=(50, 150), semantic=2)

# Add edges
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
G.add_edges_from(edges)

nx.write_gpickle(G, 'floor_plan_1.gpickle')
```

### Example 3: Complex Floor Plan with Rooms

```python
import networkx as nx

G = nx.Graph()

# Living room corners (semantic=0)
G.add_node(0, x=0, y=0, semantic=0)
G.add_node(1, x=100, y=0, semantic=0)
G.add_node(2, x=100, y=100, semantic=0)
G.add_node(3, x=0, y=100, semantic=0)

# Kitchen corners (semantic=2)
G.add_node(4, x=100, y=0, semantic=2)
G.add_node(5, x=150, y=0, semantic=2)
G.add_node(6, x=150, y=100, semantic=2)
G.add_node(7, x=100, y=100, semantic=2)

# Bedroom corners (semantic=1)
G.add_node(8, x=0, y=100, semantic=1)
G.add_node(9, x=100, y=100, semantic=1)
G.add_node(10, x=100, y=200, semantic=1)
G.add_node(11, x=0, y=200, semantic=1)

# Add edges to form rooms
# Living room edges
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
# Kitchen edges
G.add_edges_from([(4, 5), (5, 6), (6, 7), (7, 4)])
# Bedroom edges
G.add_edges_from([(8, 9), (9, 10), (10, 11), (11, 8)])
# Connections between rooms
G.add_edge(2, 7)  # Living room to kitchen
G.add_edge(3, 8)  # Living room to bedroom

nx.write_gpickle(G, 'complex_floor_plan.gpickle')
```

---

## Usage Examples

### 1. Basic Conversion

```bash
python gpickle_to_npy_converter.py \
    --input_dir data/graphs \
    --output_dir data/npy_files
```

**Output:**
- `data/npy_files/0.npy`
- `data/npy_files/1.npy`
- `data/npy_files/2.npy`
- ...

### 2. Specify Coordinate Range

If your coordinates are not in [0, 256] range:

```bash
python gpickle_to_npy_converter.py \
    --input_dir data/graphs \
    --output_dir data/npy_files \
    --coord_range 0 1000
```

This normalizes coordinates from [0, 1000] → [-1, 1]

### 3. Change Maximum Corners

If your floor plans have more than 53 corners:

```bash
python gpickle_to_npy_converter.py \
    --input_dir data/graphs \
    --output_dir data/npy_files \
    --max_corners 100
```

**Important:** If you change `max_corners`, you must also modify the model architecture to match!

### 4. Train/Val/Test Split

Create a JSON file with split:

```json
{
  "train": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "val": [10, 11],
  "test": [12, 13, 14]
}
```

Then convert with split:

```bash
python gpickle_to_npy_converter.py \
    --input_dir data/graphs \
    --output_dir data/npy_files \
    --split_file split.json
```

**Output:**
- `data/npy_files/train/0.npy`, `train/1.npy`, ...
- `data/npy_files/val/10.npy`, `val/11.npy`
- `data/npy_files/test/12.npy`, ...

### 5. Single File Conversion

```bash
python gpickle_to_npy_converter.py \
    --input data/graph_0.gpickle \
    --output data/graph_0.npy \
    --max_corners 53
```

---

## Semantic Label Reference

The semantic labels (0-13) correspond to room types:

| Label | Room Type |
|-------|-----------|
| 0 | Living room / Dining / Entrance |
| 1 | Master bedroom |
| 2 | Kitchen |
| 3 | Bathroom |
| 4 | Dining room (specific) |
| 5 | Child room |
| 6 | Study room |
| 7 | Second bedroom |
| 8 | Guest room |
| 9 | Balcony |
| 10 | Entrance (specific) |
| 11 | Storage / Storeroom |
| 12 | Walk-in closet |
| 13 | External area |

---

## Output .npy File Structure

The script creates `.npy` files with the same structure as RPLAN preprocessing:

```python
{
    'file_id': int,
    'corners': list,
    'corners_np': ndarray(n, 2),
    'adjacency_matrix': list,
    'adjacency_list': dict,
    'corner_list_np_normalized': ndarray(n, 2),
    'corner_list_np_normalized_padding': ndarray(53, 2),
    'padding_mask': ndarray(53, 1),
    'global_matrix_np_padding': ndarray(53, 53),
    'adjacency_matrix_np_padding': ndarray(53, 53),
    'edge_coords': ndarray(2809, 4),
    'edges': ndarray(2809, 1),
    'semantics': dict,
    'corner_list_np_normalized_padding_withsemantics': ndarray(53, 16)
}
```

See `NPY_FILE_STRUCTURE_COMPLETE_REFERENCE.md` for detailed documentation.

---

## Common Issues & Solutions

### Issue 1: "Nodes must have 'x' and 'y' attributes"

**Problem:** Your NetworkX graph doesn't have position attributes.

**Solution:** Add `x` and `y` attributes to all nodes:

```python
for node in G.nodes():
    G.nodes[node]['x'] = x_coordinate
    G.nodes[node]['y'] = y_coordinate
```

### Issue 2: Coordinates Outside Expected Range

**Problem:** Your coordinates are in a different range (e.g., [0, 1] or [0, 10000])

**Solution:** Use `--coord_range` to specify your range:

```bash
# For coordinates in [0, 1]
python gpickle_to_npy_converter.py ... --coord_range 0 1

# For coordinates in [0, 10000]
python gpickle_to_npy_converter.py ... --coord_range 0 10000
```

### Issue 3: More than 53 Corners

**Problem:** "Graph has 75 corners, which exceeds max_corners=53"

**Solution 1:** Increase max_corners:

```bash
python gpickle_to_npy_converter.py ... --max_corners 100
```

**Solution 2:** Simplify your graph to have ≤53 corners before conversion.

**Note:** If you change max_corners, you must retrain the model from scratch or modify the architecture!

### Issue 4: No Semantic Labels

**Problem:** Your graphs don't have semantic information.

**Solution:** The script automatically assigns default semantic label (0 = living room). You can add semantics later:

```python
# After loading graph
G = nx.read_gpickle('graph.gpickle')

# Assign semantics based on your data
for node in G.nodes():
    if some_condition:
        G.nodes[node]['semantic'] = 2  # Kitchen
    else:
        G.nodes[node]['semantic'] = 0  # Living room

# Save updated graph
nx.write_gpickle(G, 'graph_with_semantics.gpickle')
```

### Issue 5: Disconnected Graph Components

**Problem:** Your graph has multiple disconnected components (separate rooms with no connections).

**Solution:** The script handles this automatically. It will:
1. Detect each connected component
2. Treat each as a separate room
3. Assign semantics based on node attributes

---

## Verifying Conversion

After conversion, verify your `.npy` files:

```python
import numpy as np

# Load converted file
graph = np.load('data/npy_files/0.npy', allow_pickle=True).item()

# Check structure
print("Keys:", list(graph.keys()))

# Check number of corners
n_corners = int(graph['padding_mask'].sum())
print(f"Number of corners: {n_corners}")

# Check coordinates
print("First 5 corners:")
print(graph['corner_list_np_normalized_padding'][:5])

# Check semantics
print("\nCorners with semantics (first 3):")
print(graph['corner_list_np_normalized_padding_withsemantics'][:3])

# Check edges
adj = graph['adjacency_matrix_np_padding']
n_edges = int(adj[:n_corners, :n_corners].sum() // 2)
print(f"\nNumber of edges: {n_edges}")
```

---

## Integration with GSDiff Training

After conversion, use your `.npy` files with GSDiff:

### Step 1: Update Dataset Loader

Modify `datasets/path_utils.py` to point to your data:

```python
def get_data_path(dataset_name, mode, filename=None):
    if dataset_name == 'my-custom-data':
        base_path = '/path/to/data/npy_files'
        # ... rest of logic
```

### Step 2: Use Existing Dataset Class

The converted `.npy` files work with existing dataset loaders:

```python
from datasets.rplang_edge_semantics_simplified_81 import RPlanGEdgeSemanSimplified_81

# Update the dataset class to point to your data directory
dataset = RPlanGEdgeSemanSimplified_81('train')
```

### Step 3: Pre-compute CNN Features (Optional)

If using boundary-constrained generation:

```bash
python scripts/prerunningCNN.py --dataset my-custom-data
```

### Step 4: Train

```bash
python scripts/trainval_main_unconstrained.py --dataset my-custom-data
```

---

## Advanced Usage

### Custom Semantic Dimensions

If you have different room types (e.g., only 5 types):

```bash
python gpickle_to_npy_converter.py \
    --input_dir data/graphs \
    --output_dir data/npy_files \
    --semantic_dim 5
```

**Note:** You must also modify the dataset loader and model to handle 5 semantic dimensions instead of 14!

### Programmatic Usage

```python
from gpickle_to_npy_converter import convert_gpickle_to_npy

# Convert single file
result = convert_gpickle_to_npy(
    gpickle_path='data/graph_0.gpickle',
    output_path='data/graph_0.npy',
    max_corners=53,
    semantic_dim=14,
    coord_range=(0, 256)
)

print(f"Converted with {result['padding_mask'].sum()} corners")
```

### Batch Conversion in Python

```python
from gpickle_to_npy_converter import convert_directory

stats = convert_directory(
    input_dir='data/graphs',
    output_dir='data/npy_files',
    max_corners=53,
    semantic_dim=14,
    coord_range=(0, 256)
)

print(f"Converted: {stats['converted']}, Failed: {stats['failed']}")
```

---

## Comparison: RPLAN vs Custom Data

| Aspect | RPLAN | Your Custom Data |
|--------|-------|------------------|
| **Source Format** | PNG images (256×256) | NetworkX graphs (.gpickle) |
| **Corner Detection** | Automatic (image processing) | From graph nodes |
| **Edge Detection** | Automatic (wall pixels) | From graph edges |
| **Semantic Labels** | From image channels | From node attributes |
| **Preprocessing** | rplan-extract.py + rplan-process1-4.py | gpickle_to_npy_converter.py |
| **Output Format** | .npy dictionaries | .npy dictionaries (same!) |
| **Compatibility** | Direct GSDiff training | Direct GSDiff training |

---

## Troubleshooting

### Script Won't Run

**Check Python version:**
```bash
python --version  # Should be 3.7+
```

**Install dependencies:**
```bash
pip install networkx numpy tqdm
```

### Conversion Fails

**Enable verbose output:**

Modify script to add debugging:
```python
# In convert_gpickle_to_npy function, add:
print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
print(f"Sample node data: {G.nodes[list(G.nodes())[0]]}")
```

### File Format Issues

**Verify your .gpickle file:**
```python
import networkx as nx

G = nx.read_gpickle('your_file.gpickle')
print(f"Nodes: {len(G.nodes())}")
print(f"Edges: {len(G.edges())}")
print(f"Sample node attributes: {G.nodes[list(G.nodes())[0]]}")
```

---

## Next Steps

After converting your data:

1. ✅ **Verify** conversion with the verification script above
2. ✅ **Update** dataset paths in `path_utils.py`
3. ✅ **Pre-compute** CNN features (if using boundary constraints)
4. ✅ **Train** your model with `trainval_main_*.py`

For more information:
- See `NPY_FILE_STRUCTURE_COMPLETE_REFERENCE.md` for .npy format details
- See `DATA_GENERATION_PIPELINE.md` for RPLAN preprocessing pipeline
- See `CUSTOM_DATASET_GUIDE.md` for training on custom data

---

## Example Workflow

Complete workflow from NetworkX graphs to trained model:

```bash
# 1. Create NetworkX graphs and save as .gpickle files
# (Your code here - see examples above)

# 2. Convert to .npy format
python gpickle_to_npy_converter.py \
    --input_dir data/my_graphs \
    --output_dir data/my_npy \
    --max_corners 53 \
    --coord_range 0 256 \
    --split_file split.json

# 3. Verify conversion
python -c "
import numpy as np
g = np.load('data/my_npy/train/0.npy', allow_pickle=True).item()
print(f'Corners: {int(g[\"padding_mask\"].sum())}')
print('Success!')
"

# 4. Pre-compute CNN features (optional, for boundary constraints)
# python scripts/prerunningCNN.py --dataset my-data

# 5. Train GSDiff on your data
# python scripts/trainval_main_unconstrained.py \
#     --dataset my-data \
#     --batch_size 256 \
#     --steps 1000000

# 6. Generate floor plans
# python scripts/test_main.py \
#     --model_path outputs/my-model/model.pt \
#     --num_samples 100
```

---

**End of Guide**

For questions or issues, refer to the comprehensive documentation in the repository.
