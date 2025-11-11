# JSON to GSDiff .npy Conversion Guide

**Script:** `json_to_npy_converter.py`

**Purpose:** Convert JSON floor plan files into GSDiff-compatible `.npy` format for training or inference.

---

## Quick Start

```bash
# Convert all JSON files in a directory
python json_to_npy_converter.py \
    --input_dir data/my_json_files \
    --output_dir data/my_npy_files

# Convert a single file
python json_to_npy_converter.py \
    --input data/floor_plan.json \
    --output data/floor_plan.npy
```

---

## Supported JSON Formats

The script automatically detects and handles three JSON formats:

### Format 1: Node-Edge Format

Best for: Graph data with explicit nodes and edges

```json
{
    "nodes": [
        {"id": 0, "x": 50, "y": 50, "semantic": 0},
        {"id": 1, "x": 200, "y": 50, "semantic": 0},
        {"id": 2, "x": 200, "y": 150, "semantic": 2},
        {"id": 3, "x": 50, "y": 150, "semantic": 2}
    ],
    "edges": [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 0}
    ]
}
```

**Variations supported:**
- Coordinates: `x`/`y`, `pos`, `position`, `coordinates`
- Edges: `source`/`target`, `from`/`to`, `[src, tgt]` arrays
- Semantics: `semantic`, `type`, `label`

### Format 2: Corners-Adjacency Format

Best for: Pre-processed graph data with adjacency information

```json
{
    "corners": [
        [50, 50],
        [200, 50],
        [200, 150],
        [50, 150]
    ],
    "adjacency": [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ],
    "semantics": [0, 0, 2, 2]
}
```

**Variations supported:**

**With adjacency list:**
```json
{
    "corners": [[50, 50], [200, 50], ...],
    "adjacency_list": {
        "0": [1, 3],
        "1": [0, 2],
        "2": [1, 3],
        "3": [2, 0]
    },
    "semantics": [0, 0, 2, 2]
}
```

**With adjacency matrix:**
```json
{
    "corners": [[50, 50], [200, 50], ...],
    "adjacency_matrix": [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ],
    "semantics": [0, 0, 2, 2]
}
```

### Format 3: Rooms Format

Best for: Room-based floor plans where each room has its own corners

```json
{
    "rooms": [
        {
            "type": "living_room",
            "corners": [
                [0, 0],
                [100, 0],
                [100, 100],
                [0, 100]
            ]
        },
        {
            "type": "kitchen",
            "corners": [
                [100, 0],
                [200, 0],
                [200, 100],
                [100, 100]
            ]
        }
    ]
}
```

**Room types supported:**
- String names: `"living_room"`, `"kitchen"`, `"bedroom"`, etc.
- Integer labels: `0` (living), `1` (bedroom), `2` (kitchen), etc.

**Note:** Shared corners between rooms are automatically detected and merged.

---

## Semantic Label Reference

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

## Usage Examples

### 1. Basic Conversion

```bash
python json_to_npy_converter.py \
    --input_dir data/json_files \
    --output_dir data/npy_files
```

**Output:**
- `data/npy_files/0.npy`
- `data/npy_files/1.npy`
- `data/npy_files/2.npy`
- ...

Files are automatically numbered sequentially (0, 1, 2, ...) regardless of input filenames. A mapping is displayed after conversion.

### 2. Single File Conversion

```bash
python json_to_npy_converter.py \
    --input data/my_floor_plan.json \
    --output data/my_floor_plan.npy
```

### 3. Specify Coordinate Range

If your coordinates are in a different range than [0, 256]:

```bash
python json_to_npy_converter.py \
    --input_dir data/json_files \
    --output_dir data/npy_files \
    --coord_range 0 1000
```

This normalizes coordinates from [0, 1000] → [-1, 1]

### 4. Change Maximum Corners

If your floor plans have more than 53 corners:

```bash
python json_to_npy_converter.py \
    --input_dir data/json_files \
    --output_dir data/npy_files \
    --max_corners 100
```

**Important:** If you change `max_corners`, you must also modify the model architecture!

### 5. Train/Val/Test Split

Create a JSON file with split:

```json
{
  "train": [0, 1, 2, 3, 4, 5, 6, 7],
  "val": [8, 9],
  "test": [10, 11]
}
```

Then convert with split:

```bash
python json_to_npy_converter.py \
    --input_dir data/json_files \
    --output_dir data/npy_files \
    --split_file split.json
```

**Output:**
- `data/npy_files/train/0.npy`, `train/1.npy`, ...
- `data/npy_files/val/8.npy`, `val/9.npy`
- `data/npy_files/test/10.npy`, `test/11.npy`

### 6. Custom Semantic Dimensions

If you have different number of room types:

```bash
python json_to_npy_converter.py \
    --input_dir data/json_files \
    --output_dir data/npy_files \
    --semantic_dim 7
```

---

## Creating JSON Files

### Example 1: Node-Edge Format (Python)

```python
import json

floor_plan = {
    "nodes": [
        {"id": 0, "x": 0, "y": 0, "semantic": 0},
        {"id": 1, "x": 100, "y": 0, "semantic": 0},
        {"id": 2, "x": 100, "y": 100, "semantic": 0},
        {"id": 3, "x": 0, "y": 100, "semantic": 0}
    ],
    "edges": [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 0}
    ]
}

with open('floor_plan.json', 'w') as f:
    json.dump(floor_plan, f, indent=2)
```

### Example 2: Corners-Adjacency Format (Python)

```python
import json

floor_plan = {
    "corners": [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
    ],
    "adjacency": [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ],
    "semantics": [0, 0, 2, 2]
}

with open('floor_plan.json', 'w') as f:
    json.dump(floor_plan, f, indent=2)
```

### Example 3: Rooms Format (Python)

```python
import json

floor_plan = {
    "rooms": [
        {
            "type": "living_room",
            "corners": [[0, 0], [100, 0], [100, 100], [0, 100]]
        },
        {
            "type": "kitchen",
            "corners": [[100, 0], [200, 0], [200, 100], [100, 100]]
        },
        {
            "type": "bedroom",
            "corners": [[0, 100], [100, 100], [100, 200], [0, 200]]
        }
    ]
}

with open('floor_plan.json', 'w') as f:
    json.dump(floor_plan, f, indent=2)
```

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

### Issue 1: "Unrecognized JSON format"

**Problem:** Your JSON doesn't match any supported format.

**Solution:** Check that your JSON has one of:
- `"nodes"` and `"edges"` keys
- `"corners"` and (`"adjacency"` or `"adjacency_list"` or `"adjacency_matrix"`) keys
- `"rooms"` key

### Issue 2: Coordinates Outside Expected Range

**Problem:** Coordinates are in a different range (e.g., [0, 1] or [0, 10000])

**Solution:** Use `--coord_range`:

```bash
# For coordinates in [0, 1]
python json_to_npy_converter.py ... --coord_range 0 1

# For coordinates in [0, 10000]
python json_to_npy_converter.py ... --coord_range 0 10000
```

### Issue 3: More than 53 Corners

**Problem:** "Graph has 75 corners, which exceeds max_corners=53"

**Solution 1:** Increase max_corners:
```bash
python json_to_npy_converter.py ... --max_corners 100
```

**Solution 2:** Simplify your graph to have ≤53 corners before conversion.

**Note:** Changing max_corners requires retraining the model!

### Issue 4: Missing Semantic Labels

**Problem:** Your JSON doesn't have semantic information.

**Solution:** The script automatically assigns default semantic label (0 = living room). You can add semantics to your JSON:

**Node-Edge format:**
```json
{"nodes": [{"id": 0, "x": 50, "y": 50, "semantic": 2}, ...], ...}
```

**Corners-Adjacency format:**
```json
{"corners": [[50, 50], ...], "adjacency": [[0, 1], ...], "semantics": [0, 2, 2, 1]}
```

**Rooms format:**
```json
{"rooms": [{"type": "kitchen", "corners": [...]}, ...]}
```

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

After conversion, use your `.npy` files with GSDiff (same as RPLAN data):

1. **Update dataset paths** in `datasets/path_utils.py`
2. **Use existing dataset loaders** (e.g., `RPlanGEdgeSemanSimplified_81`)
3. **Pre-compute CNN features** (optional, for boundary constraints)
4. **Train** with `trainval_main_*.py`

See `CUSTOM_DATASET_GUIDE.md` for complete training instructions.

---

## Example Workflow

Complete workflow from JSON to trained model:

```bash
# 1. Create JSON files with your floor plan data
# (See examples above)

# 2. Convert to .npy format
python json_to_npy_converter.py \
    --input_dir data/my_json_files \
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

# 4. Pre-compute CNN features (optional)
# python scripts/prerunningCNN.py --dataset my-data

# 5. Train GSDiff
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

## Comparison: JSON vs NetworkX gpickle

| Aspect | JSON | NetworkX gpickle |
|--------|------|------------------|
| **Format** | Human-readable text | Python binary pickle |
| **Portability** | Works across languages | Python-only |
| **Editing** | Easy (any text editor) | Requires Python |
| **Size** | Larger file size | Smaller file size |
| **Speed** | Slower to parse | Faster to load |
| **Use Case** | Web APIs, data exchange | Python-native workflows |

Both formats convert to the same `.npy` output structure!

---

## Troubleshooting

### Script Won't Run

**Check Python version:**
```bash
python --version  # Should be 3.7+
```

**Install dependencies:**
```bash
pip install numpy tqdm
```

### Conversion Fails

**Check JSON syntax:**
```bash
python -m json.tool your_file.json
```

**Check JSON structure:**
```python
import json
with open('your_file.json') as f:
    data = json.load(f)
print(data.keys())
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
- See `CUSTOM_DATASET_GUIDE.md` for training on custom data
- See `GPICKLE_TO_NPY_CONVERSION_GUIDE.md` for NetworkX conversion

---

**End of Guide**
