# Raster-to-Graph Converter Guide

## Problem Summary

You have **raster NPY files** (2D arrays from flood-fill) but GSDiff expects **graph NPY files** (dictionaries with corners, edges, and semantics).

**Error you're seeing:**
```
ValueError: Unexpected numpy array format: shape=(2575, 1393), dtype=int32, size=3586975
```

This means your NPY files contain raw 2D arrays instead of the graph dictionaries GSDiff needs.

## Solution

Use the `raster_to_graph_converter.py` script to convert your flood-filled raster data to GSDiff's graph format.

## How It Works

```
Flood-Fill Raster                    Graph Structure
(2D array)                           (Dictionary)
┌─────────────────┐                  ┌──────────────────────┐
│ 0 0 0 0 0 0 0 0 │                  │ corners: [(x1,y1),   │
│ 0 1 1 1 0 2 2 0 │   ──────────>    │          (x2,y2)...] │
│ 0 1 1 1 0 2 2 0 │                  │ edges: adjacency mtx │
│ 0 0 0 0 0 0 0 0 │                  │ semantics: room types│
└─────────────────┘                  └──────────────────────┘
0 = walls
1,2,3... = room IDs                   Padded to 53 corners
```

### Conversion Steps

1. **Extract Room Boundaries** - Find contours for each room ID using OpenCV
2. **Simplify Polygons** - Reduce corners using Douglas-Peucker algorithm
3. **Build Graph** - Create corners list and adjacency matrix from polygons
4. **Add Semantics** - Map room IDs to semantic vectors (room types)
5. **Normalize & Pad** - Normalize coords to [-1, 1], pad to 53 corners
6. **Save Dictionary** - Package everything in the required GSDiff format

## Usage

### Basic Usage

```bash
python raster_to_graph_converter.py \
    --input_dir path/to/your/raster_npy_files \
    --output_dir datasets/rplang-v3-withsemantics/test
```

### With Validation

```bash
python raster_to_graph_converter.py \
    --input_dir path/to/your/raster_npy_files \
    --output_dir datasets/rplang-v3-withsemantics/test \
    --image_size 256 \
    --validate
```

### Arguments

- `--input_dir`: Directory containing your flood-filled raster NPY files
- `--output_dir`: Where to save converted graph NPY files
- `--image_size`: Size of your raster images (default: 256)
  - If your rasters are 2575×1393, use `--image_size 2575`
- `--validate`: Run validation after conversion to ensure correctness

## Input Format

Your raster NPY files should contain 2D numpy arrays where:

```python
# Shape: (height, width)
# Values:
#   0 = walls/boundaries
#   1 = room 1 (e.g., living room)
#   2 = room 2 (e.g., bedroom)
#   3 = room 3 (e.g., kitchen)
#   ... and so on
```

**Example:**
```python
import numpy as np

# Your flood-filled raster
raster = np.load('0.npy')
print(raster.shape)  # (2575, 1393)
print(np.unique(raster))  # [0, 1, 2, 3, 4, 5] - 0=walls, 1-5=rooms
```

## Output Format

The converter creates NPY files containing dictionaries with all GSDiff-required fields:

```python
{
    'file_id': 0,
    'corners': [(x1, y1), (x2, y2), ...],
    'adjacency_matrix': [[0, 1, 0, ...], ...],
    'adjacency_list': [[1, 2], [0, 3], ...],
    'corners_np': array([[x1, y1], [x2, y2], ...]),
    'corner_list_np_normalized': array([...]),  # Normalized to [-1, 1]
    'corner_list_np_normalized_padding': array((53, 2)),
    'padding_mask': array((53, 1)),  # 1=real, 0=padding
    'global_matrix_np_padding': array((53, 53)),
    'adjacency_matrix_np_padding': array((53, 53)),
    'edges': array((53*53, 1)),
    'edge_coords': array((53*53, 4)),
    'semantics': {(x1,y1): [0,1,0,...], ...},
    'corner_list_np_normalized_padding_withsemantics': array((53, 16))
}
```

## Room Type Mapping

The converter uses this room type mapping by default:

```python
1  -> 'LivingRoom'
2  -> 'MasterRoom' (Master bedroom)
3  -> 'Kitchen'
4  -> 'Bathroom'
5  -> 'DiningRoom'
6  -> 'ChildRoom' (Child bedroom)
7  -> 'StudyRoom'
8  -> 'SecondRoom' (Second bedroom)
9  -> 'GuestRoom'
10 -> 'Balcony'
11 -> 'Entrance'
12 -> 'Storage'
13 -> 'WallIn' (Walk-in closet)
14 -> 'External'
```

**To customize:** Edit `ROOM_TYPE_MAPPING` in `raster_to_graph_converter.py`

## Customization

### Adjust Polygon Simplification

If you're getting too many corners (>53), or polygons look too angular:

```python
# In raster_to_graph_converter.py, line ~100
def extract_room_boundaries(raster_array, room_id, simplify_tolerance=2.0):
    # Increase simplify_tolerance to reduce corners
    # Default: 2.0
    # For simpler polygons: 4.0 or 5.0
    # For more detailed: 1.0
```

### Change Maximum Corners

```python
# In raster_to_graph_converter.py, line ~145
room_polygons = extract_all_rooms(raster_array, max_corners_per_room=15)
# Reduce max_corners_per_room if total exceeds 53
# Default: 15
# For simpler: 10 or 8
```

### Adjust Image Size

If your rasters are not 256×256:

```bash
python raster_to_graph_converter.py \
    --input_dir your_input \
    --output_dir your_output \
    --image_size 2575  # <-- Use your actual image size
```

## Complete Workflow

### 1. Check Your Current Data

```bash
python -c "
import numpy as np
data = np.load('path/to/your/0.npy', allow_pickle=True)
print('Type:', type(data))
print('Shape:', data.shape if hasattr(data, 'shape') else 'N/A')
print('Unique values:', np.unique(data) if hasattr(data, 'shape') else 'N/A')
"
```

**Expected output for raster:**
```
Type: <class 'numpy.ndarray'>
Shape: (2575, 1393)  # or similar
Unique values: [0 1 2 3 4 5]  # 0=walls, 1-5=rooms
```

### 2. Convert Your Data

```bash
# Create output directory
mkdir -p datasets/rplang-v3-withsemantics/test

# Run converter
python raster_to_graph_converter.py \
    --input_dir path/to/your/flood_filled_npys \
    --output_dir datasets/rplang-v3-withsemantics/test \
    --image_size 2575 \
    --validate
```

### 3. Verify Conversion

```bash
python -c "
import numpy as np
data = np.load('datasets/rplang-v3-withsemantics/test/0.npy', allow_pickle=True)
graph = data.item()
print('Type:', type(graph))
print('Keys:', list(graph.keys()))
print('Corners shape:', graph['corner_list_np_normalized_padding_withsemantics'].shape)
print('✓ Correct format!')
"
```

**Expected output:**
```
Type: <class 'dict'>
Keys: ['file_id', 'corners', 'adjacency_matrix', ...]
Corners shape: (53, 16)
✓ Correct format!
```

### 4. Run test_boun.py

```bash
python scripts/test_boun.py
```

Should now work without the "Unexpected numpy array format" error!

## Troubleshooting

### Error: "No rooms found in file"

**Problem:** The raster array has no room IDs (only walls/0s)

**Solution:**
- Check your flood-fill output
- Ensure room IDs are 1, 2, 3, ... (not all zeros)
- Visualize: `import matplotlib.pyplot as plt; plt.imshow(raster); plt.show()`

### Error: "Too many corners"

**Problem:** Total corners across all rooms exceeds 53

**Solution:**
1. Increase `simplify_tolerance` (e.g., from 2.0 to 5.0)
2. Decrease `max_corners_per_room` (e.g., from 15 to 8)
3. Edit the converter:
   ```python
   # Line ~145 in raster_to_graph_converter.py
   room_polygons = extract_all_rooms(raster_array, max_corners_per_room=8)
   # And line ~100
   epsilon = 5.0  # Increase from 2.0
   ```

### Warning: "Truncating to 53 corners"

**Problem:** After simplification, still too many corners

**Solution:** This is OK! The converter automatically truncates. But for better results:
- Merge small rooms in your flood-fill process
- Increase simplification tolerance
- Reduce corners per room

### Coordinates Out of Range

**Problem:** Corners outside [-1, 1] range

**Solution:** Ensure `--image_size` matches your actual raster dimensions:
```bash
python raster_to_graph_converter.py \
    --input_dir input \
    --output_dir output \
    --image_size 2575  # Match your actual image width/height
```

## Comparison: Old vs New Approach

### OLD (json_to_npy_converter.py)
```
JSON bounding boxes → Sort wall endpoints by angle → Create polygons
```
**Problem:** Misshapen rooms, walls not connected, triangular artifacts

### NEW (Your flood-fill + This converter)
```
JSON → Draw ALL walls as raster → Gap closing → Flood fill → Extract boundaries → Graph
```
**Advantage:** Clean boundaries, all walls connected, accurate room shapes

## Advanced: Customize Room Semantics

If you have custom room type metadata:

```python
# In raster_to_graph_converter.py

# Option 1: Load from JSON file
with open('room_metadata.json', 'r') as f:
    room_metadata = json.load(f)

def create_semantic_vector(corner, corner_to_rooms, room_polygons):
    vector = [0] * 14
    rooms = corner_to_rooms.get(corner, set())

    for room_id in rooms:
        # Use your metadata
        room_type = room_metadata[str(room_id)]['type']
        if room_type in SEMANTIC_INDICES:
            idx = SEMANTIC_INDICES[room_type]
            vector[idx] = 1

    return vector
```

## Testing with Subset

Test with a few files first:

```bash
# Copy 5 files to test directory
mkdir test_input test_output
cp path/to/your/rasters/{0,1,2,3,4}.npy test_input/

# Convert
python raster_to_graph_converter.py \
    --input_dir test_input \
    --output_dir test_output \
    --validate

# If successful, convert all files
python raster_to_graph_converter.py \
    --input_dir path/to/all/rasters \
    --output_dir datasets/rplang-v3-withsemantics/test \
    --validate
```

## Summary

1. ✅ Your flood-fill approach solves the disconnected walls problem
2. ✅ This converter transforms raster → graph format
3. ✅ Handles boundary extraction, simplification, and semantic mapping
4. ✅ Outputs exact GSDiff format with all required fields
5. ✅ Validates output to ensure compatibility

Run the converter, and your data will work with GSDiff!
