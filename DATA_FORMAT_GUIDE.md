# GSDiff Data Format Guide

## Problem: Wrong Data Format

If you're getting an error like:
```
ValueError: Unexpected numpy array format: shape=(2575, 1393), dtype=int32, size=3586975
ERROR: Loaded numpy file contains a raw array instead of a dictionary.
```

This means your numpy files are in the **wrong format**. You're saving raw arrays (like images or masks) instead of the required dictionary structure.

## Quick Solution: Raster-to-Graph Converter

**If you have flood-filled raster NPY files (2D arrays with room IDs), use the automated converter:**

```bash
python raster_to_graph_converter.py \
    --input_dir path/to/your/raster_npys \
    --output_dir datasets/rplang-v3-withsemantics/test \
    --image_size 2575 \  # Your raster dimensions
    --validate
```

**See [RASTER_TO_GRAPH_GUIDE.md](RASTER_TO_GRAPH_GUIDE.md) for detailed instructions.**

The converter automatically:
- ✅ Extracts room boundaries from raster data
- ✅ Builds graph structure (corners, edges, adjacency)
- ✅ Adds semantic information
- ✅ Normalizes and pads to GSDiff format
- ✅ Validates output

---

## Manual Approach: Understanding the Format

If you're creating a custom converter or need to understand the exact format:

## Required Data Format

Each `.npy` file must contain a **Python dictionary** (not a raw numpy array) with the following structure:

```python
{
    'file_id': int,                                           # File identifier
    'corner_list_np_normalized_padding_withsemantics': ndarray,  # Shape: (53, 16), float
    'padding_mask': ndarray,                                  # Shape: (53, 1), uint8
    'global_matrix_np_padding': ndarray,                      # Shape: (53, 53), bool or uint8
    'edges': ndarray,                                         # Edge adjacency data
    'semantics': dict or ndarray                              # Room semantic information
}
```

### Field Descriptions

1. **`corner_list_np_normalized_padding_withsemantics`** (required)
   - Shape: `(53, 16)`
   - Type: `float` (normalized between -1 and 1)
   - Content: Corner coordinates (first 2 columns) + semantic information (remaining 14 columns)
   - Padded to 53 corners maximum

2. **`padding_mask`** (required)
   - Shape: `(53, 1)`
   - Type: `uint8`
   - Content: 1 = valid corner, 0 = padding
   - Indicates which of the 53 positions contain actual corners

3. **`global_matrix_np_padding`** (required)
   - Shape: `(53, 53)`
   - Type: `bool` or `uint8`
   - Content: Global attention matrix for graph connections
   - 1 = corners can attend to each other, 0 = no connection

4. **`edges`** (required)
   - Shape: Varies, typically `(N*N, 1)` where N is number of corners
   - Type: `int` or `float`
   - Content: Edge connectivity information between corners

5. **`file_id`** (optional but recommended)
   - Type: `int`
   - Content: Unique identifier for the file

6. **`semantics`** (optional)
   - Type: `dict` or `ndarray`
   - Content: Semantic labels for rooms/regions

## How to Fix Your Conversion Script

If you have a script like `json_to_npy_floodfill.py` that's saving raw arrays, you need to modify it to save dictionaries instead.

### ❌ WRONG (Current approach):
```python
# This saves a raw array - WRONG!
import numpy as np

# Your processing code here...
image_array = process_floorplan()  # Returns a (2575, 1393) array

# DON'T DO THIS:
np.save('output.npy', image_array)
```

### ✅ CORRECT (Required approach):
```python
# This saves a dictionary - CORRECT!
import numpy as np

# Your processing code here...
corners = extract_corners()  # Extract corner points
edges = extract_edges()      # Extract edges
semantics = extract_semantics()  # Extract room semantics

# Normalize and pad corners to shape (53, 16)
corners_padded = pad_corners(corners, max_corners=53)

# Create padding mask (1 for valid corners, 0 for padding)
padding_mask = create_padding_mask(corners, max_corners=53)

# Create global attention matrix (53x53)
global_matrix = create_attention_matrix(corners, max_corners=53)

# Create the dictionary
data_dict = {
    'file_id': file_id,
    'corner_list_np_normalized_padding_withsemantics': corners_padded,
    'padding_mask': padding_mask,
    'global_matrix_np_padding': global_matrix,
    'edges': edges,
    'semantics': semantics
}

# Save the dictionary (not just an array!)
np.save('output.npy', data_dict)
```

## Example: Complete Conversion Template

Here's a template for converting your data:

```python
import numpy as np
import json

def convert_your_data_to_gsdiff_format(input_file, output_file):
    """
    Convert your custom data format to GSDiff format.

    Args:
        input_file: Your input data file (JSON, image, etc.)
        output_file: Output .npy file path
    """

    # 1. Load your data
    # (Replace this with your actual data loading code)
    with open(input_file, 'r') as f:
        your_data = json.load(f)

    # 2. Extract/process corner points
    # This is where YOU need to extract corners from your data
    corners = extract_corners_from_your_data(your_data)  # YOUR CODE HERE
    num_corners = len(corners)

    # 3. Pad corners to 53 and add semantics
    # Corners should be normalized to [-1, 1] range
    corners_padded = np.zeros((53, 16), dtype=np.float32)
    if num_corners > 0:
        # First 2 columns: x, y coordinates (normalized to [-1, 1])
        corners_padded[:num_corners, 0:2] = normalize_coordinates(corners[:, 0:2])
        # Remaining 14 columns: semantic information
        corners_padded[:num_corners, 2:16] = extract_semantic_features(corners)

    # 4. Create padding mask
    padding_mask = np.zeros((53, 1), dtype=np.uint8)
    padding_mask[:num_corners, 0] = 1  # 1 = valid, 0 = padding

    # 5. Create global attention matrix
    global_matrix = np.zeros((53, 53), dtype=np.uint8)
    # All valid corners can attend to each other
    global_matrix[:num_corners, :num_corners] = 1

    # 6. Extract edges
    edges = extract_edges_from_your_data(your_data)  # YOUR CODE HERE

    # 7. Create the dictionary
    data_dict = {
        'file_id': get_file_id(input_file),
        'corner_list_np_normalized_padding_withsemantics': corners_padded,
        'padding_mask': padding_mask,
        'global_matrix_np_padding': global_matrix,
        'edges': edges,
        'semantics': extract_semantics(your_data)
    }

    # 8. Save as numpy file
    np.save(output_file, data_dict)
    print(f"Saved: {output_file}")


def normalize_coordinates(coords, image_size=512):
    """
    Normalize coordinates from [0, image_size] to [-1, 1]
    """
    return (coords / (image_size / 2)) - 1.0


def extract_corners_from_your_data(data):
    """
    Extract corner points from your data format.

    Returns:
        ndarray of shape (N, 2) where N is number of corners
    """
    # TODO: Implement based on your data format
    # Example: If you have a floodfill mask, detect corners
    raise NotImplementedError("Implement corner extraction for your data format")


def extract_edges_from_your_data(data):
    """
    Extract edge connectivity from your data.

    Returns:
        ndarray representing edge connections
    """
    # TODO: Implement based on your data format
    raise NotImplementedError("Implement edge extraction for your data format")


def extract_semantic_features(corners):
    """
    Extract semantic features for each corner.

    Returns:
        ndarray of shape (N, 14) with semantic information
    """
    # TODO: Implement semantic feature extraction
    # These 14 columns typically represent room types, wall types, etc.
    return np.zeros((len(corners), 14), dtype=np.float32)


def extract_semantics(data):
    """
    Extract overall semantic information.
    """
    # TODO: Implement based on your requirements
    return {}


def get_file_id(filepath):
    """Extract numeric file ID from filepath."""
    import os
    return int(os.path.splitext(os.path.basename(filepath))[0])


# Usage example:
if __name__ == '__main__':
    import glob

    input_files = glob.glob('your_data/*.json')  # Or whatever your input format is

    for input_file in input_files:
        output_file = input_file.replace('.json', '.npy')
        output_file = output_file.replace('your_data', 'datasets/rplang-v3-withsemantics/test')

        convert_your_data_to_gsdiff_format(input_file, output_file)
```

## Validation Script

Use this to validate your converted files:

```python
import numpy as np
from datasets.path_utils import get_data_path

def validate_npy_file(filepath):
    """Validate that a .npy file has the correct format."""

    try:
        # Load the file
        data = np.load(filepath, allow_pickle=True)

        # Check if it's a 0-dimensional array (expected)
        if data.ndim == 0:
            graph = data.item()
        elif data.size == 1:
            graph = data.flatten()[0]
        else:
            print(f"❌ FAIL: {filepath}")
            print(f"   File contains a raw array of shape {data.shape}")
            print(f"   Expected: dictionary saved as 0-d array")
            return False

        # Check if it's a dictionary
        if not isinstance(graph, dict):
            print(f"❌ FAIL: {filepath}")
            print(f"   File contains {type(graph)}, not a dictionary")
            return False

        # Check required keys
        required_keys = [
            'corner_list_np_normalized_padding_withsemantics',
            'padding_mask',
            'global_matrix_np_padding',
            'edges'
        ]

        missing_keys = [k for k in required_keys if k not in graph]
        if missing_keys:
            print(f"❌ FAIL: {filepath}")
            print(f"   Missing required keys: {missing_keys}")
            return False

        # Check shapes
        corners = graph['corner_list_np_normalized_padding_withsemantics']
        if corners.shape != (53, 16):
            print(f"❌ FAIL: {filepath}")
            print(f"   corners shape is {corners.shape}, expected (53, 16)")
            return False

        padding_mask = graph['padding_mask']
        if padding_mask.shape != (53, 1):
            print(f"❌ FAIL: {filepath}")
            print(f"   padding_mask shape is {padding_mask.shape}, expected (53, 1)")
            return False

        global_matrix = graph['global_matrix_np_padding']
        if global_matrix.shape != (53, 53):
            print(f"❌ FAIL: {filepath}")
            print(f"   global_matrix shape is {global_matrix.shape}, expected (53, 53)")
            return False

        print(f"✅ PASS: {filepath}")
        return True

    except Exception as e:
        print(f"❌ ERROR: {filepath}")
        print(f"   {str(e)}")
        return False


# Validate all test files
import os
test_dir = 'datasets/rplang-v3-withsemantics/test'
files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]

print(f"Validating {len(files)} files...")
passed = sum(1 for f in files if validate_npy_file(os.path.join(test_dir, f)))
print(f"\n{passed}/{len(files)} files passed validation")
```

## Reference: Official Data Processing

The official data processing pipeline is in `datasets/rplan-process4.py`. Here's the relevant excerpt (lines 1072-1079):

```python
''' write'''
new_graph = copy.deepcopy(graph)
new_graph['semantics'] = normalized_seman_d
new_graph['corner_list_np_normalized_padding_withsemantics'] = result

if file_id in train_fnids:
    np.save('./rplandata/Data/rplang-v3-withsemantics/train/' + str(file_id) + '.npy', new_graph)
elif file_id in val_fnids:
    np.save('./rplandata/Data/rplang-v3-withsemantics/val/' + str(file_id) + '.npy', new_graph)
elif file_id in test_fnids:
    np.save('./rplandata/Data/rplang-v3-withsemantics/test/' + str(file_id) + '.npy', new_graph)
```

Note that `new_graph` is a **dictionary**, not a raw array.

## Quick Fix Steps

1. **Find your conversion script** (`json_to_npy_floodfill.py` or similar)

2. **Locate where you call `np.save()`**

3. **Check what you're saving**: If you're saving a raw array/image, that's the problem

4. **Create a dictionary** with all required keys instead

5. **Save the dictionary** with `np.save(filepath, your_dict)`

6. **Validate** using the validation script above

7. **Re-run your conversion** for all your data files

8. **Test** by running `test_boun.py` again

## Need Help?

If you're stuck:

1. Check if your conversion script exists in a different branch
2. Look at `datasets/rplan-process4.py` for a complete example
3. Use the template provided above
4. Run the validation script to check your output

## Summary

**The key point:** You must save a **dictionary** containing all required fields, not a raw numpy array. The dictionary gets automatically converted to a 0-dimensional numpy array when saved with `np.save()`, which is the correct format.
