# Floor Plan Corner Analysis Tool

A standalone tool for analyzing GSDiff floor plan datasets. This script helps you understand the corner distribution in your training data and visualize floor plan structures.

**Supports both .npy and .json file formats!**

## Features

- **Count corners**: Analyzes actual corners (excluding padding) in each floor plan
- **Statistics**: Provides min/max/mean/median corner counts across dataset
- **Distribution**: Shows corner count distribution to identify data patterns
- **Visualization**: Displays floor plans with corners and walls/edges
- **Multi-format**: Works with both .npy (GSDiff format) and .json files

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

### Basic Analysis

Analyze all files in a dataset directory:

```bash
python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train
```

### Limited Sample Analysis

Analyze only the first 100 files:

```bash
python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train --num-samples 100
```

### With Visualization

Analyze and visualize sample floor plans:

```bash
python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/val --num-samples 10 --visualize
```

### Single File Analysis

Analyze and visualize a specific floor plan:

```bash
# .npy file
python analyze_floor_plans.py ./datasets/rplang-v3-withsemantics/train/0.npy --visualize

# .json file
python analyze_floor_plans.py ./my_floor_plan.json --visualize
```

### JSON File Analysis

Analyze a directory of JSON floor plans:

```bash
python analyze_floor_plans.py ./my_json_floor_plans/ --num-samples 10 --visualize
```

## Output Example

```
Analyzing 1000 floor plan files from: ./datasets/rplang-v3-withsemantics/train
======================================================================

Corner Statistics:
  Total files analyzed: 1000
  Min corners: 12
  Max corners: 147
  Mean corners: 68.45
  Median corners: 65.00
  Std deviation: 23.12

Corner Count Distribution (top 10 most common):
  64 corners: 45 files (4.50%)
  72 corners: 42 files (4.20%)
  58 corners: 38 files (3.80%)
  ...

Example files by corner count:
  12 corners: 234.npy
  65 corners: 0.npy
  147 corners: 589.npy
```

## Visualization-Driven Counting

**Important:** When using `--visualize`, the tool uses a **visualization-first approach**:

1. **Visualizes each floor plan first** - Shows you exactly what will be counted
2. **Counts displayed corners** - Uses the corners actually shown in the visualization
3. **Compares counts** - Alerts you if raw count differs from displayed count

This ensures you see exactly what you're counting, eliminating confusion about padding, filtering, or data issues.

### What You'll See

When using `--visualize`, the tool displays interactive plots showing:
- **Red circles**: Corners without semantic information
- **Green circles**: Corners with semantic labels
- **Blue lines**: Walls/edges connecting corners
- **Numbers**: Corner indices for reference
- **Title shows**: Displayed corner count, semantics count, and edge count

If the raw data count differs from what's displayed, you'll see a warning:
```
Note: Raw count (150) differs from displayed count (127)
```

## Data Formats

### .npy Format (GSDiff)

This tool works with GSDiff `.npy` files containing:
- `corner_list_np_normalized_padding_withsemantics`: Corner coordinates and semantics
- `padding_mask`: Identifies real vs padded corners
- `edges`: Adjacency matrix for wall connections (optional)

### .json Format

The tool automatically detects and supports multiple JSON formats:

**Format 1: Direct graph structure**
```json
{
  "corners": [[x1, y1], [x2, y2], ...],
  "edges": [[i, j], [k, l], ...]
}
```
or use `"vertices"` or `"junctions"` instead of `"corners"`

**Format 2: RPLAN-style annotations**
```json
{
  "boxes": [{"x": 10, "y": 20, "width": 50, "height": 30}, ...],
  "lines": [[x1, y1, x2, y2], ...]
}
```
or use `"walls"` instead of `"lines"`

**Format 3: Simple corner list**
```json
[[x1, y1], [x2, y2], [x3, y3], ...]
```

The tool automatically extracts corners from whichever format you use!

## Use Cases

1. **Training diagnostics**: Check if your dataset has enough corners for the model capacity
2. **Data validation**: Identify outliers (too few/many corners)
3. **Model sizing**: Determine appropriate `max_corners` parameter for training
4. **Visual inspection**: Verify floor plan structure and edge connectivity

## Standalone Tool

This branch contains ONLY the analysis tool - it's completely independent from the main GSDiff repository. This makes it easy to:
- Use on different machines
- Share with others
- Run without installing GSDiff dependencies

## License

Same as GSDiff (see main repository)
