# GSDiff Quick Reference Summaries

**Date:** 2025-11-10  
**Purpose:** Quick reference summaries of REPORT.md and DATA_GENERATION_PIPELINE.md

---

## Table of Contents

1. [REPORT.md Summary - GSDiff Input Pipeline](#reportmd-summary)
2. [DATA_GENERATION_PIPELINE.md Summary - Preprocessing Pipeline](#data_generation_pipelinemd-summary)
3. [Quick Comparison Table](#quick-comparison)
4. [Integration Guidance](#integration-guidance)

---

## REPORT.md Summary

**Full Title:** GSDiff Floor Plan Generation Input Pipeline Analysis Report  
**Purpose:** How GSDiff loads and processes data for model inference

### Key Points

#### Input Format

- **Pre-processed `.npy` graph files** (NOT raw images during generation)
- Graph structure: 53 corners (padded), each with (x,y) coords + 14→9 semantic labels
- Files loaded from: `datasets/rplang-v3-withsemantics/{train,val,test}/*.npy`

#### Three Generation Modes

1. **Unconstrained:** Pure generative mode without constraints
2. **Topology-constrained:** Generation with room connectivity constraints
3. **Boundary-constrained:** Generation guided by wall/boundary images

#### Data Components

Each `.npy` file contains:

| Component | Shape | Description |
|-----------|-------|-------------|
| `corner_list_np_normalized_padding_withsemantics` | (53, 16) | Coordinates (2) + semantics (14) |
| `global_matrix_np_padding` | (53, 53) | Attention mask for transformer |
| `padding_mask` | (53, 1) | Binary: 1=real corner, 0=padding |
| `edges` | (2809, 1) | Flattened adjacency matrix (53×53) |

#### CNN Features

- **Pre-computed** boundary image embeddings
- Shape: (1024, 16, 16)
- Only used for boundary-constrained mode
- Stored in: `rplang-v3-withsemantics-prerunCNN-16/`

#### Coordinate System

- **Original:** [0, 255] pixel space
- **Normalized:** [-1, 1] range
- **Formula:** `(pixel_coord - 128) / 128`

#### Semantic Simplification

- **Original:** 16 room type categories
- **Simplified:** 9 categories (merges similar types)
- **Purpose:** Better learning efficiency, less sparse labels

**Merging Strategy:**

- Living + Dining → Living
- ChildRoom + StudyRoom + GuestRoom → SecondRoom
- Storage + Entrance → Entrance
- Wall-in removed

#### Diffusion Process

- **Timesteps:** 1000 steps (t=999 → 0)
- **Schedule:** Cosine beta schedule
- **Process:** Iterative denoising from pure noise

#### Model Architecture

- **Type:** Transformer-based diffusion model
- **Layers:** 24 transformer layers
- **Embedding Dimension:** 512
- **Attention Heads:** 4
- **Input Encodings:**
  - Corner positional encoding (2D → 512D)
  - Semantic embedding (8D → 512D)
  - Time embedding (timestep → 512D)
  - CNN feature projection (1024 → 256 channels)

#### Data Flow During Inference

```
Load .npy file
    ↓
Extract components (corners, semantics, masks, edges)
    ↓
Semantic simplification (16 → 9 dims)
    ↓
Load pre-computed CNN features
    ↓
Convert to PyTorch tensors
    ↓
Transfer to GPU
    ↓
Append padding mask to corners
    ↓
Encode inputs:
  - Corner positional encoding
  - Semantic embedding
  - Time embedding
  - CNN feature projection
    ↓
Combine embeddings (sum)
    ↓
Transformer processing (24 layers)
    ↓
Diffusion loop (1000 → 0)
    ↓
Output: Generated floor plan
```

### Bottom Line

Model takes **structured graph data**, not images. Images only used for boundary-constrained mode via pre-computed CNN features.

---

## DATA_GENERATION_PIPELINE.md Summary

**Full Title:** GSDiff Data Generation Pipeline  
**Purpose:** How raw RPLAN PNGs are converted to `.npy` graph files

### Key Points

#### Input Dataset

- **Source:** RPLAN dataset from USTC
- **Count:** 80,788 residential floor plan images
- **Format:** PNG, 256×256 pixels, RGBA (4 channels)
- **Location:** `datasets/rplandata/Data/floorplan_dataset/`

#### Channel 1 Encoding (THE KEY!)

**Channel 1 (Green channel) contains ALL semantic information:**

| Pixel Value | Meaning |
|-------------|---------|
| 0 | Living Room / Dining / Entrance |
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
| 12 | Walk-in Closet |
| 13 | External Area |
| **≥14** | **Walls / Boundaries** |

#### 14-Stage Pipeline

**Stage 1: Binary Wall Extraction**

- Pixels ≥14 → 255 (white walls)
- Pixels ≤13 → 0 (black rooms)
- Output: 256×256 binary image

**Stage 2: Morphological Processing**

- Erosion (thin walls)
- Skeletonization (centerlines)
- Dilation (clean up)
- Smoothing (remove noise)
- Output: Single-pixel-width wall structure

**Stage 3: Quality Filtering (Three-Stage)**

- **Filter 1:** Remove 2×2 white blocks (artifacts)
  - 80,788 → 75,350 files
- **Filter 2:** Remove dead ends (topological errors)
  - 75,350 → 71,814 files
- **Filter 3:** Validate connectivity with original
  - 71,814 → 71,763 files (final valid set)

**Stage 4: Corner Detection**

- Analyze pixel neighborhoods
- **L-corner:** 2 perpendicular neighbors (sum = 254)
- **T-corner:** 3 neighbors (sum = 253)
- **X-corner:** 4 neighbors (sum = 252)
- Output: List of (x, y) corner coordinates

**Stage 5: Edge Extraction**

- Connect corners with wall segments
- Requirements:
  - Orthogonal (same row OR column)
  - Continuous white pixels
  - No intermediate corners
- Output: List of (corner1, corner2) edge pairs

**Stage 6: Graph Construction**

- Build adjacency matrix from edges
- Map corners to indices
- Create undirected graph

**Stage 7: Room Detection**

- Use NetworkX cycle basis algorithm
- Each cycle = one room polygon
- Output: List of room polygons

**Stage 8: Semantic Label Assignment**

- Query original PNG Channel 1 pixels inside each room
- Majority vote determines room type
- Assign semantic labels to corners based on adjacent rooms
- Output: 14-dimensional semantic vector per corner

**Stage 9: Coordinate Normalization**

- Formula: `(pixel_coord - 128) / 128`
- Range: [0, 255] → [-1, 1]
- Example: (0,0)→(-1,-1), (128,128)→(0,0), (255,255)→(0.99,0.99)

**Stage 10: Padding to Fixed Size**

- Pad all graphs to 53 corners (max in dataset)
- Real corners: first n positions
- Padding corners: zeros in remaining positions
- Create padding mask: 1=real, 0=padding

**Stage 11: Attention Matrix Creation**

- Global attention: all real corners attend to each other
- Shape: (53, 53)
- Top-left (n×n) block = 1, rest = 0

**Stage 12: Edge Coordinate Generation**

- Generate all pairwise edge coordinates (53×53 = 2809)
- edge_coords[i*53+j] = [x_i, y_i, x_j, y_j]
- edges[i*53+j] = 1 if edge exists, 0 otherwise

**Stage 13: Save .npy Files**

- Save complete dictionary with all components
- Location: `rplang-v3-withsemantics/{train,val,test}/*.npy`

**Stage 14: CNN Feature Pre-computation**

- Render boundary image from corners + edges
- Pass through ResNet-like CNN encoder
- Extract features at 16×16 resolution
- Save as `*_feat16.npy`: (1, 1024, 16, 16)
- Location: `rplang-v3-withsemantics-prerunCNN-16/`

#### Key Script: rplan-extract.py

**File:** `datasets/rplan-extract.py` (407 lines)

**Outputs:** `structure_graphs.npy` (intermediate format)

**Structure Graph Format:**

```python
structure_graph = {
    (x, y): [up_neighbor, left_neighbor, down_neighbor, right_neighbor]
    # Each neighbor is either (x2, y2) or (-1, -1) if no neighbor
}
```

**Next Steps:**

- `rplan-process1.py` through `rplan-process10.py` refine this data
- Final output: 71,763 complete `.npy` files

#### Final Output

**File Count:**

- Training: 65,763 files
- Validation: 3,000 files
- Testing: 3,000 files
- **Total:** 71,763 valid files

**Each .npy File Contains:**

```python
{
    'corner_list_np_normalized_padding': (53, 2),
    'corner_list_np_normalized_padding_withsemantics': (53, 16),
    'adjacency_matrix_np_padding': (53, 53),
    'global_matrix_np_padding': (53, 53),
    'padding_mask': (53, 1),
    'edge_coords': (2809, 4),
    'edges': (2809, 1),
    # ... additional metadata
}
```

### Bottom Line

PNG images are **preprocessed offline once**. The semantic channel encodes everything. Pipeline extracts graph structure automatically from pixel data.

---

## Quick Comparison

| Aspect | REPORT.md | DATA_GENERATION_PIPELINE.md |
|--------|-----------|----------------------------|
| **Focus** | Model input & inference | Data preprocessing |
| **Timeframe** | During generation | Before training (offline) |
| **PNG Role** | Only for CNN features | Source of all data |
| **Key Output** | Model predictions | `.npy` graph files |
| **Main Process** | Load → Encode → Diffuse → Generate | Extract → Process → Structure → Save |
| **Runs When** | Every generation | Once per dataset |
| **Your Need** | Understand model expectations | Understand input data format |

---

## Integration Guidance

### For Graph2Plan Integration

#### Use DATA_GENERATION_PIPELINE.md to understand

1. **What format you need to create:**
   - 53-corner padded graph structure
   - Normalized coordinates [-1, 1]
   - Semantic labels (simplified to 9 categories)
   - Adjacency matrices
   - Padding masks

2. **How to create that format:**
   - Convert your room adjacency graph → corner coordinates
   - Generate edges from room boundaries
   - Normalize and pad to 53 corners
   - Create attention masks

3. **What preprocessing looks like:**
   - Even if you're not processing RPLAN images
   - You need equivalent data structures
   - Reference for validation and debugging

#### Use REPORT.md to understand

1. **What GSDiff expects during inference:**
   - Exact tensor shapes and formats
   - How data flows through the model
   - Where each component is used

2. **How to format your data for the model:**
   - Coordinate normalization
   - Semantic encoding scheme
   - Padding strategy
   - Attention mask structure

3. **What you can control:**
   - Boundary constraints (via CNN features)
   - Room connectivity (via topology)
   - Generation parameters (timesteps, etc.)

### Critical Integration Steps

1. **Create Graph2Plan → GSDiff Converter:**

   ```
   Your room adjacency graph
       ↓
   Generate corner positions (layout algorithm)
       ↓
   Normalize coordinates
       ↓
   Pad to 53 corners
       ↓
   Create semantic labels
       ↓
   Build adjacency matrix
       ↓
   Save as .npy in GSDiff format
   ```

2. **Generate Boundary Images (for constrained mode):**

   ```
   Your boundary shape (rectangle/L-shape/polygon)
       ↓
   Render as 256×256 binary image
       ↓
   Encode with GSDiff CNN
       ↓
   Get (1024, 16, 16) features
       ↓
   Save as _feat16.npy
   ```

3. **Test with GSDiff:**

   ```
   Load your .npy file
       ↓
   Run through GSDiff inference
       ↓
   Verify output floor plan
       ↓
   Iterate on conversion if needed
   ```

---

## Key Takeaways

### From REPORT.md

- ✅ Model uses **structured graph data**, not raw images
- ✅ Images only needed for **boundary-constrained mode**
- ✅ Data must be **normalized and padded** to fixed size
- ✅ **Pre-computed features** optimize inference speed
- ✅ **Diffusion process** runs 1000 denoising steps

### From DATA_GENERATION_PIPELINE.md

- ✅ RPLAN PNGs encode **semantics in Channel 1**
- ✅ Preprocessing is **fully automatic** from images
- ✅ **Quality filtering** reduces 80K → 71K valid samples
- ✅ **Structure graphs** are intermediate representation
- ✅ Final `.npy` format is **standardized and fixed-size**

### For Your Project

- ✅ You **don't need** RPLAN preprocessing scripts
- ✅ You **do need** to create equivalent `.npy` files
- ✅ Focus on **graph structure conversion**, not image processing
- ✅ Understand **data format requirements** from both reports
- ✅ Test with **simple cases** first (rectangle boundary, few rooms)

---

**End of Quick Reference**

*Generated: 2025-11-10*  
*For: Graph2Plan → GSDiff Integration*
