# GSDiff Data Fix Summary

## Problem Solved: Empty Ground Truth PNGs

The ground truth PNGs were rendering as empty/white images because the test NPY files had **incorrectly normalized coordinates**.

## Root Cause

Your raster floor plan files are **2575×1393 pixels**, but the `raster_to_graph_converter.py` was using the default `image_size=256` parameter. This caused coordinates to be normalized incorrectly:

**Before (WRONG):**
```
Corner coordinates: [6.75, 12.49] - way outside [-1, 1] range!
```

**After (CORRECT):**
```
Corner coordinates: [-0.23, 0.34] - properly in [-1, 1] range ✓
```

## Solution Applied

Regenerated all test NPY files with correct normalization:

```bash
python raster_to_graph_converter.py \
    --input_dir temp_raster_files \
    --output_dir datasets/rplang-v3-withsemantics/test \
    --image_size 2575 \  # <-- CRITICAL: Must match your actual raster size
    --validate
```

## Current Status

### ✅ Working Files (3/5)

**File 0 (apartment_1.npy):**
- 50 corners, 7 rooms detected
- Ground truth PNG renders correctly
- Shows proper floor plan with colored rooms and walls

**File 1 (apartment_2.npy):**
- 53 corners, 6 rooms detected
- Ground truth PNG renders correctly
- Multiple rooms with proper geometry

**File 2 (apartment_3.npy):**
- 53 corners, 8 rooms detected
- Ground truth PNG renders correctly
- Complex layout with 8 distinct rooms

### ⚠️ Partial Issues (2/5)

**File 3 (apartment_4.npy):**
- 53 corners, but **0 cycles detected**
- Graph is disconnected (6 nodes in main component, 47 disconnected)
- Likely due to complex apartment layout with separate room groups

**File 4 (apartment_5.npy):**
- 53 corners, but **0 cycles detected**
- Graph is disconnected (7 nodes in main component, 46 disconnected)
- 2 corners have only 1 connection (unusual)

## Files Generated

### Test Data Files (regenerated)
- `datasets/rplang-v3-withsemantics/test/0-4.npy` - Graph NPY files with correct normalization
- `datasets/rplang-v3-withsemantics-withboundary/test/0-4.npy` - Withboundary files
- `datasets/prerunning_cnn_featuremaps/0-4.npy` - CNN feature maps (dummy/random for testing)

### Ground Truth PNGs (verified)
- `test_gt_outputs/test_gt_0.png` ✅ 2.2KB - Shows floor plan
- `test_gt_outputs/test_gt_1.png` ✅ 2.3KB - Shows floor plan
- `test_gt_outputs/test_gt_2.png` ✅ 2.2KB - Shows floor plan
- `test_gt_outputs/test_gt_3.png` ⚠️ 1.9KB - Empty (disconnected graph)
- `test_gt_outputs/test_gt_4.png` ⚠️ 1.9KB - Empty (disconnected graph)

### Diagnostic Tools Added
- `test_gt_rendering.py` - Standalone PNG generator for verification
- `inspect_test_data.py` - Inspect NPY file structure and coordinates
- `check_edge_connectivity.py` - Analyze graph connectivity issues

## Code Changes

### scripts/test_boun.py
Changed device from CUDA to CPU mode:
```python
device = 'cpu'  # Changed from 'cuda:0'
```

### datasets/rplang_edge_semantics_simplified.py
Already fixed with robust numpy array format handling (from previous commits).

### datasets/rplang_edge_semantics_simplified_81.py
Already fixed with robust feature map loading (from previous commits).

## What Works Now

1. ✅ Data loads without errors (no more `.item()` crashes)
2. ✅ Coordinates properly normalized to [-1, 1] range
3. ✅ Binary semantics (0/1 values) for room types
4. ✅ Ground truth PNGs render correctly for 3/5 test files
5. ✅ CNN feature maps generated (dummy data, but correct format)
6. ✅ All required dictionary keys present in NPY files

## Known Issues & Next Steps

### Files 3 & 4: Disconnected Graph Problem

**Issue:** The floor plans have multiple separate room components that aren't connected in a single graph.

**Why it happens:**
- Apartments 4 and 5 have more complex layouts
- The raster-to-graph conversion creates separate polygons for each room
- These rooms may not share corners (disconnected)

**Possible solutions:**
1. **Check original JSON/raster data:** Verify apartments 4 and 5 have properly connected walls
2. **Adjust converter parameters:**
   - Increase `max_corners_per_room` to capture more detail
   - Decrease `simplify_tolerance` to preserve more corners
3. **Manual review:** Inspect the original apartment_4.npy and apartment_5.npy raster files
4. **Alternative approach:** Use the working apartments (1-3) for initial testing

### Testing Without Model Weights

The script `scripts/test_boun.py` requires pretrained model weights:
```
FileNotFoundError: model_stage2_best_065000.pt
```

To test the data pipeline without models, use the standalone script:
```bash
python test_gt_rendering.py
```

## Verification Commands

### Check coordinate ranges:
```bash
python inspect_test_data.py
```

Expected output:
```
Corner 0: coords=[-0.23  0.34], semantics=[1. 0. 0. ...]  ✓ In [-1,1]
Corner 1: coords=[-0.77  0.34], semantics=[1. 0. 0. ...]  ✓ In [-1,1]
```

### Check graph connectivity:
```bash
python check_edge_connectivity.py
```

### Generate ground truth PNGs:
```bash
python test_gt_rendering.py
```

Check output in `test_gt_outputs/` directory.

## Important Notes

1. **Image Size Parameter:** Always use `--image_size 2575` when converting your data
2. **CNN Feature Maps:** Current feature maps are dummy/random data - they allow code to run but won't produce meaningful model results
3. **For Production:** You need actual CNN-extracted features from a pretrained boundary encoder
4. **Files 3 & 4:** May need manual investigation of the original floor plan data

## Summary

**Main Achievement:** Fixed the empty PNG issue! ✅

Your GSDiff data pipeline now works correctly for 60% of test files (3/5). The remaining 2 files have graph connectivity issues that are separate from the original normalization problem.

For immediate testing, you can use files 0, 1, and 2, which render beautiful floor plans with:
- Proper room boundaries
- Correct semantics (room types)
- Multiple rooms per floor plan
- Walls and corners properly rendered
