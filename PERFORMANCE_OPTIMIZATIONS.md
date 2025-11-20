# GPU Training Performance Optimizations

## Problem
Training was running as slow as CPU despite being on a GPU machine.

## Root Causes Identified

### 1. **Data Loading Bottleneck** (CRITICAL)
- **Issue**: `num_workers=0` forced single-threaded data loading on main thread
- **Impact**: GPU was idle while waiting for data from disk
- **Fix**: Set `num_workers=4` with `persistent_workers=True` and `prefetch_factor=2`

### 2. **Blocking GPU Transfers**
- **Issue**: `.to(device)` calls blocked computation
- **Impact**: CPU-GPU transfer waits stalled the pipeline
- **Fix**: Added `non_blocking=True` to all `.to(device)` calls

### 3. **No Mixed Precision Training**
- **Issue**: Training in FP32 is ~2x slower than FP16
- **Impact**: Wasted GPU compute capability
- **Fix**: Enabled `torch.cuda.amp.autocast()` and `GradScaler`

### 4. **Inefficient Gradient Clearing**
- **Issue**: Using `param.grad.zero_()` allocates memory
- **Impact**: Unnecessary memory operations
- **Fix**: Use `optimizer.zero_grad(set_to_none=True)`

## Applied Optimizations

### Configuration Changes
```python
# Added performance settings
num_workers = 4  # Multi-threaded data loading (adjust based on CPU cores)
use_amp = True  # Mixed precision training for ~2x speedup
gradient_accumulation_steps = 1  # Can increase for larger effective batch size
```

### DataLoader Optimization
```python
DataLoader(
    ...,
    num_workers=num_workers,              # 4 parallel workers
    pin_memory=True,                      # Fast CPU-GPU transfer
    persistent_workers=True,              # Keep workers alive between epochs
    prefetch_factor=2                     # Prefetch 2 batches ahead
)
```

### Async GPU Transfers
```python
# Before: feat_16.to(device).float()
# After:
feat_16.to(device, non_blocking=True).float()
```

### Mixed Precision Training
```python
scaler = torch.cuda.amp.GradScaler()

# Forward pass
with torch.cuda.amp.autocast():
    output = model(inputs)
    loss = criterion(output, target)

# Backward pass
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
scaler.step(optimizer)
scaler.update()
```

## Expected Performance Improvements

### Conservative Estimates:
1. **Data Loading**: 2-4x faster (from single to multi-threaded)
2. **Mixed Precision**: 1.5-2x faster (FP16 vs FP32)
3. **Async Transfers**: 1.1-1.3x faster (overlapped transfer/compute)
4. **Total Expected**: **3-10x speedup** depending on bottleneck

### Realistic Scenario:
- **Before**: 100 steps/hour
- **After**: 300-500 steps/hour

## Tuning Recommendations

### 1. Adjust `num_workers`
```python
# Rule of thumb: num_workers = 4 * num_GPUs
# For 1 GPU: Try 4, 6, or 8
# Monitor: If CPU usage < 100%, increase num_workers
```

### 2. Monitor GPU Utilization
```bash
# On GPU machine, run:
nvidia-smi -l 1

# Target: GPU utilization should be 90-100%
# If lower, data loading is still a bottleneck
```

### 3. Gradient Accumulation (Optional)
```python
# If you want larger effective batch size without OOM:
gradient_accumulation_steps = 4  # Effective batch_size = 256 * 4 = 1024
```

### 4. Batch Size Tuning
```python
# Increase batch_size until GPU memory is ~90% full
# Larger batches = better GPU utilization
# Try: 256 → 512 → 1024 (monitor with nvidia-smi)
```

## Troubleshooting

### Issue: Still Slow After Changes
**Check:**
1. Verify `num_workers > 0` is actually set
2. Run `nvidia-smi` to check GPU utilization
3. Ensure data is not on slow network storage
4. Check if dataset is loading from disk vs RAM cache

### Issue: Out of Memory (OOM)
**Solutions:**
1. Reduce `batch_size` (256 → 128)
2. Reduce `num_workers` (frees system RAM)
3. Disable `persistent_workers=True`
4. Set `gradient_accumulation_steps > 1`

### Issue: "CUDA out of memory" during validation
**Solution:**
```python
# Reduce validation batch size
batch_size_val = 1000  # Instead of 3000
```

## Additional Performance Tips

### 1. Pin Memory Correctly
- Ensure `pin_memory=True` in DataLoader
- Only works with CPU tensors being transferred to GPU

### 2. Minimize CPU-GPU Sync Points
- Avoid `.item()` or `.cpu()` in training loop
- Accumulate metrics on GPU, transfer once per log interval

### 3. Profile Your Code
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # Training loop
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Verification

After applying changes, you should see:
- ✅ Training steps complete 3-5x faster
- ✅ GPU utilization at 90-100% (check with `nvidia-smi`)
- ✅ Data loading no longer bottleneck (workers busy)
- ✅ Memory usage stable (no leaks from workers)

---

**Date Applied:** 2025-11-20  
**Status:** Ready for GPU training
