# Fixing "No space left on device" Error with num_workers > 0

## Problem
When `num_workers > 0` in PyTorch DataLoader, you get:
```
OSError: [Errno 28] No space left on device
```

This happens because PyTorch's multiprocessing uses `/dev/shm` (shared memory), which is often too small in Docker containers or has limited space.

---

## Solution 1: Automatic Fallback (Already Implemented)

The training script now automatically detects this error and falls back to `num_workers=0`:

```python
# The script will try num_workers=4 first
# If it fails, it automatically uses num_workers=0
# You'll see a warning message explaining the fallback
```

**Pros**: No configuration needed, training starts automatically
**Cons**: Slower data loading (but better than crashing)

---

## Solution 2: Increase Docker Shared Memory (RECOMMENDED)

If you're running in Docker, increase the shared memory size:

### Option A: Docker Run Command
```bash
docker run --shm-size=8g your-image-name

# Recommended sizes:
# --shm-size=2g  (minimum for 2 workers)
# --shm-size=4g  (good for 4 workers)
# --shm-size=8g  (recommended for 4-8 workers)
```

### Option B: Docker Compose
```yaml
services:
  training:
    image: your-image
    shm_size: '8gb'  # or '8g'
```

### Option C: Kubernetes
```yaml
spec:
  containers:
  - name: training
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 8Gi
```

---

## Solution 3: Use tmpfs Mount Instead of /dev/shm

Mount a tmpfs volume to bypass `/dev/shm`:

```bash
docker run -v /tmp:/tmp your-image-name

# Then in your code, set this environment variable:
export TMPDIR=/tmp
```

---

## Solution 4: Manually Set num_workers=0

If you can't modify Docker settings, edit the script:

```python
# In trainval_main_boun.py, line ~42:
num_workers = 0  # Force single-process loading
```

**Performance Impact**: 
- Training will be ~20-30% slower
- But still much faster than CPU training
- GPU will have some idle time waiting for data

---

## Solution 5: Check and Clean /dev/shm

Sometimes `/dev/shm` has leftover files from crashed processes:

```bash
# Check current usage
df -h /dev/shm

# Clean up (be careful!)
rm -rf /dev/shm/*

# Check again
df -h /dev/shm
```

---

## How to Choose the Right Solution

### If you control Docker/container config:
✅ **Use Solution 2** (increase --shm-size to 4-8GB)

### If you're on a shared cluster/can't change Docker:
✅ **Use Solution 1** (automatic fallback - already implemented)
✅ Or manually set `num_workers=0` in the script

### If you're on bare metal Linux:
✅ Check `/dev/shm` size: `df -h /dev/shm`
✅ If it's small (<1GB), remount it larger:
```bash
sudo mount -o remount,size=8G /dev/shm
```

---

## Performance Comparison

| Configuration | Speed | Notes |
|--------------|-------|-------|
| `num_workers=0` | 1.0x (baseline) | Safe, always works |
| `num_workers=4` with enough shm | 2-3x faster | Recommended |
| `num_workers=8` with enough shm | 2.5-4x faster | Diminishing returns |

---

## Diagnostic Commands

### Check /dev/shm space:
```bash
df -h /dev/shm
```

### Check what's using /dev/shm:
```bash
ls -lh /dev/shm
du -sh /dev/shm/*
```

### Monitor during training:
```bash
watch -n 1 'df -h /dev/shm'
```

### Find PyTorch shared memory files:
```bash
ls -lh /dev/shm/torch_*
```

---

## What the Script Now Does Automatically

1. **Checks /dev/shm space** and warns if low
2. **Tries to create DataLoader** with num_workers=4
3. **Catches the error** if /dev/shm is full
4. **Automatically falls back** to num_workers=0
5. **Continues training** without crashing

You'll see output like:
```
/dev/shm space: 0.06 GB free / 0.06 GB total
WARNING: /dev/shm has low space. Consider using num_workers=0 or increasing --shm-size in Docker
Creating DataLoader with num_workers=4...
WARNING: Failed to create DataLoader with num_workers=4
Error: [Errno 28] No space left on device
Falling back to num_workers=0 (single-process data loading)
Note: Training will be slower. To fix, increase Docker --shm-size or use tmpfs mount
```

---

## Recommended Fix for Your Case

Since you're on a GPU machine (likely Docker), run your container with:

```bash
docker run --gpus all --shm-size=8g -it your-image

# Or if using docker-compose, add to your service:
shm_size: '8gb'
```

This will allow `num_workers=4` to work properly and give you the full performance benefit.

---

**Status**: Script updated with automatic fallback
**Date**: 2025-11-20
