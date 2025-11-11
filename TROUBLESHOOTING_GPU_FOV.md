# Troubleshooting Guide - GPU and FOV Issues

## Issue 1: GPU Not Being Used (SOLVED)

### Problem
```
WARNING - GPU requested but CUDA not available. Using CPU.
```

### Root Cause
PyTorch was installed with CPU-only version: `torch 2.9.0+cpu`

### Solution

#### Step 1: Uninstall CPU PyTorch
```powershell
pip uninstall torch torchvision torchaudio -y
```

#### Step 2: Install GPU PyTorch (CUDA 12.1)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Verify Installation
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce GTX 1650
```

#### Step 4: Test in Application
Run the application and check Stage 3 log:
- ✓ **Before**: `Using device: cpu`
- ✓ **After**: `Using device: cuda:0`

### Performance Improvement
- **Current (CPU)**: ~1.0 second per image (medium model)
- **With GPU**: ~0.15 second per image → **6-7× faster**

---

## Issue 2: FOV Not Changing Output (NEEDS VERIFICATION)

### What You Reported
> "I tested with different FOV and no difference was found"

### What the Logs Show
Looking at your actual pipeline runs, **FOV IS being used**:

```
Run 1: fov=110, overlap=144.44%  ← FOV 110° was used
Run 2: fov=50, overlap=11.11%    ← FOV 50° was used
```

The system is correctly:
1. Reading FOV from UI
2. Passing it to the transform
3. Logging the value

### Possible Causes for "No Visible Difference"

#### Cause 1: Output Files Not Being Regenerated
**Problem**: Pipeline may be skipping existing files.

**Check**:
```powershell
# Delete output folder completely
rm -r -Force "C:\Users\User\Documents\APLICATIVOS\Arquivos_Teste\ResidencialOne\TESTE\stage2_cubemap"

# Run pipeline again with different FOV
```

#### Cause 2: Viewing Wrong Files
**Problem**: Looking at old output files instead of new ones.

**Check**:
- Sort output folder by "Date Modified"
- Verify timestamps match your recent runs
- Open NEWEST files only

#### Cause 3: FOV Difference Too Subtle
**Problem**: For 8-tile cubemap, tiles are positioned at 0°, 90°, 180°, 270° yaw. If you're comparing tiles from different yaw angles, FOV differences may not be obvious.

**Check**:
- Compare the SAME tile (e.g., `tile_0_0.png`) from both runs
- Look at tile `0_0` (front-facing) where FOV difference is most visible

#### Cause 4: Caching (Less Likely)
**Problem**: E2P transform cache might be serving old results.

**Check**:
The cache key includes FOV, so this should NOT happen. But if suspected:
- Restart the application
- Clear cache manually if implemented

### How to Verify FOV is Working

#### Test 1: Run the Verification Script
```powershell
python test_fov_verification.py
```

This will:
- Generate 4 images with FOV 50°, 90°, 110°, 150°
- Save to `test_fov_verification/` folder
- Calculate image statistics to prove they're different
- Show visual comparison

#### Test 2: Manual Visual Test

1. **Clear output folder**:
   ```powershell
   rm -r -Force "OUTPUT_FOLDER\stage2_cubemap"
   ```

2. **Run pipeline with FOV 50°**:
   - Stage 2 → Cubemap → 8-tile
   - Set FOV: 50°
   - Run pipeline
   - **Rename** output folder to `output_fov_50`

3. **Run pipeline with FOV 150°**:
   - Stage 2 → Cubemap → 8-tile
   - Set FOV: 150°
   - Run pipeline
   - **Rename** output folder to `output_fov_150`

4. **Compare**:
   ```
   output_fov_50/frame_00001_tile_0_0.png   ← Zoomed in (telephoto)
   output_fov_150/frame_00001_tile_0_0.png  ← Wide angle (fisheye)
   ```
   
   Open both side-by-side. Difference should be VERY obvious:
   - FOV 50°: Much closer, less content visible
   - FOV 150°: Much wider, more distorted at edges

### Expected Visual Differences

```
FOV 50° (Telephoto):          FOV 90° (Normal):           FOV 150° (Wide):
┌─────────────┐              ┌─────────────┐             ┌─────────────┐
│   [zoom]    │              │  [normal]   │             │ [fisheye]   │
│             │              │             │             │             │
│    scene    │              │   scene     │             │   scene     │
│   detail    │              │   balanced  │             │  distorted  │
└─────────────┘              └─────────────┘             └─────────────┘
   Narrow view                 Medium view                 Wide view
   High detail                 Balanced                    Edge distortion
```

### What FOV Actually Does (Technical)

From `e2p_transform.py` line 97-98:

```python
focal_length_x = 1.0 / math.tan(h_fov_rad / 2)
focal_length_y = 1.0 / math.tan(v_fov_rad / 2)
```

**Math explanation**:
- FOV 50°: `focal_length = 1.0 / tan(25°) = 2.14` → **High magnification** (zoom in)
- FOV 90°: `focal_length = 1.0 / tan(45°) = 1.00` → **Normal** (1:1)
- FOV 150°: `focal_length = 1.0 / tan(75°) = 0.27` → **Low magnification** (zoom out)

The focal length directly affects how much of the equirectangular image is sampled for each output pixel.

### Debugging Steps

If FOV still appears not to work after verification:

1. **Enable debug logging**:
   ```python
   # In batch_orchestrator.py, add before transform call:
   logger.info(f"DEBUG: Calling E2P with h_fov={tile['fov']}")
   ```

2. **Check transform cache**:
   ```python
   # In e2p_transform.py, add logging:
   logger.info(f"Cache key: {cache_key}")
   logger.info(f"Cache hit: {cache_key in self.cache}")
   ```

3. **Verify output files are new**:
   ```powershell
   # Check file timestamps
   Get-ChildItem "OUTPUT_FOLDER\stage2_cubemap" | Sort-Object LastWriteTime
   ```

### Quick Diagnostic Command

```powershell
# Run this to see if transform is actually working:
python -c "from src.transforms.e2p_transform import E2PTransform; import cv2; import numpy as np; e2p = E2PTransform(); test_img = np.random.randint(0, 255, (1920, 3840, 3), dtype=np.uint8); img50 = e2p.equirect_to_pinhole(test_img, 0, 0, 0, 50, 512, 512); img150 = e2p.equirect_to_pinhole(test_img, 0, 0, 0, 150, 512, 512); print('Images identical:', np.array_equal(img50, img150)); print('Mean difference:', abs(img50.mean() - img150.mean()))"
```

Expected: `Images identical: False`

---

## Summary

### GPU Issue
✅ **Clear solution**: Reinstall PyTorch with CUDA support

### FOV Issue
❓ **Needs verification**: Logs show FOV is being used. Most likely:
- Files not being regenerated (clear output folder)
- Viewing wrong/old files (check timestamps)
- Comparing wrong tiles (use same tile number)

Run `test_fov_verification.py` to prove FOV is working at the transform level.

If test passes but pipeline still shows no difference:
→ Issue is in pipeline file management, not in FOV implementation
