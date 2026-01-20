# GPU Compatibility Guide for 360toolkit

## Overview

360toolkit supports GPU acceleration for AI masking (Stage 3) using PyTorch and CUDA. This guide helps you ensure your GPU is properly configured.

## Supported GPUs

### ✅ Fully Supported (with correct PyTorch version)
- **RTX 50-series** (Blackwell): 5090, 5080, 5070 Ti, 5070 - Requires PyTorch nightly
- **RTX 40-series** (Ada Lovelace): 4090, 4080, 4070 Ti, 4070, 4060 Ti, 4060
- **RTX 30-series** (Ampere): 3090 Ti, 3090, 3080 Ti, 3080, 3070 Ti, 3070, 3060 Ti, 3060
- **RTX 20-series** (Turing): 2080 Ti, 2080 Super, 2080, 2070 Super, 2070, 2060 Super, 2060
- **GTX 16-series**: 1660 Ti, 1660 Super, 1660, 1650 Super, 1650
- **GTX 10-series** (Pascal): 1080 Ti, 1080, 1070 Ti, 1070, 1060, 1050 Ti, 1050

### ⚠️ Requires Special Configuration
- **RTX 50-series GPUs**: Need PyTorch nightly build (stable PyTorch doesn't support sm_120 compute capability yet)

### ❌ Not Supported
- GPUs older than GTX 10-series (Pascal)
- AMD GPUs (PyTorch CUDA is NVIDIA-only)
- Intel GPUs

## Common Issues & Fixes

### Issue 1: "no kernel image is available for execution on the device"

**Symptoms:**
```
CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible
```

**Cause:** Your RTX 50-series GPU has compute capability `sm_120` (Blackwell), but stable PyTorch only supports up to `sm_90`.

**Fix for RTX 50-series:**
1. Run the provided batch file: `update_pytorch_for_rtx50.bat`
2. Or manually install:
   ```bash
   pip uninstall torch torchvision
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
3. Restart 360toolkit

**Fix for RTX 30/40-series:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Issue 2: "CUDA not available"

**Symptoms:**
```
CUDA available: False
Using CPU mode
```

**Possible causes:**
1. No NVIDIA GPU installed
2. NVIDIA drivers not installed/outdated
3. PyTorch CPU-only version installed

**Fix:**
1. Install/update NVIDIA drivers: https://www.nvidia.com/drivers
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

### Issue 3: Slow masking performance

**Symptoms:**
- Masking takes 10+ seconds per image
- Log shows "Using CPU mode"

**Fix:**
1. Check GPU compatibility: Run `python check_gpu_compatibility.py`
2. Update PyTorch if needed (see Issue 1)
3. If GPU still not working, CPU mode is automatic fallback (slower but functional)

## Checking Your Configuration

### Method 1: Run the compatibility checker
```bash
python check_gpu_compatibility.py
```

This will test your GPU and show:
- GPU model and compute capability
- PyTorch version and CUDA support
- Actual GPU functionality test
- Recommended fixes if issues detected

### Method 2: Check 360toolkit logs
When you run 360toolkit, look for these log messages:

**✅ Good (GPU working):**
```
✓ GPU compatibility verified - CUDA operations successful
Using device: cuda:0
```

**⚠️ Problem (falling back to CPU):**
```
✗ GPU architecture incompatibility detected
→ Falling back to CPU for masking operations
```

## Performance Impact

### With GPU (CUDA):
- **Small model (yolov8s)**: ~0.5 seconds per image
- **Medium model (yolov8m)**: ~1.0 seconds per image
- **Large model (yolov8l)**: ~1.5 seconds per image

### Without GPU (CPU fallback):
- **Small model**: ~3-5 seconds per image
- **Medium model**: ~5-8 seconds per image
- **Large model**: ~8-12 seconds per image

## Installation Recommendations

### For RTX 50-series (5090, 5080, 5070 Ti, 5070):
```bash
# Use PyTorch nightly (CUDA 12.8+)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install ultralytics opencv-python pillow piexif PyQt6
```

### For RTX 30/40-series, GTX 10/16-series:
```bash
# Use stable PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python pillow piexif PyQt6
```

### No GPU / CPU only:
```bash
# CPU-only PyTorch (smaller download)
pip install torch torchvision
pip install ultralytics opencv-python pillow piexif PyQt6
```

## Troubleshooting Steps

1. **Check GPU is detected:**
   ```bash
   nvidia-smi
   ```
   Should show your GPU with driver version

2. **Run compatibility checker:**
   ```bash
   python check_gpu_compatibility.py
   ```

3. **Check PyTorch CUDA:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Your GPU name
   ```

4. **Test GPU computation:**
   ```python
   import torch
   x = torch.zeros(1, device='cuda')
   y = x + 1
   print(y)  # Should work without errors
   ```

## Still Having Issues?

If GPU still doesn't work after trying all fixes:
1. The app automatically falls back to CPU - slower but functional
2. Check if your GPU is in the supported list above
3. Ensure NVIDIA drivers are up to date (download from nvidia.com)
4. Consider using CPU mode if GPU troubleshooting takes too long

## Update History

- **January 2026**: Added RTX 50-series (Blackwell) support via PyTorch nightly
- **Current**: Supports compute capabilities sm_50 through sm_120

---

**Note**: GPU acceleration only affects Stage 3 (AI Masking). Stages 1 and 2 use CPU regardless of GPU availability.
