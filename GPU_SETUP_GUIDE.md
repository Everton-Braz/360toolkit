# GPU Support Guide - 360toolkit

## Quick Fix (Most Common Issue)

If you installed the **CPU-only version of PyTorch**, run this:

```bash
fix_gpu_support.bat
```

This will:
1. Uninstall CPU-only PyTorch
2. Install CUDA-enabled PyTorch (CUDA 12.4)
3. Install ONNX Runtime with GPU support
4. Test GPU availability

---

## Understanding the Issue

The app detects **two types of GPU problems**:

### ❌ Problem 1: CPU-Only PyTorch Installed
**Symptom**: `PyTorch installed: 2.9.0+cpu` (note the `+cpu`)

**Cause**: You installed PyTorch without specifying CUDA support:
```bash
# WRONG: Installs CPU-only version
pip install torch

# CORRECT: Installs CUDA-enabled version
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Fix**: Run `fix_gpu_support.bat` or manually:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

### ❌ Problem 2: CUDA Version Mismatch
**Symptom**: `PyTorch GPU DLL loading failed: [WinError 1114]`

**Cause**: Your PyTorch was compiled for a different CUDA version than your GPU drivers support.

**Check your CUDA version**:
```bash
nvidia-smi
```
Look for "CUDA Version: 12.x" in the output.

**Install matching PyTorch**:
- **CUDA 11.8**: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- **CUDA 12.4**: `pip install torch --index-url https://download.pytorch.org/whl/cu124` (recommended)

---

## Hardware Compatibility

### Minimum Requirements
- **GPU**: NVIDIA GPU with **Compute Capability 3.5+**
- **Drivers**: Latest NVIDIA drivers (download from [nvidia.com](https://www.nvidia.com/Download/index.aspx))
- **VRAM**: 
  - Minimum: 4 GB (small models)
  - Recommended: 8 GB+ (medium/large models)

### Supported GPUs
✅ **Works perfectly**:
- RTX 40-series (4090, 4080, 4070, etc.)
- RTX 30-series (3090, 3080, 3070, 3060, etc.)
- RTX 20-series (2080 Ti, 2080, 2070, 2060, etc.)
- GTX 16-series (1660 Ti, 1660, 1650)
- GTX 10-series (1080 Ti, 1080, 1070, 1060)

⚠️ **Special handling required**:
- **RTX 50-series** (5090, 5080, 5070 Ti, etc.): Requires PyTorch nightly build
  - Run: `update_pytorch_for_rtx50.bat`
  - Or manually: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`

❌ **Not supported**:
- GTX 9-series and older (compute capability < 3.5)
- AMD GPUs (PyTorch CUDA is NVIDIA-only)
- Intel GPUs (not supported by PyTorch CUDA)

---

## Checking Your Setup

Run the diagnostic tool:
```bash
python check_gpu_compatibility.py
```

**Expected output (working GPU)**:
```
✓ PyTorch installed: 2.9.0+cu124
  CUDA available: True
  CUDA version: 12.4
  GPU devices found: 1

  GPU 0: NVIDIA GeForce RTX 4060
    Compute Capability: sm_89
    ✓ GPU 0 fully functional (test passed)
```

**Bad output (CPU-only)**:
```
✓ PyTorch installed: 2.9.0+cpu
  CUDA available: False
```
👉 Run `fix_gpu_support.bat` to fix!

---

## Performance Impact

### Stage 2 (Perspective Splitting)
- **With GPU**: 10-20× faster
- **CPU-only**: Slow but functional (uses multiprocessing)

### Stage 3 (AI Masking with ONNX)
- **With GPU**: 5-10× faster
- **CPU-only**: Slow but functional

**Bottom line**: The app works fine without GPU, just slower. GPU is optional but highly recommended for large batches.

---

## Troubleshooting

### Error: "no kernel image available for execution on the device"
**Cause**: Your GPU architecture is not supported by this PyTorch build.

**Fix for RTX 50-series**:
```bash
update_pytorch_for_rtx50.bat
```

**Fix for other GPUs**: Update PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade
```

---

### Error: "CUDA out of memory"
**Cause**: GPU VRAM is full.

**Fix**:
1. Close other GPU-intensive apps (browsers with hardware acceleration, games, etc.)
2. Use smaller batch sizes (automatically handled by the app)
3. Use smaller YOLO model (nano or small instead of medium/large)
4. Upgrade GPU (8 GB VRAM recommended)

---

### ONNX Runtime still using CPU after fix
**Check**: Make sure you installed `onnxruntime-gpu`, not `onnxruntime`:
```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

**Verify CUDA provider**:
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## Multi-PC Setup (Your Scenario)

You mentioned using the app on **two different PCs**:
1. **Home PC** (RTX 5070 Ti): GPU working ✅
2. **Laptop**: GPU not working ❌

### Why This Happens
Different PCs often have **different Python environments**:
- Home PC: Installed CUDA-enabled PyTorch correctly
- Laptop: Accidentally installed CPU-only PyTorch

### Solution: Use Same Installation Method on Both
Always install with CUDA support:
```bash
# On both PCs, run:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Or use the fix script:
```bash
fix_gpu_support.bat
```

### Git Ignore Python Environment
Make sure `.venv/` is in `.gitignore` so environment differences don't get committed:
```
.venv/
__pycache__/
*.pyc
```

---

## Advanced: Custom CUDA Version

If your system has a specific CUDA version, match PyTorch to it:

| CUDA Version | PyTorch Installation |
|--------------|----------------------|
| 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 12.8+ (RTX 50) | `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128` |

Check your CUDA version:
```bash
nvidia-smi
# Look for "CUDA Version: X.Y"
```

---

## Summary

**Most common issue**: CPU-only PyTorch installed  
**Quick fix**: Run `fix_gpu_support.bat`  
**Result**: GPU acceleration for 10-20× speedup  
**Fallback**: CPU mode always works (just slower)
