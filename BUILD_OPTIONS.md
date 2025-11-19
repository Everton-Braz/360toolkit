# 360FrameTools - Build Options

This project offers **two build configurations** optimized for different use cases:

## 🖥️ CPU Build (Default - Smaller)

**Best for**: Most users, smaller deployments, occasional use

### Characteristics
- **Binary size**: ~780 MB
- **Masking speed**: ~10 minutes for 1000 images
- **Requirements**: Any system (no GPU needed)
- **Installation**: Simple (no CUDA dependencies)

### How to Build
```bash
# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Build
pyinstaller 360FrameTools_MINIMAL.spec
```

Or use the build script:
```bash
build_cpu_version.bat
```

## 🚀 GPU Build (Faster Masking)

**Best for**: Power users, large batches, time-critical workflows

### Characteristics
- **Binary size**: ~2.3 GB
- **Masking speed**: ~5 minutes for 1000 images (2× faster)
- **Requirements**: NVIDIA GPU + CUDA Toolkit
- **Installation**: Requires CUDA setup

### How to Build
```bash
# Install GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements-gpu.txt

# Build
pyinstaller 360FrameTools_GPU.spec
```

Or use the build script:
```bash
build_gpu_version.bat
```

### GPU Prerequisites
1. **NVIDIA GPU** with CUDA support (GTX 1060 or better)
2. **CUDA Toolkit** 11.8 or 12.1
   - Download: https://developer.nvidia.com/cuda-downloads
3. **GPU drivers** (latest recommended)
   - Download: https://www.nvidia.com/drivers

See [GPU_BUILD_GUIDE.md](GPU_BUILD_GUIDE.md) for detailed instructions.

## Comparison Table

| Feature | CPU Build | GPU Build |
|---------|-----------|-----------|
| **Binary Size** | ~780 MB | ~2.3 GB |
| **Stage 1 (Extraction)** | Same speed | Same speed |
| **Stage 2 (Transforms)** | Same speed | Same speed |
| **Stage 3 (Masking)** | Slower | **2× faster** |
| **100 images** | 1 minute | 30 seconds |
| **1000 images** | 10 minutes | 5 minutes |
| **Requirements** | Any PC | NVIDIA GPU + CUDA |
| **Setup Complexity** | Simple | Moderate |
| **Distribution** | Easy | Requires GPU |

## Which Build Should I Use?

### Choose CPU Build if:
✅ You don't have an NVIDIA GPU
✅ Processing small batches (<100 images)
✅ Binary size is a concern
✅ Simpler deployment preferred
✅ Occasional use (preprocessing workflow)

### Choose GPU Build if:
✅ You have NVIDIA GPU (GTX 1060+)
✅ Processing large batches (1000+ images)
✅ Time is critical
✅ Larger images (4K+)
✅ Regular/daily use

## Files Reference

### CPU Build
- `requirements.txt` - CPU dependencies
- `360FrameTools_MINIMAL.spec` - CPU build configuration
- `build_cpu_version.bat` - CPU build script

### GPU Build
- `requirements-gpu.txt` - GPU dependencies
- `360FrameTools_GPU.spec` - GPU build configuration
- `build_gpu_version.bat` - GPU build script
- `GPU_BUILD_GUIDE.md` - Detailed GPU setup guide

## Performance Notes

### Stage 1 (Frame Extraction)
- Uses FFmpeg (subprocess) or SDK
- **Not GPU-accelerated**
- Same speed on both builds

### Stage 2 (Perspective Splitting)
- Uses OpenCV cv2.remap()
- CPU-bound operation
- **Not GPU-accelerated**
- Same speed on both builds

### Stage 3 (Masking with YOLOv8)
- Uses PyTorch for inference
- **GPU-accelerated** (if available)
- CPU: ~0.6s per image
- GPU: ~0.15s per image (4× faster)

**Key Insight**: GPU only accelerates Stage 3 (masking). Stages 1 and 2 are identical.

## Common Issues

### CPU Build
**Issue**: "Module not found: torch"
**Solution**: Install CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### GPU Build
**Issue**: "CUDA not available"
**Solution**: 
1. Install CUDA Toolkit
2. Verify with `nvidia-smi`
3. Reinstall PyTorch with correct CUDA version

**Issue**: "DLL load failed"
**Solution**: Add CUDA bin to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`

## Testing Your Build

After building, test all stages:

```bash
# Navigate to build
cd dist/360FrameTools

# Run the application
360FrameTools.exe

# Test Stage 1: Extract frames from video
# Test Stage 2: Split to perspectives
# Test Stage 3: Generate masks

# For GPU build, verify GPU is used:
nvidia-smi  # Should show GPU usage during masking
```

## Distribution

### CPU Build Distribution
- Simpler: Just zip `dist/360FrameTools/`
- Users need: Nothing (self-contained)
- Size: ~780 MB compressed

### GPU Build Distribution
- Larger: Zip `dist/360FrameTools/`
- Users need: NVIDIA GPU + drivers (CUDA not needed, bundled)
- Size: ~2.3 GB compressed
- Warning: Mention GPU requirement clearly

## Support

For build issues:
- CPU build: Check [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)
- GPU build: Check [GPU_BUILD_GUIDE.md](GPU_BUILD_GUIDE.md)
- General: Check [README.md](README.md)

---

**Quick Decision**: If you have a GPU and process many images, use GPU build. Otherwise, use CPU build.
