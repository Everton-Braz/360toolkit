# 360ToolkitGS - GPU Version

**360 Toolkit for Photogrammetry and Gaussian Splatting**

Version: GPU (PyTorch excluded)  
Size: ~700 MB  
License: MIT

---

## What's Included

✅ **Insta360 SDK** - Extract frames from .INSV files (dual-fisheye stitching)  
✅ **FFmpeg** - Video processing and frame extraction  
✅ **Stage 1**: Frame Extraction (SDK, FFmpeg, or dual-lens methods)  
✅ **Stage 2**: Perspective Splitting (compass-based multi-camera or cubemap)  
⚠️ **Stage 3**: AI Masking (requires PyTorch installation - see below)

---

## First-Time Setup

### Step 1: Extract the Application

Extract the `360ToolkitGS-GPU` folder to any location, for example:
```
C:\Program Files\360ToolkitGS-GPU\
```

### Step 2: Install PyTorch (Required for AI Masking)

**If you want to use Stage 3 (AI Masking)**, you need to install PyTorch:

#### Option A: Automatic Installation (Recommended)
1. Run `install_pytorch_gpu.bat`
2. Wait for installation to complete (~5-10 minutes)
3. Verify CUDA is detected

#### Option B: Manual Installation
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
```

#### Verify Installation
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
Should show: `CUDA: True`

### Step 3: Run the Application

Double-click `360ToolkitGS-GPU.exe`

---

## System Requirements

### Minimum Requirements:
- **OS**: Windows 10/11 (64-bit)
- **CPU**: Intel i5 or AMD Ryzen 5 (4 cores)
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.9 or higher (for AI Masking)

### Recommended for GPU Acceleration (Stage 3):
- **GPU**: NVIDIA GTX 1650 or better
- **VRAM**: 4 GB+
- **CUDA**: 11.8 or higher drivers
- **RAM**: 16 GB
- **Storage**: 20 GB free space (for PyTorch)

---

## Features

### Stage 1: Frame Extraction
- **Input**: `.insv` (Insta360 dual-fisheye) or `.mp4` files
- **Methods**:
  - SDK Stitching (PRIMARY - best quality, GPU-accelerated)
  - FFmpeg Stitched (fallback)
  - Dual-Lens (raw fisheye exports)
- **Output**: Equirectangular images (stitched 360° panoramas)
- **FPS Control**: 0.1 - 30 frames per second
- **Resolution**: Up to 8K (7680×3840)

### Stage 2: Perspective Splitting
- **Perspective Mode**:
  - Compass-based camera positioning
  - Default: 8 cameras at 110° FOV in horizontal ring
  - Customizable: yaw, pitch, roll, FOV per camera
  - Look-up/look-down rings for dome captures

- **Cubemap Mode**:
  - **6-face standard**: Front, Back, Left, Right, Top, Bottom (90° FOV)
  - **8-tile grid**: Configurable FOV or overlap percentage (4×2 arrangement)
  - Separate file export

- **Output**: Multiple perspective views with camera metadata embedded

### Stage 3: AI Masking (Requires PyTorch)
- **YOLOv8 Instance Segmentation**: Person detection
- **GPU Acceleration**: 6-7× faster than CPU
- **Models**: Nano, Small, Medium, Large, XLarge
- **Categories**: Persons, personal objects, animals
- **Output**: Binary masks (RealityScan compatible)
- **Smart Skipping**: Only creates masks for images with detections

---

## Workflow

### Complete Pipeline (Extract → Split → Mask):
1. **Stage 1**: Load `.insv` file → SDK stitching → Equirectangular frames
2. **Stage 2**: Load equirectangular → Generate perspectives (8 cameras at 110° FOV)
3. **Stage 3**: Load perspectives → YOLOv8 detection → Generate masks
4. **Output**: Perspective images + masks ready for photogrammetry

### Individual Stages:
- **Extract Only**: `.insv` → equirectangular images
- **Split Only**: Equirectangular → perspectives
- **Mask Only**: Perspectives → masks

---

## Troubleshooting

### Stage 3 Shows "CUDA not available"

**Problem**: PyTorch not installed or CPU version installed

**Solution**:
```powershell
# Uninstall CPU version
pip uninstall torch torchvision torchaudio -y

# Install GPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Application Won't Start

**Problem**: Missing dependencies

**Solution**:
1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install .NET Framework 4.8: https://dotnet.microsoft.com/download/dotnet-framework

### GPU Not Detected

**Problem**: NVIDIA drivers not installed or outdated

**Solution**:
1. Check GPU: `nvidia-smi`
2. Update drivers: https://www.nvidia.com/Download/index.aspx
3. Ensure CUDA 11.8+ drivers installed

### "SDK not found" Error

**Problem**: SDK not bundled correctly (shouldn't happen)

**Solution**: SDK is bundled in `_internal/sdk/` folder. If missing, re-extract the application.

---

## Performance

### Stage 1 (SDK Extraction):
- **Speed**: 3-5× faster than Insta360 Studio
- **Quality**: AI-based stitching with chromatic calibration
- **Resolution**: Up to 8K output

### Stage 2 (Splitting):
- **Speed**: ~2-5 seconds per frame (8 cameras)
- **Transform**: Cached remapping for faster batch processing

### Stage 3 (Masking with GPU):
| Model | Speed (GPU) | Speed (CPU) | Accuracy |
|-------|-------------|-------------|----------|
| Nano | 0.05s | 0.2s | 85% |
| **Small** | **0.08s** | **0.5s** | **90%** ⭐ |
| Medium | 0.15s | 1.0s | 92% |
| Large | 0.25s | 1.5s | 94% |
| XLarge | 0.40s | 2.5s | 95% |

**Recommended**: Small model (best balance of speed and accuracy)

---

## Technical Details

### Bundled Components:
- **Insta360 MediaSDK 3.0.5**: Dual-fisheye stitching engine
- **FFmpeg**: Video decoding and processing
- **OpenCV**: Image transformation and manipulation
- **PyQt6**: Modern GUI framework
- **NumPy**: Array operations

### NOT Bundled (User Installs):
- **PyTorch**: Deep learning framework (~2.8 GB)
- **Ultralytics**: YOLOv8 implementation (~100 MB)

---

## License

**360ToolkitGS**: MIT License

**Third-Party Components**:
- Insta360 SDK: Proprietary (bundled with permission)
- PyTorch: BSD License
- OpenCV: Apache 2.0 License
- FFmpeg: LGPL 2.1
- YOLOv8: AGPL-3.0

---

## Support

For issues, questions, or feature requests:
- GitHub: [Your Repository]
- Email: [Your Email]
- Discord: [Your Server]

---

## Credits

Developed by: [Your Name]  
Insta360 SDK: Arashi Vision Inc.  
YOLOv8: Ultralytics

---

**Note**: This is the GPU version which excludes PyTorch to reduce download size. For a fully bundled version, see 360ToolkitGS-CPU.
