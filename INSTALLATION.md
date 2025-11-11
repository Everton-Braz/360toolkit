# 360FrameTools Installation Guide

## Quick Setup (Windows)

### 1. Prerequisites
- **Python 3.9+** - Download from [python.org](https://www.python.org/downloads/)
- **Git** (optional) - For cloning repository
- **8 GB RAM minimum** (16 GB recommended for GPU)
- **Windows 10/11** (for Insta360 SDK support)

### 2. Install Python Dependencies

Open PowerShell in the project directory:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install GPU Support (Optional but Recommended)

For CUDA GPU acceleration:

```powershell
# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Requirements**:
- NVIDIA GPU (GTX 1060 or better)
- CUDA 11.8+ installed
- 8 GB VRAM minimum

### 4. Verify Installation

```powershell
python test_setup.py
```

This will check all dependencies and report any issues.

### 5. Run Application

**Option A: Launcher Script**
```powershell
.\run.bat
```

**Option B: Python Command**
```powershell
python -m src.main
```

---

## Detailed Installation

### Installing FFmpeg (Optional)

FFmpeg enables faster frame extraction (Stage 1).

1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add to PATH: `C:\ffmpeg\bin`
4. Verify: `ffmpeg -version`

### Installing CUDA (For GPU Support)

1. Check GPU compatibility: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
2. Download CUDA Toolkit 11.8: [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
3. Install CUDA Toolkit
4. Verify: `nvcc --version`
5. Install PyTorch with CUDA (see step 3 above)

### Troubleshooting

#### PyQt6 Import Error
```powershell
pip install --upgrade PyQt6
```

#### CUDA Not Detected
```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Ultralytics/YOLOv8 Issues
```powershell
pip install --upgrade ultralytics
```

#### Memory Issues
- Reduce batch size in Stage 3 settings
- Lower cache size in application settings
- Use smaller YOLOv8 model (nano or small)

---

## Development Setup

For developers contributing to the project:

```powershell
# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Run integration tests
python tests/pipeline/test_full_workflow.py
```

---

## Configuration Files

### Camera Presets
Edit: `src/config/camera_presets.json`

Add custom camera configurations for different scenarios.

### Default Settings
Edit: `src/config/defaults.py`

Modify default FPS, FOV, model sizes, etc.

---

## System Requirements

### Minimum
- CPU: Intel Core i5 or AMD Ryzen 5
- RAM: 8 GB
- Storage: 20 GB free space
- GPU: None (CPU mode works but slower)

### Recommended
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 16 GB
- Storage: 50 GB free space (SSD)
- GPU: NVIDIA RTX 3060 or better (8 GB VRAM)

### For Large Projects
- CPU: Intel Core i9 or AMD Ryzen 9
- RAM: 32 GB
- Storage: 100+ GB SSD
- GPU: NVIDIA RTX 4070 or better (12 GB VRAM)

---

## Platform Support

| Platform | Support | Notes |
|----------|---------|-------|
| Windows 10/11 | ✅ Full | Recommended |
| Linux | ⚠️ Partial | No Insta360 SDK support |
| macOS | ⚠️ Partial | No Insta360 SDK support |

**Note**: Insta360 SDK (Stage 1 extraction) is Windows-only. Other platforms can use FFmpeg/OpenCV methods or process pre-extracted equirectangular images (Stage 2+3 only).

---

## Next Steps

After installation:

1. **Verify Setup**: Run `python test_setup.py`
2. **Read Documentation**: See `README.md` for usage guide
3. **Check UI Spec**: Review `specs/ui_specification.md`
4. **Start Application**: Run `.\run.bat` or `python -m src.main`

---

## Getting Help

- **Logs**: Check `360frametools.log` for detailed errors
- **Test Setup**: Run `python test_setup.py` to diagnose issues
- **Documentation**: See `specs/` folder for technical details

---

**Last Updated**: November 5, 2025  
**Version**: 1.0.0
