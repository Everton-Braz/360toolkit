# 360ToolkitGS - GPU FULL Build Guide

**Complete guide for building standalone Windows executable with PyTorch GPU and Ultralytics bundled**

---

## Overview

This guide documents the **working build configuration** for creating a **GPU-accelerated executable** with all AI dependencies (PyTorch 2.5.1 + CUDA 11.8, Ultralytics 8.3.x) fully bundled.

**Build Result**: `360ToolkitGS-FULL.exe` (~8-10 GB)
- ✅ PyTorch GPU with CUDA support
- ✅ Ultralytics YOLOv8 for AI masking
- ✅ Insta360 MediaSDK for video stitching
- ✅ FFmpeg for video processing
- ✅ Complete standalone - no external dependencies

---

## Prerequisites

### 1. Python Environment with GPU Support

**CRITICAL**: Use a conda environment with PyTorch GPU already installed.

```bash
# Activate environment with PyTorch GPU
conda activate instantsplat  # Or your environment name

# Verify PyTorch GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected output: PyTorch: 2.5.1+cu118, CUDA: True
```

**Environment used in working build**:
- Path: `C:\Users\User\miniconda3\envs\instantsplat`
- Python: 3.10.13
- PyTorch: 2.5.1+cu118
- CUDA Toolkit: 11.8

### 2. Install Required Packages

**Install in the SAME environment as PyTorch GPU**:

```bash
# Activate environment
conda activate instantsplat

# Install PyInstaller
pip install pyinstaller==6.16.0

# Install GUI framework
pip install PyQt6

# Install image metadata
pip install piexif

# Install AI detection (CRITICAL - must be in build environment)
pip install ultralytics

# Install additional dependencies for ultralytics
pip install lap  # Required by ultralytics.trackers

# Verify installations
python -c "import PyQt6, piexif, ultralytics; print('All packages installed')"
```

### 3. Verify SDK and FFmpeg

```bash
# Check SDK path
dir "C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1\bin\MediaSDKTest.exe"

# Check FFmpeg
where ffmpeg
```

---

## Build Configuration

### Spec File: `360FrameTools_FULL.spec`

**Key configuration points**:

```python
# 1. Collect PyTorch with all CUDA libraries
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

# 2. Collect Ultralytics COMPLETELY (not just data files)
ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all('ultralytics')

# 3. Manual PyTorch DLL bundling
pytorch_dlls_lib = [
    (str(dll), '.') 
    for dll in torch_lib_path.glob('*.dll')
]
pytorch_dlls_bin = [
    (str(dll), '.') 
    for dll in torch_bin_path.glob('*.dll')
]

# 4. Runtime hooks for PATH setup
runtime_hooks=[
    'runtime_hook_pytorch.py',  # Adds torch/lib and torch/bin to PATH
    'runtime_hook_sdk.py'        # Configures SDK path
]

# 5. ONE-DIR build (required for DLL loading)
EXE(..., console=True)
COLLECT(..., name='360ToolkitGS-FULL')
```

**Why `collect_all()` for ultralytics is CRITICAL**:
- `collect_data_files('ultralytics')` only gets data files → runtime import fails
- `collect_all('ultralytics')` gets data + binaries + hidden imports → works perfectly

---

## Build Process

### Step 1: Clean Previous Builds

```bash
cd C:\Users\User\Documents\APLICATIVOS\360ToolKit

# Remove old build artifacts
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
```

### Step 2: Activate GPU Environment

```bash
conda activate instantsplat
```

### Step 3: Run PyInstaller

```bash
# Full clean build (first time or after major changes)
pyinstaller 360FrameTools_FULL.spec --clean --noconfirm

# Incremental build (faster, after minor code changes)
pyinstaller 360FrameTools_FULL.spec --noconfirm
```

**Build time**:
- Clean build: ~5-10 minutes (collects 8000+ torch files, 272 ultralytics files)
- Incremental: ~2-3 minutes (only recompiles changed modules)

### Step 4: Verify Build Output

```bash
# Check executable exists
Test-Path "dist\360ToolkitGS-FULL\360ToolkitGS-FULL.exe"

# Check directory size
(Get-ChildItem "dist\360ToolkitGS-FULL" -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
# Expected: ~8-10 GB
```

**Build output structure**:
```
dist/360ToolkitGS-FULL/
├── 360ToolkitGS-FULL.exe          # Main executable
├── _internal/                      # Bundled libraries
│   ├── torch/                      # PyTorch (5-6 GB)
│   │   ├── lib/                    # CUDA DLLs
│   │   └── bin/                    # Torch binaries
│   ├── ultralytics/                # YOLOv8 (150 MB)
│   ├── sdk/                        # Insta360 SDK
│   │   ├── bin/MediaSDKTest.exe
│   │   └── modelfile/*.ins
│   ├── cv2/                        # OpenCV
│   ├── PyQt6/                      # GUI framework
│   └── ...other dependencies
└── base_library.zip                # Compressed Python stdlib
```

---

## Runtime Verification

### Test 1: Launch Application

```bash
cd dist\360ToolkitGS-FULL
.\360ToolkitGS-FULL.exe
```

**Expected console output**:
```
[PyTorch Hook] Starting runtime hook execution...
[PyTorch Hook] Torch lib found: ...\torch\lib
[PyTorch Hook] Added 3 paths to PATH
[PyTorch Hook] Runtime environment configured successfully!
[SDK Hook] Found bundled SDK at: ...\sdk
Starting 360FrameTools v1.0.0
Application window opened
```

### Test 2: Verify PyTorch GPU

Look for this in logs when running Stage 3 (masking):
```
PyTorch loaded successfully. Version: 2.5.1, CUDA available: True
CUDA detection: available=True, version=11.8, devices=1
Using CUDA device: NVIDIA GeForce GTX 1650
Using device: cuda:0
```

### Test 3: Full Pipeline

1. **Stage 1**: Extract frames from `.insv` file
   - SDK should be detected and used
   - Check for `[OK] MediaSDK extraction completed successfully!`

2. **Stage 2**: Split perspectives
   - Process equirectangular → perspective views
   - Or generate cubemap (6-face/8-tile)

3. **Stage 3**: Generate masks
   - YOLOv8 model auto-downloads (first run only)
   - GPU acceleration active
   - Masks created for detected persons/objects

**Success indicators**:
- ✅ No "PyTorch not found" errors
- ✅ No "Ultralytics package not installed" errors
- ✅ CUDA device detected and used
- ✅ All stages complete without crashes
- ✅ Output files generated correctly

---

## Troubleshooting

### Build Errors

**Error**: `NameError: name 'Analysis' is not defined`
```python
# Fix: Add to top of spec file
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
```

**Error**: `Hidden import 'ultralytics' not found`
```bash
# Fix: Install ultralytics in BUILD environment
conda activate instantsplat
pip install ultralytics
```

**Error**: `collect_data_files - skipping data collection for module 'ultralytics'`
```python
# Fix: Change from collect_data_files to collect_all
ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all('ultralytics')
```

### Runtime Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`
- **Cause**: PyTorch DLLs not in PATH
- **Fix**: Verify `runtime_hook_pytorch.py` is included in spec

**Error**: `No module named 'ultralytics'`
- **Cause**: Ultralytics not fully bundled
- **Fix**: Use `collect_all('ultralytics')` not `collect_data_files`

**Error**: `CUDA not available` (but GPU exists)
- **Cause**: CUDA DLLs not bundled
- **Fix**: Ensure `collect_all('torch')` includes binaries

**Error**: `Pipeline Failed: None` popup
- **Cause**: Stage input discovery failing
- **Fix**: Fixed in latest version (calls `discover_stage_input_folder(stage, output_dir)`)

### Performance Issues

**Build takes >30 minutes**:
- Normal for clean builds (collecting 8000+ torch files)
- Use incremental builds (`--noconfirm` without `--clean`) for code changes

**Executable size >15 GB**:
- Check if debug symbols included
- Verify excludes in spec: `excludes=['IPython', 'jupyter', 'notebook', ...]`

**Slow AI masking**:
- Verify CUDA is active: Check logs for `Using device: cuda:0`
- If CPU fallback, check CUDA Toolkit installation

---

## Build Environment Reference

### Working Configuration (Tested & Verified)

```yaml
OS: Windows 10/11 x64
Python: 3.10.13
Environment: C:\Users\User\miniconda3\envs\instantsplat

Core Dependencies:
  - PyTorch: 2.5.1+cu118
  - CUDA Toolkit: 11.8
  - Ultralytics: 8.3.227
  - PyQt6: 6.8.x
  - OpenCV (cv2): 4.10.x
  - NumPy: 1.26.x
  - Pillow: 10.4.x
  - piexif: 1.1.3

Build Tools:
  - PyInstaller: 6.16.0
  - conda: 24.x (miniconda3)

External Tools:
  - Insta360 MediaSDK: 3.0.5-build1
  - FFmpeg: Latest stable
```

### File Sizes Reference

```
Build artifacts:
  - dist\360ToolkitGS-FULL\: ~8-10 GB total
    - PyTorch + CUDA: ~5-6 GB
    - Ultralytics: ~150 MB
    - OpenCV: ~80 MB
    - PyQt6: ~200 MB
    - NumPy/Scipy: ~100 MB
    - Other libs: ~1 GB
    - Application code: ~50 MB
    - SDK + models: ~500 MB
```

---

## Deployment

### Portable Distribution

**Option 1**: ZIP archive
```bash
# Create portable package
Compress-Archive -Path "dist\360ToolkitGS-FULL\*" -DestinationPath "360ToolkitGS-FULL-Portable.zip"
```

**Option 2**: Installer (Inno Setup)
- Use `installer_setup.iss` configuration
- Creates single-file installer with uninstaller
- Adds desktop shortcuts and Start Menu entries

### Minimum System Requirements

**Hardware**:
- CPU: Intel Core i5 or AMD Ryzen 5 (quad-core minimum)
- RAM: 16 GB (8 GB minimum, 32 GB recommended for large batches)
- GPU: NVIDIA GPU with CUDA 11.8 support
  - GTX 1650 or higher
  - 4 GB VRAM minimum (6+ GB recommended)
- Disk: 15 GB free space (10 GB for app, 5+ GB for processing)

**Software**:
- OS: Windows 10 (64-bit) or Windows 11
- GPU Drivers: NVIDIA GeForce Game Ready Driver 526.98 or newer
- No Python, CUDA, or other dependencies needed (all bundled)

---

## Continuous Integration Notes

### Automated Builds

```yaml
# Example GitHub Actions workflow
name: Build GPU Executable

on: [push, workflow_dispatch]

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Miniconda
        uses: conda-incubator/setup-miniconda@v2
        with:
          environment-file: environment_gpu.yml
          
      - name: Install dependencies
        run: |
          pip install pyinstaller pyqt6 piexif ultralytics lap
          
      - name: Build executable
        run: |
          pyinstaller 360FrameTools_FULL.spec --clean --noconfirm
          
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: 360ToolkitGS-FULL
          path: dist/360ToolkitGS-FULL/
```

---

## Version History

### v1.0.0 (November 2025) - Working Build ✅

**Key achievements**:
- ✅ PyTorch 2.5.1 GPU fully bundled with CUDA 11.8
- ✅ Ultralytics 8.3.x working with YOLOv8 models
- ✅ Stage input auto-discovery for standalone stage runs
- ✅ Runtime hooks for DLL PATH management
- ✅ Insta360 SDK integrated for AI stitching
- ✅ Cubemap export (6-face + 8-tile)
- ✅ Multi-category masking (persons, objects, animals)

**Build method**:
- Environment: `instantsplat` conda env with PyTorch GPU pre-installed
- Spec: `360FrameTools_FULL.spec` with `collect_all()` for torch/ultralytics
- Size: ~8-10 GB executable directory
- Runtime: Fully standalone, no external dependencies

---

## Support & References

**Project Documentation**:
- Main README: `README.md`
- UI Specification: `specs/ui_specification.md`
- Cubemap Guide: `guides/GUIDE-Cubemap-8Tiles.md`

**Build Scripts**:
- Full build: `build_gpu_version.bat`
- Spec file: `360FrameTools_FULL.spec`
- Runtime hooks: `runtime_hook_pytorch.py`, `runtime_hook_sdk.py`

**External Resources**:
- PyInstaller docs: https://pyinstaller.org/
- PyTorch installation: https://pytorch.org/get-started/locally/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Insta360 SDK: Contact Insta360 developer support

---

**Last Updated**: November 12, 2025  
**Build Status**: ✅ Working - Production Ready  
**Tested On**: Windows 10/11, NVIDIA GTX 1650, 16GB RAM
