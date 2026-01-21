# 360ToolkitGS Build Instructions

## Overview

This document explains how to build a distributable executable (.exe) for 360ToolkitGS.

**Build Type: ONNX Runtime** (Lightweight, ~500 MB instead of 6+ GB with PyTorch)

## Prerequisites

1. **Python 3.11** (tested, other versions may work)
2. **PyInstaller**: `pip install pyinstaller`
3. **ONNX Runtime GPU**: `pip install onnxruntime-gpu` (for CUDA support)
4. **All app dependencies** installed (see requirements.txt)

## Quick Build

### Option 1: Use the batch script
```cmd
build.bat
```

### Option 2: Manual build
```cmd
# 1. Export ONNX models (one-time, requires ultralytics)
python export_onnx_models.py

# 2. Build with PyInstaller
python -m PyInstaller 360ToolkitGS-Build.spec --noconfirm
```

## Output

- **Location**: `dist/360ToolkitGS/`
- **Main Executable**: `360ToolkitGS.exe`
- **Expected Size**: ~400-600 MB

## Distribution

### Create ZIP for upload:
```cmd
create_release_zip.bat
```

This creates: `releases/360ToolkitGS-v1.1.0-Windows-x64.zip`

### Upload to Gumroad:
1. Go to https://evertonbraz.gumroad.com/l/360toolkit
2. Upload the ZIP file
3. Update version notes

## What's Included in the Build

| Component | Size | Description |
|-----------|------|-------------|
| ONNX Runtime | ~150 MB | CUDA + TensorRT GPU inference |
| Insta360 SDK | ~200 MB | Frame extraction from .INSV files |
| FFmpeg | ~50 MB | Video processing fallback |
| YOLOv8 Models | ~160 MB | Object detection (nano, small, medium) |
| PyQt6 | ~50 MB | GUI framework |
| OpenCV | ~50 MB | Image processing |

## Size Comparison

| Build Type | Size | Notes |
|------------|------|-------|
| **ONNX (this)** | ~500 MB | Lightweight, CPU + GPU |
| PyTorch Full | ~6 GB | Huge, includes all PyTorch |

## Troubleshooting

### Build fails with encoding error
The spec file uses ASCII characters only. If you see encoding errors, ensure your terminal uses UTF-8:
```cmd
chcp 65001
```

### ONNX Runtime not found
```cmd
pip install onnxruntime-gpu
```

### CUDA DLLs not bundled
This is OK - ONNX Runtime includes its own CUDA binaries in the wheel package. The app will use:
1. CUDAExecutionProvider (NVIDIA GPU)
2. CPUExecutionProvider (fallback)

### Missing DLLs at runtime
If the built exe fails with DLL errors:
1. Run from command prompt to see errors
2. Check that all files are in `dist/360ToolkitGS/_internal/`
3. May need to copy missing DLLs manually

## File Structure After Build

```
dist/360ToolkitGS/
├── 360ToolkitGS.exe          # Main executable
├── yolov8n-seg.onnx          # YOLOv8 nano model
├── yolov8s-seg.onnx          # YOLOv8 small model (recommended)
├── yolov8m-seg.onnx          # YOLOv8 medium model
├── _internal/                 # Python runtime + dependencies
│   ├── PyQt6/                # GUI framework
│   ├── cv2/                  # OpenCV
│   ├── onnxruntime/          # ONNX Runtime + CUDA
│   └── ...
├── sdk/                      # Insta360 SDK
│   ├── bin/                  # SDK binaries
│   └── modelfile/            # AI stitch models
└── ffmpeg/                   # FFmpeg binaries
    ├── ffmpeg.exe
    └── ffprobe.exe
```

## GPU Support

The ONNX Runtime build includes:
- **CUDAExecutionProvider** - NVIDIA GPUs (CUDA 12.x)
- **TensorRTExecutionProvider** - TensorRT optimization
- **CPUExecutionProvider** - Fallback for any CPU

No additional CUDA installation required on target machines - the ONNX Runtime wheels include CUDA binaries.

## Development Notes

### To modify what's bundled:
Edit `360ToolkitGS-Build.spec`

### To add hidden imports:
Add to `hiddenimports` list in the spec file

### To exclude more modules:
Add to `excludes` list in the spec file

### Build without SDK (smaller):
Comment out the SDK datas section in the spec file

## Version History

- **v1.1.0**: ONNX Runtime build, ~500 MB
- **v1.0.0**: PyTorch build, ~6 GB
