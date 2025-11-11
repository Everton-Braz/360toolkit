# GPU Support Installation Guide

## Problem Identified

1. **PyTorch CPU-only version installed**: `torch 2.9.0+cpu`
2. **CUDA is available**: NVIDIA GTX 1650 with CUDA 12.9 driver
3. **Solution**: Reinstall PyTorch with CUDA support

## Installation Steps

### Step 1: Uninstall CPU-only PyTorch

```powershell
pip uninstall torch torchvision torchaudio -y
```

### Step 2: Install PyTorch with CUDA 12.1 Support

Since you have CUDA 12.9 driver, you can use PyTorch compiled for CUDA 12.1 (compatible with 12.x drivers):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Alternative for CUDA 11.8** (if 12.1 has issues):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify GPU Installation

```powershell
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
PyTorch version: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce GTX 1650
```

### Step 4: Test YOLOv8 with GPU

```powershell
python -c "from ultralytics import YOLO; import torch; model = YOLO('yolov8n-seg.pt'); model.to('cuda'); print('YOLOv8 on GPU:', next(model.parameters()).device)"
```

## Performance Improvement Expected

With GPU acceleration:
- **YOLOv8 nano**: ~0.05s/image (was ~0.2s on CPU) → **4× faster**
- **YOLOv8 small**: ~0.08s/image (was ~0.5s on CPU) → **6× faster**
- **YOLOv8 medium**: ~0.15s/image (was ~1.0s on CPU) → **7× faster**

## Troubleshooting

### If CUDA is still not detected after installation:

1. **Check PyTorch installation type**:
   ```powershell
   python -c "import torch; print(torch.__version__)"
   ```
   Should show `cu121` or `cu118` in version string (NOT `cpu`)

2. **Check NVIDIA driver**:
   ```powershell
   nvidia-smi
   ```
   Should show CUDA version 12.9 or higher

3. **Reinstall with specific CUDA version**:
   Visit: https://pytorch.org/get-started/locally/
   Select your configuration and copy the install command

### If installation is very slow:

PyTorch CUDA packages are large (~2-3 GB). Be patient during download.

## After Installation

The application will automatically detect and use GPU:
- Stage 3 masking will show: `Using device: cuda:0`
- Processing speed will be significantly faster
- No code changes needed - GPU is auto-detected

## Notes

- **VRAM usage**: GTX 1650 has 4GB VRAM
  - YOLOv8 nano: ~500 MB
  - YOLOv8 small: ~800 MB
  - YOLOv8 medium: ~1.5 GB
  - YOLOv8 large: ~2.5 GB
  - YOLOv8 xlarge: ~4 GB (may need batch_size=1)

- **Recommended models for GTX 1650**:
  - Best speed/accuracy: `small` or `medium`
  - Maximum quality (if processing single images): `large`
  - Avoid `xlarge` on 4GB VRAM unless batch_size=1
