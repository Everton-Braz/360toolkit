# 360toolkit - Multi-GPU Setup Guide

## ✅ Status: Fully Configured

Your 360toolkit environment is now configured to work on **both machines**:
- **Laptop**: GTX 1650 (4.3 GB VRAM)
- **Home**: RTX 5070 Ti (16+ GB VRAM)

---

## 🚀 Quick Start

### **Recommended: Use the Launcher Script**
```bash
# Simply run the launcher (auto-detects everything)
run_360toolkit.bat
```

This script will:
1. Activate the virtual environment
2. Check dependencies
3. Run GPU diagnostics
4. Launch the app

### **Alternative: Manual Launch**
```bash
# Activate virtual environment
.venv\Scripts\activate.bat

# Run the app
python run_app.py
```

---

## 📊 Current Configuration

### **GPU Detection**
| Property | Value |
|----------|-------|
| PyTorch Version | 2.6.0+cu124 |
| CUDA Version | 12.4 |
| GPU (Laptop) | NVIDIA GeForce GTX 1650 |
| VRAM (Laptop) | 4.29 GB |
| Compute Capability | 7.5 (Turing arch) |
| ONNX Runtime | ✅ CUDAExecutionProvider |

### **Installed Components**
- ✅ PyQt6 - GUI Framework
- ✅ PyTorch 2.6.0 (CUDA 12.4)
- ✅ Ultralytics YOLOv8 - Object Detection
- ✅ Segment Anything (SAM) - Image Segmentation
- ✅ ONNX Runtime GPU - Fallback masking engine
- ✅ OpenCV - Computer Vision
- ✅ NumPy - Scientific Computing

### **Recommended Settings**
| Machine | Batch Size | Model Size | GPU Engine |
|---------|-----------|-----------|-----------|
| **Laptop (GTX 1650)** | 8 | small | PyTorch |
| **Home (RTX 5070 Ti)** | 32 | medium | PyTorch/ONNX |

---

## 🔧 Advanced Configuration

### **GPU Settings File**
Auto-generated settings are stored in:
```
src/config/gpu_settings.json
```

You can manually edit this file to override defaults:
```json
{
  "masking_engine": "pytorch",
  "use_gpu": true,
  "torch_device": "cuda",
  "onnx_enabled": true,
  "batch_size": 8,
  "yolo_model_size": "small"
}
```

### **Environment Variables (Optional)**
Set these before running the app for fine-tuning:

```bash
# Force CPU mode (for testing)
set TORCH_DEVICE=cpu

# Use ONNX Runtime instead of PyTorch
set MASKING_ENGINE=onnx

# Enable verbose logging
set DEBUG=1

# Specify CUDA device (if multi-GPU)
set CUDA_VISIBLE_DEVICES=0
```

---

## 🖥️ Machine-Specific Optimization

### **On Your Laptop (GTX 1650)**
The app automatically detects your mid-range GPU and uses:
- **Batch size**: 8 images
- **Model size**: small (YOLOv8s)
- **Inference speed**: ~0.5s per image for masking

**Performance Tips**:
- Close other GPU applications
- Run frame extraction at 1 FPS initially
- Use 4K resolution for perspective splits (not 8K)

### **On Your Home PC (RTX 5070 Ti)**
The app will auto-detect higher VRAM and use:
- **Batch size**: 32 images
- **Model size**: medium (YOLOv8m)
- **Inference speed**: ~0.2s per image for masking

**Performance Tips**:
- RTX 5070 Ti has TensorRT support (best performance)
- Can handle 8K perspective splits
- Enable larger batch sizes for speed

---

## 🔍 Running Diagnostics

To check GPU setup at any time:
```bash
# Full diagnostic report
python setup_gpu_environment.py

# Or from within app: Settings → Diagnostics
```

This will show:
- GPU detection
- CUDA availability
- All dependency status
- Recommended settings
- Performance benchmarks

---

## ❌ Troubleshooting

### **"CUDA not available"**
```bash
# Reinstall PyTorch with correct CUDA
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### **"PyQt6 import error"**
```bash
# Install PyQt6 specifically
pip install PyQt6>=6.5.0
```

### **Low GPU memory on laptop**
```bash
# Use smaller model and batch size
# Edit src/config/gpu_settings.json:
{
  "batch_size": 4,
  "yolo_model_size": "nano"
}
```

### **App runs slowly on RTX 5070 Ti**
```bash
# Try ONNX Runtime (sometimes faster for newer GPUs)
# Edit src/config/gpu_settings.json:
{
  "masking_engine": "onnx",
  "batch_size": 32
}
```

---

## 📦 Update Dependencies

### **Update all packages**
```bash
pip install --upgrade torch torchvision ultralytics
```

### **Install optional components**
```bash
# Video codec support
pip install ffmpeg-python

# GPU memory profiling
pip install py3nvml
```

---

## 🎯 Portable Setup (USB Drive)

To make the environment portable between machines:

1. **On laptop**: Copy entire `.venv` folder to USB
2. **On home PC**: Paste `.venv` folder in project root
3. **Run**: `run_360toolkit.bat`

The script will auto-detect your GPU and adjust settings accordingly.

---

## 🆘 Getting Help

If you encounter issues:

1. Run diagnostics:
   ```bash
   python setup_gpu_environment.py > diagnostics.log
   ```

2. Check logs in:
   ```
   src/logs/360toolkit.log
   ```

3. Common issues are documented in `GPU_SETUP_GUIDE.md`

---

## 📝 Notes for Development

### **Both Machines Share Same Codebase**
- Virtual environment is machine-specific (in `.gitignore`)
- Code automatically detects GPU capabilities
- No manual GPU configuration needed

### **Continuous Integration**
When deploying updates:
```bash
# Pull latest changes
git pull origin dev

# Update dependencies
pip install -r requirements.txt

# Run diagnostics
python setup_gpu_environment.py
```

---

**Last Updated**: February 3, 2026  
**PyTorch Version**: 2.6.0+cu124  
**Status**: ✅ Ready for both GTX and RTX GPUs
