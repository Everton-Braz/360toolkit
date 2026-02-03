# 360ToolkitGS Build Options

## Available Builds

### 1. ONNX Build (Lightweight)
**Spec file:** `360ToolkitGS-Build.spec` (or `360ToolkitGS-ONNX.spec`)

**Size:** ~500-600 MB

**GPU Support:**
| Stage | GPU | Method |
|-------|-----|--------|
| Stage 1 (Extract) | ✅ Yes | Insta360 SDK (built-in CUDA) |
| Stage 2 (E2P) | ❌ No | OpenCV CPU fallback |
| Stage 3 (Mask) | ✅ Yes | ONNX Runtime CUDA |

**Best for:**
- Distribution via internet
- Users with limited storage
- When Stage 2 performance is acceptable on CPU

**Build command:**
```powershell
python -m PyInstaller 360ToolkitGS-ONNX.spec --noconfirm
```

---

### 2. Full GPU Build (Maximum Performance)
**Spec file:** `360ToolkitGS-FullGPU.spec`

**Size:** ~12-13 GB

**GPU Support:**
| Stage | GPU | Method |
|-------|-----|--------|
| Stage 1 (Extract) | ✅ Yes | Insta360 SDK (built-in CUDA) |
| Stage 2 (E2P) | ✅ Yes | PyTorch CUDA (TorchE2PTransform) |
| Stage 3 (Mask) | ✅ Yes | ONNX Runtime CUDA |

**Best for:**
- Maximum performance
- Large batch processing
- Professional/local installation

**Build command:**
```powershell
python -m PyInstaller 360ToolkitGS-FullGPU.spec --noconfirm
```

---

## Size Breakdown (Full GPU Build)

| Component | Size |
|-----------|------|
| PyTorch CUDA | ~3.5 GB |
| CUDA/cuDNN libraries | ~2.5 GB |
| OpenCV CUDA | ~1.5 GB |
| Insta360 SDK | ~500 MB |
| ONNX Runtime | ~300 MB |
| ONNX Models | ~170 MB |
| FFmpeg | ~150 MB |
| PyQt6 | ~200 MB |
| Other deps | ~3 GB |

---

## Performance Comparison

### Stage 2 (E2P Transform) - 100 frames @ 4K

| Build | GPU | Time | Speed |
|-------|-----|------|-------|
| ONNX (CPU) | ❌ | ~180s | ~0.5 fps |
| Full GPU | ✅ | ~15s | ~6.5 fps |

**Improvement:** ~12x faster with GPU

---

## Recommended Build Selection

- **For Distribution:** Use ONNX build (~600 MB)
- **For Power Users:** Use Full GPU build (~13 GB)
- **USB/Portable:** Use ONNX build (fits on any USB drive)

---

## Creating Distribution Packages

### ZIP Package (ONNX)
```powershell
Compress-Archive -Path "dist\360ToolkitGS\*" -DestinationPath "360ToolkitGS-ONNX-v1.0.zip"
```

### ZIP Package (Full GPU) - Consider splitting or using 7z
```powershell
# 7-Zip with compression
& "C:\Program Files\7-Zip\7z.exe" a -t7z -mx=5 "360ToolkitGS-FullGPU-v1.0.7z" "dist\360ToolkitGS\*"
```
