# 360ToolkitGS - Version Comparison Guide

Choose the right version for your needs: **GPU** or **CPU**?

---

## 📊 Quick Comparison Table

| Feature | GPU Version | CPU Version |
|---------|-------------|-------------|
| **Installation** | Extract + run install script | Extract & run immediately |
| **Setup Time** | ~5 minutes (first time) | 0 minutes |
| **Download Size** | ~700 MB | ~800 MB |
| **Disk Space Required** | ~4 GB (with PyTorch+CUDA) | ~1 GB |
| **Hardware Requirements** | NVIDIA GPU (GTX 1650+) | Any CPU (4+ cores) |
| **Stage 1 Speed** (Extraction) | ✅ Same | ✅ Same |
| **Stage 2 Speed** (Splitting) | ✅ Same | ✅ Same |
| **Stage 3 Speed** (Masking) | ⚡ **6-7× faster** | ⚠️ Slower |
| **Works Everywhere** | ❌ NVIDIA GPU only | ✅ Yes |
| **User Friendliness** | ⚠️ Requires setup | ✅ No setup |

---

## ⚡ Performance Comparison (Stage 3 Masking)

### Small YOLOv8 Model

| Scenario | GPU Version | CPU Version | Time Difference |
|----------|-------------|-------------|-----------------|
| 1 image | 0.08s | 0.5s | 6× faster |
| 10 images | 0.8s | 5s | 6× faster |
| 100 images | 8s | 50s | 6× faster |
| 500 images | 40s | 4 min | 6× faster |
| 1000 images | 1.3 min | 8.3 min | 6× faster |

### Medium YOLOv8 Model

| Scenario | GPU Version | CPU Version | Time Difference |
|----------|-------------|-------------|-----------------|
| 1 image | 0.15s | 1.2s | 8× faster |
| 100 images | 15s | 2 min | 8× faster |
| 500 images | 1.2 min | 10 min | 8× faster |

**Key insight**: The more images you process, the more time GPU version saves.

---

## 🎯 Which Version Should You Choose?

### Choose **GPU Version** if:

✅ You have an **NVIDIA GPU** (GTX 1650 or better)  
✅ You have **4+ GB VRAM**  
✅ You process **many images** (500+)  
✅ You need **fastest masking** performance  
✅ You don't mind **5-minute setup** (one-time)  
✅ You have **CUDA drivers** installed  

**Best for**:
- Professional photogrammetry workflows
- Large datasets (1000+ images)
- Batch processing multiple projects
- Time-sensitive projects
- Users with gaming PCs

---

### Choose **CPU Version** if:

✅ You **don't have NVIDIA GPU** (Intel/AMD only)  
✅ You want **simplest installation** (extract & run)  
✅ You process **fewer images** (<100 at a time)  
✅ **Masking speed** is not critical  
✅ You want **fully self-contained** app  
✅ You need **works-everywhere** solution  

**Best for**:
- Beginners and casual users
- Small projects (<100 images)
- Laptops without dedicated GPU
- Quick testing and prototyping
- Users who value simplicity

---

## 💾 Installation Comparison

### GPU Version Setup

1. **Download**: `360ToolkitGS-GPU.zip` (~700 MB)
2. **Extract** to folder
3. **Run** `install_pytorch_gpu.bat` (one-time, ~5 min)
   - Downloads PyTorch+CUDA (~2.8 GB)
   - Verifies GPU detection
4. **Launch** `360ToolkitGS-GPU.exe`

**Total time**: ~10 minutes (first time)  
**Total disk space**: ~4 GB

---

### CPU Version Setup

1. **Download**: `360ToolkitGS-CPU.zip` (~800 MB)
2. **Extract** to folder
3. **Launch** `360ToolkitGS-CPU.exe`

**Total time**: ~1 minute  
**Total disk space**: ~1 GB

---

## 📋 Feature Parity

Both versions have **identical features** for Stages 1 and 2:

### Stage 1: Frame Extraction
- ✅ Insta360 SDK integration
- ✅ .INSV and .mp4 support
- ✅ Configurable FPS (0.1-30)
- ✅ Resolution options (2K-8K)
- ✅ All stitch quality modes

### Stage 2: Perspective Splitting
- ✅ Compass-based positioning
- ✅ Customizable FOV and cameras
- ✅ Interactive preview
- ✅ Multi-ring support (look-up/down)

### Stage 3: AI Masking
- ✅ YOLOv8 (all 5 model sizes)
- ✅ Multi-category detection
- ✅ RealityScan-compatible masks
- ⚡ **GPU version 6-7× faster**

**Only difference**: Masking speed (Stage 3)

---

## 🔧 Hardware Requirements

### GPU Version

**Minimum**:
- NVIDIA GPU: GTX 1650 (4 GB VRAM)
- CUDA compute capability: 3.5+
- RAM: 8 GB
- CPU: 4 cores
- Storage: 5 GB free

**Recommended**:
- NVIDIA GPU: RTX 3060 (8+ GB VRAM)
- RAM: 16 GB
- CPU: 6+ cores
- Storage: 10 GB free

**Check GPU compatibility**:
```cmd
nvidia-smi
```

---

### CPU Version

**Minimum**:
- CPU: Intel Core i5 or AMD Ryzen 5 (4 cores)
- RAM: 8 GB
- Storage: 2 GB free

**Recommended**:
- CPU: Intel Core i7 or AMD Ryzen 7 (6+ cores)
- RAM: 16 GB
- Storage: 5 GB free

**Note**: No GPU required, works on any PC/laptop

---

## 💰 Cost Comparison

| Aspect | GPU Version | CPU Version |
|--------|-------------|-------------|
| **Software** | Free (MIT License) | Free (MIT License) |
| **Download bandwidth** | ~700 MB | ~800 MB |
| **Disk space** | 4 GB | 1 GB |
| **Hardware requirement** | NVIDIA GPU ($150+) | Any CPU |
| **Time saved (500 images)** | 40s | 4 min |

**If you already have NVIDIA GPU**: GPU version is a no-brainer (faster, same features)

**If you need to buy GPU**: CPU version may be sufficient for occasional use

---

## 🔄 Can I Use Both?

**Yes!** Both versions can coexist:

1. Install to **different folders**:
   - `C:\360ToolkitGS-GPU\`
   - `C:\360ToolkitGS-CPU\`

2. Use GPU version on **desktop** (powerful GPU)
3. Use CPU version on **laptop** (no dedicated GPU)

Both use the same project formats and settings.

---

## 📈 Real-World Scenarios

### Scenario 1: Professional Photogrammetry Studio
**Project**: 2000 images from Insta360 X3, need masks
- **GPU version**: 2000 images × 0.08s = 2.7 minutes
- **CPU version**: 2000 images × 0.5s = 16.7 minutes
- **Time saved**: 14 minutes per project
- **Recommendation**: **GPU version** ⚡

---

### Scenario 2: Hobbyist/Student
**Project**: 50 images, occasional use
- **GPU version**: 50 images × 0.08s = 4 seconds (+ 5 min setup first time)
- **CPU version**: 50 images × 0.5s = 25 seconds (no setup)
- **Time saved**: Negligible for small batches
- **Recommendation**: **CPU version** 🎯

---

### Scenario 3: Research Lab
**Project**: Multiple datasets, 500-1000 images each
- **GPU version**: 1000 images × 0.08s = 1.3 minutes per dataset
- **CPU version**: 1000 images × 0.5s = 8.3 minutes per dataset
- **Time saved**: 7 minutes per dataset × many datasets = significant
- **Recommendation**: **GPU version** ⚡

---

### Scenario 4: Traveling Photographer
**Equipment**: Laptop without dedicated GPU
- **GPU version**: Won't work (no NVIDIA GPU)
- **CPU version**: Works perfectly
- **Recommendation**: **CPU version** 🎯 (only option)

---

## 🚀 Quick Decision Tree

```
Do you have NVIDIA GPU?
│
├─ YES → Do you process 100+ images regularly?
│   │
│   ├─ YES → 🎯 **GPU VERSION** (6-7× faster masking)
│   │
│   └─ NO → Your choice:
│       ├─ Want fastest? → GPU VERSION
│       └─ Want simplest? → CPU VERSION
│
└─ NO → 🎯 **CPU VERSION** (only option, works great!)
```

---

## 📝 Summary

| If you value... | Choose... |
|----------------|-----------|
| **Speed** (fastest masking) | GPU Version |
| **Simplicity** (extract & run) | CPU Version |
| **Compatibility** (works everywhere) | CPU Version |
| **Professional workflows** | GPU Version |
| **Occasional use** | CPU Version |
| **Large datasets** (500+ images) | GPU Version |
| **Small projects** (<100 images) | Either version |

---

## ❓ Still Not Sure?

**Try CPU version first**:
- Download is similar size (~800 MB)
- Works immediately (no setup)
- Test the workflow and features
- If masking is too slow → upgrade to GPU version

**You can always switch later!** Both versions use same project formats.

---

## 📞 Need Help Deciding?

Consider these questions:

1. **Do you have NVIDIA GPU?**
   - Yes + 4GB VRAM → GPU version
   - No → CPU version

2. **How many images do you typically process?**
   - <100 → CPU version is fine
   - 100-500 → GPU version saves minutes
   - 500+ → GPU version saves hours

3. **How often do you use masking?**
   - Rarely → CPU version
   - Daily → GPU version

4. **What's your priority?**
   - Speed → GPU version
   - Simplicity → CPU version

---

**Still unsure? Start with CPU version (works for everyone, no setup required!)**

You can always upgrade to GPU version later if you need more speed.

---

**Built with ❤️ for the photogrammetry and Gaussian splatting community**
