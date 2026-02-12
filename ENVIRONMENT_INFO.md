# 360ToolKit - Environment Information

## ⚠️ CRITICAL: Use the Correct Conda Environment

### Required Environment: `360pipeline`

**The application MUST be run using the `360pipeline` conda environment.**

```bash
conda activate 360pipeline
python run_app.py
```

OR for testing:
```bash
conda activate 360pipeline
python test_mode_c_pose_transfer.py
```

---

## Environment Details

### Current Setup (February 2026)

**Available Environments:**
- `base` (default) - ❌ Missing YOLO+SAM dependencies
- `360pipeline` - ✅ **USE THIS ONE** - Has all required packages

**Python Locations:**
- Conda (miniconda3): `C:\Users\Everton-PC\miniconda3\python.exe` (Python 3.13.9)
- System Python: `C:\Users\Everton-PC\AppData\Local\Programs\Python\Python311\python.exe`

---

## Package Status

### ✅ In `360pipeline` Environment (READY):

| Package | Version | Status |
|---------|---------|--------|
| **ultralytics** | 8.3.193 | ✅ Installed |
| **torch** | 2.6.0+cu124 | ✅ Installed (CUDA 12.4) |
| **torchvision** | 0.21.0+cu124 | ✅ Installed |
| **segment-anything** | 1.0 | ✅ Installed |
| **pycolmap** | ✅ | Installed |

**GPU Hardware:**
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **CUDA Driver**: 13.1 (Driver 591.74)
- **Compute Capability**: sm_120 (Blackwell architecture)

**⚠️ CUDA Compatibility Note:**
- PyTorch 2.6.0 supports compute capabilities up to sm_90
- RTX 5070 Ti (sm_120) is **newer than PyTorch supports**
- CUDA **IS available** and working, but not fully optimized for sm_120
- For full RTX 5070 Ti optimization, PyTorch 2.7+ (or nightly) is needed
- **Current setup works fine** for YOLO+SAM masking (just not 100% optimized)

### ❌ In `base` Environment (DO NOT USE):

- ultralytics: ❌ NOT installed
- segment-anything: ❌ NOT installed
- torch: 2.10.0+cpu (CPU-only, no GPU support)

---

## Installation Instructions

### ✅ All Packages Already Installed!

The `360pipeline` environment is **ready to use** with:
- ✅ Ultralytics YOLO (8.3.193)
- ✅ Segment Anything (1.0)
- ✅ PyTorch with CUDA 12.4 support (2.6.0+cu124)

### Model Files (Already Downloaded):

- ✅ `yolov8m.pt` - YOLOv8 medium model (present in root)
- ✅ `sam_vit_b_01ec64.pth` - SAM ViT-B checkpoint (present in root)

### Optional: Upgrade to PyTorch Nightly (for full RTX 5070 Ti optimization):

```bash
conda activate 360pipeline
pip uninstall torch torchvision -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
pip install torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Note:** Current PyTorch 2.6.0+cu124 works fine, just not 100% optimized for sm_120.

---

## Test Results Summary

### Last Test Run (Mode C - Flat Folder):

**Command:** `python test_mode_c_pose_transfer.py` (in `base` environment)

**Results:**
- ✅ Flat folder structure: 135 images in `mode_c_test/images/`
- ✅ Descriptive naming: `cam00_yaw+090_pitch+000_1068.jpg`
- ✅ COLMAP reconstruction: 117/135 registered, 1847 3D points
- ✅ LichtFeld export: transforms.json + pointcloud
- ✅ RealityScan export: images + sparse + database
- ⚠️ **YOLO+SAM masking: 0 masks** (because wrong environment)

**Issue:** Test ran in `base` environment which lacks ultralytics/SAM → masking was skipped

**Solution:** Run in `360pipeline` environment:
```bash
conda activate 360pipeline
pip install segment-anything  # Install SAM first
python test_mode_c_pose_transfer.py
```

---Already installed! You may be in the wrong environment. Check:
```bash
conda activate 360pipeline
python -c "from segment_anything import sam_model_registry; print('OK')"
```

### Problem: No masks generated but no errors
**Check:**
1. ✅ Are you in `360pipeline` environment? (`conda env list` - look for `*`)
2. ✅ Are model files present? (`Test-Path yolov8m.pt`, `Test-Path sam_vit_b_01ec64.pth`)
3. ✅ Is ultralytics installed? (`pip show ultralytics`)
4. ✅ Is SAM installed? (`pip show segment-anything`)
5. ✅ Is CUDA working? (`python -c "import torch; print(torch.cuda.is_available())"`)

**All checks above should return OK/True.**

### Problem: Warning about RTX 5070 Ti sm_120 compatibility
**Status:** This is expected. PyTorch 2.6.0 doesn't fully support sm_120 (Blackwell) yet.
**Impact:** Minimal - CUDA still works, just not 100% optimized.
**Solution:** Upgrade to PyTorch nightly (see Installation Instructions above) for full sm_120 support.Are you in `360pipeline` environment? (`conda info --envs`)
2. Are model files present? (`ls yolov8m.pt`, `ls sam_vit_b_01ec64.pth`)
3. Is ultralytics installed? (`pip show ultralytics`)
4. Is SAM installed? (`pip show segment-anything`)

### Problem: CUDA not available
**Expected:** PyTorch in `360pipeline` is CPU-only (torch 2.8.0). Masking will work but slower.
**Solution (optional):** Install CUDA-enabled PyTorch:
```bash
conda ✅ Verify all dependencies in one command:
```bash
conda activate 360pipeline
python -c "from ultralytics import YOLO; from segment_anything import sam_model_registry; import torch; print('✓ YOLO: OK'); print('✓ SAM: OK'); print('✓ PyTorch:', torch.__version__); print('✓ CUDA:', torch.cuda.is_available())"
```
- [ ] Check model files: `Test-Path yolov8m.pt`, `Test-Path sam_vit_b_01ec64.pth`
- [ ] Run app/test: `python test_mode_c_pose_transfer.py`

**Expected output:**
```
✓ YOLO: OK
✓ SAM: OK
✓ PyTorch: 2.6.0+cu124
✓ CUDA: True
```

**If you see the sm_120 compatibility warning:** It's normal and doesn't affect functionality.

## Summary Checklist

Before running the app or tests:

- [ ] Activate `360pipeline` environment: `conda activate 360pipeline`
- [ ] Verify ultralytics: `python -c "from ultralytics import YOLO; print('OK')"`
- [ ] Install SAM if needed: `pip install segment-anything`
- [ ] Verify SAM: `python -c "from segment_anything import sam_model_registry; print('OK')"`
- [ ] Check model files: `ls yolov8m.pt`, `ls sam_vit_b_01ec64.pth`
- [ ] Run app/test: `python test_mode_c_pose_transfer.py`

---

**Last Updated:** February 6, 2026
