# 360ToolkitGS - Complete Build Guide

This guide walks you through building distributable Windows executables for both GPU and CPU versions.

---

## 📋 Prerequisites

### Required Software
- [x] **Python 3.8+** (3.10 recommended)
- [x] **PyInstaller** - `pip install pyinstaller`
- [x] **Insta360 SDK** - Downloaded and extracted
- [x] **FFmpeg** - Installed and accessible

### Python Dependencies (Development)
```bash
pip install PyQt6 numpy opencv-python ultralytics piexif
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Paths in Spec Files
Both `360FrameTools.spec` (GPU) and `360ToolkitGS-CPU.spec` contain paths that **must match your system**:

```python
# Update these paths in BOTH spec files:
SDK_PATH = Path('C:/Users/User/Documents/Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1/MediaSDK-3.0.5-20250619-win64/MediaSDK')
FFMPEG_PATH = Path('C:/Program Files (x86)/ffmpeg/bin')
```

**How to find your FFmpeg path**:
```cmd
where ffmpeg
```

---

## 🚀 Quick Start (Build Both Versions)

```bash
cd C:\Users\User\Documents\APLICATIVOS\360ToolKit
build_all.bat
```

Select option **3** to build both versions.

---

## 📦 Build GPU Version

### Step 1: Prepare Environment

```bash
# Ensure PyTorch with CUDA is installed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Step 2: Build

**Option A: Use build script** (recommended)
```bash
build_gpu_version.bat
```

**Option B: Manual build**
```bash
pyinstaller 360FrameTools.spec
```

### Step 3: Verify Output

Check: `dist\360ToolkitGS-GPU\`
- Executable: `360ToolkitGS-GPU.exe`
- Size: ~700 MB
- Includes: `install_pytorch_gpu.bat`, `README.txt`

### Step 4: Test

**On development machine** (with PyTorch):
```bash
cd dist\360ToolkitGS-GPU
.\360ToolkitGS-GPU.exe
```

**On clean machine** (without Python):
1. Copy `dist\360ToolkitGS-GPU\` to test machine
2. Run `install_pytorch_gpu.bat` (installs PyTorch+CUDA)
3. Launch `360ToolkitGS-GPU.exe`
4. Test all 3 stages

---

## 📦 Build CPU Version

### Step 1: Switch to PyTorch CPU

**Important**: CPU version should bundle PyTorch CPU (not CUDA) to avoid 3.5 GB size.

```bash
# Remove CUDA version
pip uninstall torch torchvision torchaudio -y

# Install CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify CPU version
python -c "import torch; print(torch.__version__)"
# Should show: 2.x.x+cpu
```

### Step 2: Build

**Option A: Use build script** (recommended)
```bash
build_cpu_version.bat
```

**Option B: Manual build**
```bash
pyinstaller 360ToolkitGS-CPU.spec
```

### Step 3: Verify Output

Check: `dist\360ToolkitGS-CPU\`
- Executable: `360ToolkitGS-CPU.exe`
- Size: ~800 MB (with PyTorch CPU)
- Includes: `README.txt`

### Step 4: Test

**Critical**: Test on machine **WITHOUT Python installed**
1. Copy `dist\360ToolkitGS-CPU\` to clean machine
2. Launch `360ToolkitGS-CPU.exe` directly
3. Should work immediately without setup
4. Test all 3 stages

---

## 🔧 Troubleshooting Build Issues

### Issue: "Module not found" during build

**Cause**: PyInstaller didn't detect dependency

**Solution**: Add to `hiddenimports` in spec file:
```python
hiddenimports = [
    'missing_module_name',
    # ... existing imports
]
```

---

### Issue: SDK not found at runtime

**Cause**: SDK path incorrect or files not bundled

**Solution**:
1. Verify `SDK_PATH` in spec file is correct
2. Check SDK files exist: `C:\...\MediaSDK\bin\` and `modelfile\`
3. Rebuild with correct path

---

### Issue: FFmpeg not found

**Cause**: FFmpeg path incorrect or not bundled

**Solution**:
1. Find FFmpeg: `where ffmpeg`
2. Update `FFMPEG_PATH` in spec file
3. Rebuild

---

### Issue: CPU version is 3.5 GB (too large)

**Cause**: PyTorch CUDA bundled instead of CPU version

**Solution**:
```bash
# Switch to PyTorch CPU
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Rebuild
pyinstaller 360ToolkitGS-CPU.spec
```

---

### Issue: "GPU requested but not available" in GPU version

**Expected behavior**: GPU version excludes PyTorch, user must install it.

**Solution for users**:
1. Run `install_pytorch_gpu.bat` (included in distribution)
2. Or manually: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

### Issue: Application crashes on startup

**Possible causes**:
1. Missing DLL dependencies
2. Antivirus blocking
3. Corrupted build

**Solutions**:
1. **Test with console**: Change `console=True` in spec file, rebuild, check error messages
2. **Check antivirus**: Temporarily disable and test
3. **Clean rebuild**:
   ```bash
   rmdir /s /q build dist
   pyinstaller <spec_file>.spec
   ```

---

### Issue: PyInstaller not found

**Solution**:
```bash
pip install pyinstaller
```

---

## 📏 Expected Build Sizes

### GPU Version
- **Spec file**: `360FrameTools.spec`
- **Output**: `dist\360ToolkitGS-GPU\`
- **Size**: ~700 MB
- **Contents**:
  - SDK (~200 MB)
  - FFmpeg (~80 MB)
  - Python runtime (~50 MB)
  - Dependencies (OpenCV, NumPy, PyQt6) (~350 MB)
  - Application code (~20 MB)
- **Excludes**: PyTorch (user installs separately)

### CPU Version
- **Spec file**: `360ToolkitGS-CPU.spec`
- **Output**: `dist\360ToolkitGS-CPU\`
- **Size**: ~800 MB (with PyTorch CPU) or **~3.5 GB** (if PyTorch CUDA bundled by mistake)
- **Contents**:
  - Everything from GPU version
  - PyTorch CPU (~500 MB)
- **Includes**: Everything needed to run

**If CPU version is >1 GB**: You bundled PyTorch CUDA by mistake. Reinstall PyTorch CPU and rebuild.

---

## 🎯 Distribution Checklist

### Before Building
- [ ] Updated SDK_PATH and FFMPEG_PATH in spec files
- [ ] Verified FFmpeg exists: `where ffmpeg`
- [ ] For GPU build: PyTorch CUDA installed
- [ ] For CPU build: PyTorch CPU installed (not CUDA!)
- [ ] PyInstaller installed: `pip install pyinstaller`

### After Building
- [ ] GPU version size ~700 MB
- [ ] CPU version size ~800 MB (not 3.5 GB!)
- [ ] Both executables exist
- [ ] README files copied to dist folders
- [ ] install_pytorch_gpu.bat in GPU version folder

### Testing GPU Version
- [ ] Runs on development machine (with PyTorch)
- [ ] Shows "Using device: cuda:0" in Stage 3
- [ ] Tested on clean machine after running install_pytorch_gpu.bat
- [ ] Stage 1 works (SDK extraction)
- [ ] Stage 2 works (perspective splitting)
- [ ] Stage 3 works (masking with GPU)

### Testing CPU Version
- [ ] Runs on clean machine WITHOUT Python
- [ ] Launches immediately without setup
- [ ] Stage 1 works (SDK extraction)
- [ ] Stage 2 works (perspective splitting)
- [ ] Stage 3 works (masking with CPU)
- [ ] Shows "Using device: cpu" in logs

### Create Distribution Packages
- [ ] Compress GPU version:
  ```bash
  cd dist
  tar -a -c -f 360ToolkitGS-GPU.zip 360ToolkitGS-GPU
  ```
- [ ] Compress CPU version:
  ```bash
  cd dist
  tar -a -c -f 360ToolkitGS-CPU.zip 360ToolkitGS-CPU
  ```
- [ ] Verify ZIP sizes: GPU ~700 MB, CPU ~800 MB
- [ ] Test extracting and running from ZIP

---

## 📝 Build Scripts Summary

| Script | Purpose | Usage |
|--------|---------|-------|
| `build_all.bat` | Build both versions | Run and select option |
| `build_gpu_version.bat` | Build GPU version only | Run directly |
| `build_cpu_version.bat` | Build CPU version only | Run directly |

All scripts include:
- Prerequisite checks
- Path verification
- Automatic file copying
- Size reporting
- Error handling

---

## 🔄 Rebuilding After Code Changes

### Quick Rebuild (GPU)
```bash
rmdir /s /q build dist\360ToolkitGS-GPU
pyinstaller 360FrameTools.spec
```

### Quick Rebuild (CPU)
```bash
rmdir /s /q build dist\360ToolkitGS-CPU
pyinstaller 360ToolkitGS-CPU.spec
```

### Full Clean Build
```bash
rmdir /s /q build dist
build_all.bat
```

---

## 🎨 Optional: Adding Application Icon

1. Create `resources/icon.ico` (256×256 PNG → ICO)
2. Update both spec files:
   ```python
   exe = EXE(
       # ... other parameters
       icon='resources/icon.ico',  # Add this line
   )
   ```
3. Rebuild

---

## 📦 Optional: Creating Installers with Inno Setup

### Step 1: Download Inno Setup
https://jrsoftware.org/isinfo.php

### Step 2: Create installer script

**GPU Version** (`installer-gpu.iss`):
```inno
[Setup]
AppName=360ToolkitGS GPU
AppVersion=1.0
DefaultDirName={autopf}\360ToolkitGS-GPU
OutputDir=installers
OutputBaseFilename=360ToolkitGS-GPU-Setup

[Files]
Source: "dist\360ToolkitGS-GPU\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{autodesktop}\360ToolkitGS GPU"; Filename: "{app}\360ToolkitGS-GPU.exe"
Name: "{autoprograms}\360ToolkitGS GPU"; Filename: "{app}\360ToolkitGS-GPU.exe"
```

**CPU Version** (`installer-cpu.iss`):
```inno
[Setup]
AppName=360ToolkitGS CPU
AppVersion=1.0
DefaultDirName={autopf}\360ToolkitGS-CPU
OutputDir=installers
OutputBaseFilename=360ToolkitGS-CPU-Setup

[Files]
Source: "dist\360ToolkitGS-CPU\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{autodesktop}\360ToolkitGS CPU"; Filename: "{app}\360ToolkitGS-CPU.exe"
Name: "{autoprograms}\360ToolkitGS CPU"; Filename: "{app}\360ToolkitGS-CPU.exe"
```

### Step 3: Compile installers
```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer-gpu.iss
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer-cpu.iss
```

Output: `installers\360ToolkitGS-GPU-Setup.exe` and `installers\360ToolkitGS-CPU-Setup.exe`

---

## 📊 Build Time Estimates

| Version | Clean Build | Rebuild | Test Time |
|---------|-------------|---------|-----------|
| GPU | 5-10 min | 2-5 min | 10 min |
| CPU | 8-15 min | 3-8 min | 10 min |
| Both | 15-25 min | 5-13 min | 20 min |

**Note**: Build time depends on:
- CPU speed
- Disk speed (SSD vs HDD)
- Antivirus scanning (can slow significantly)
- Number of dependencies

**Tip**: Exclude `build\` and `dist\` folders from antivirus real-time scanning to speed up builds.

---

## 🐛 Common Build Warnings (Safe to Ignore)

```
WARNING: lib not found: <some_dll> dependent of <some_lib>
```
**Ignore**: PyInstaller may warn about optional DLLs that aren't needed.

```
WARNING: Hidden import "X" not found
```
**Check**: If application works, ignore. Otherwise add to `hiddenimports`.

```
WARNING: Duplicate module <name>
```
**Ignore**: PyInstaller handles this automatically.

---

## ✅ Final Checklist

### Before Release
- [ ] Both versions built successfully
- [ ] Tested on development machine
- [ ] **Tested on clean machine without Python** (critical!)
- [ ] GPU version tested with and without PyTorch
- [ ] CPU version works out-of-box
- [ ] All 3 stages functional in both versions
- [ ] README files included
- [ ] LICENSE file included
- [ ] Distribution ZIPs created
- [ ] File sizes reasonable (GPU ~700 MB, CPU ~800 MB)

### Documentation to Include
- [ ] `README.txt` (version-specific)
- [ ] `LICENSE` (MIT + SDK attribution)
- [ ] `VERSION_COMPARISON.md` (help users choose)
- [ ] `install_pytorch_gpu.bat` (GPU version only)

---

## 📞 Build Support

If builds fail:
1. Check all paths in spec files match your system
2. Verify all prerequisites installed
3. Try clean rebuild: `rmdir /s /q build dist`
4. Enable console mode in spec file: `console=True`
5. Check error messages in terminal

---

**Build script version**: 1.0  
**Last updated**: 2025-01  
**Target OS**: Windows 10/11 (64-bit)
