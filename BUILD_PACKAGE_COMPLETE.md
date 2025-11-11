# 360ToolkitGS - Build Package Complete ✅

**All files created for building and distributing GPU and CPU versions**

---

## 📦 What's Been Created

### PyInstaller Spec Files
✅ `360FrameTools.spec` - GPU version configuration (PyTorch excluded)  
✅ `360ToolkitGS-CPU.spec` - CPU version configuration (PyTorch bundled)

### Build Scripts
✅ `build_gpu_version.bat` - Build GPU version only  
✅ `build_cpu_version.bat` - Build CPU version only  
✅ `build_all.bat` - Master script to build both versions

### User Installation
✅ `install_pytorch_gpu.bat` - End-user script to install PyTorch+CUDA (GPU version)

### Documentation for Users
✅ `README_GPU_VERSION.md` - Complete guide for GPU version users  
✅ `README_CPU_VERSION.md` - Complete guide for CPU version users  
✅ `VERSION_COMPARISON.md` - Detailed GPU vs CPU comparison  
✅ `DISTRIBUTION_README.md` - Main distribution overview

### Documentation for Developers
✅ `BUILD_GUIDE.md` - Complete build instructions and troubleshooting

---

## 🚀 Quick Start: Build Both Versions

```bash
cd C:\Users\User\Documents\APLICATIVOS\360ToolKit
build_all.bat
```

**Select option 3** to build both versions automatically.

---

## 📋 Build Requirements Checklist

Before building, ensure:

### For GPU Version
- [ ] PyInstaller installed: `pip install pyinstaller`
- [ ] PyTorch with CUDA installed: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- [ ] SDK path correct in `360FrameTools.spec`
- [ ] FFmpeg path correct in `360FrameTools.spec`
- [ ] GPU available: `python -c "import torch; print(torch.cuda.is_available())"`

### For CPU Version
- [ ] PyInstaller installed: `pip install pyinstaller`
- [ ] **PyTorch CPU** installed (NOT CUDA): `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- [ ] Verify CPU version: `python -c "import torch; print('+cpu' in torch.__version__)"`
- [ ] SDK path correct in `360ToolkitGS-CPU.spec`
- [ ] FFmpeg path correct in `360ToolkitGS-CPU.spec`

---

## 📊 Expected Output

### After Building GPU Version

**Location**: `dist\360ToolkitGS-GPU\`

**Contents**:
```
360ToolkitGS-GPU\
├── 360ToolkitGS-GPU.exe        (Main executable)
├── install_pytorch_gpu.bat     (User installation script)
├── README.txt                   (User guide)
├── LICENSE                      (MIT + attributions)
└── _internal\                   (Dependencies)
    ├── sdk\
    │   ├── bin\                (SDK DLLs)
    │   └── modelfile\          (AI stitching models)
    ├── ffmpeg\
    │   └── ffmpeg.exe
    └── ... (other dependencies)
```

**Size**: ~700 MB  
**User requirement**: Install PyTorch+CUDA separately

---

### After Building CPU Version

**Location**: `dist\360ToolkitGS-CPU\`

**Contents**:
```
360ToolkitGS-CPU\
├── 360ToolkitGS-CPU.exe        (Main executable)
├── README.txt                   (User guide)
├── LICENSE                      (MIT + attributions)
└── _internal\                   (Dependencies)
    ├── sdk\
    │   ├── bin\
    │   └── modelfile\
    ├── ffmpeg\
    │   └── ffmpeg.exe
    ├── torch\                  (PyTorch CPU - bundled!)
    └── ... (other dependencies)
```

**Size**: ~800 MB (with PyTorch CPU)  
**User requirement**: None! Works out-of-box

---

## ✅ Testing Checklist

### Test GPU Version

#### On Development Machine (with Python + PyTorch)
- [ ] `dist\360ToolkitGS-GPU\360ToolkitGS-GPU.exe` launches
- [ ] Application opens without errors
- [ ] Stage 1 works (SDK extraction)
- [ ] Stage 2 works (perspective splitting)
- [ ] Stage 3 shows "Using device: cuda:0"
- [ ] Masking produces correct output

#### On Clean Machine (without Python)
- [ ] Copy `dist\360ToolkitGS-GPU\` to test machine
- [ ] Run `install_pytorch_gpu.bat`
- [ ] Script completes successfully
- [ ] `360ToolkitGS-GPU.exe` launches
- [ ] All stages work correctly
- [ ] GPU acceleration active

---

### Test CPU Version

#### On Clean Machine (without Python) - CRITICAL TEST
- [ ] Copy `dist\360ToolkitGS-CPU\` to test machine
- [ ] **Do NOT install Python**
- [ ] Launch `360ToolkitGS-CPU.exe` directly
- [ ] Application opens without errors
- [ ] Stage 1 works (SDK extraction)
- [ ] Stage 2 works (perspective splitting)
- [ ] Stage 3 shows "Using device: cpu"
- [ ] Masking produces correct output

---

## 📤 Distribution Package Creation

### Create ZIP Archives

```bash
cd dist

# GPU version
tar -a -c -f 360ToolkitGS-GPU.zip 360ToolkitGS-GPU

# CPU version
tar -a -c -f 360ToolkitGS-CPU.zip 360ToolkitGS-CPU
```

**Or use PowerShell**:
```powershell
Compress-Archive -Path "dist\360ToolkitGS-GPU" -DestinationPath "360ToolkitGS-GPU.zip"
Compress-Archive -Path "dist\360ToolkitGS-CPU" -DestinationPath "360ToolkitGS-CPU.zip"
```

### Verify ZIP Packages
- [ ] GPU ZIP size: ~700 MB
- [ ] CPU ZIP size: ~800 MB
- [ ] Extract and test both
- [ ] All files present in extracted folders
- [ ] Executables run from extracted folders

---

## 📚 Documentation to Upload

Upload these files to your distribution platform (GitHub, website, etc.):

### Required Downloads
- `360ToolkitGS-GPU.zip` (~700 MB)
- `360ToolkitGS-CPU.zip` (~800 MB)

### Documentation (separate downloads or in repo)
- `VERSION_COMPARISON.md` - Help users choose version
- `DISTRIBUTION_README.md` - Main overview and quick start

### Already Included in ZIPs
- `README.txt` (version-specific guide)
- `LICENSE` (MIT + attributions)
- `install_pytorch_gpu.bat` (GPU version only)

---

## 🎯 Final Checklist Before Release

### Build Quality
- [ ] Both versions built successfully
- [ ] No build errors or critical warnings
- [ ] GPU version ~700 MB (not 3.5 GB!)
- [ ] CPU version ~800 MB (not 3.5 GB!)
- [ ] All files present in dist folders

### Testing
- [ ] GPU version tested on development machine
- [ ] GPU version tested on clean machine
- [ ] CPU version tested on clean machine WITHOUT Python
- [ ] All 3 stages work in both versions
- [ ] GPU masking shows CUDA acceleration
- [ ] CPU masking works (slower but functional)

### Documentation
- [ ] README files copied to dist folders
- [ ] LICENSE file in both dist folders
- [ ] install_pytorch_gpu.bat in GPU dist folder
- [ ] VERSION_COMPARISON.md prepared for distribution
- [ ] DISTRIBUTION_README.md prepared for distribution

### Distribution Packages
- [ ] GPU ZIP created (~700 MB)
- [ ] CPU ZIP created (~800 MB)
- [ ] Both ZIPs tested (extract + run)
- [ ] Documentation files ready for upload

### Legal Compliance (SDK License)
- [ ] MIT License chosen and included
- [ ] SDK attribution in LICENSE file
- [ ] Third-party licenses documented
- [ ] No "Insta360" in application name ✅ (360ToolkitGS)
- [ ] Not distributing SDK standalone ✅ (bundled in app)
- [ ] No GPL license ✅ (using MIT)

---

## 📊 Size Breakdown

### GPU Version (~700 MB)
- SDK: ~200 MB
- FFmpeg: ~80 MB
- Python runtime: ~50 MB
- OpenCV + NumPy: ~150 MB
- PyQt6: ~100 MB
- Other dependencies: ~100 MB
- Application code: ~20 MB
- **PyTorch: EXCLUDED** (user installs ~2.8 GB separately)

### CPU Version (~800 MB)
- Everything from GPU version: ~700 MB
- PyTorch CPU: ~500 MB
- **Total: ~800 MB** (still excludes 2.3 GB CUDA components)

**If your CPU build is >1 GB**: You bundled PyTorch CUDA by mistake!

---

## 🐛 Common Issues & Solutions

### Issue: CPU build is 3.5 GB
**Solution**: PyTorch CUDA was bundled instead of CPU version
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pyinstaller 360ToolkitGS-CPU.spec
```

### Issue: "SDK not found" at runtime
**Solution**: Update SDK_PATH in spec files to match your system
```python
SDK_PATH = Path('C:/Users/User/Documents/Windows_CameraSDK.../MediaSDK')
```

### Issue: "FFmpeg not found"
**Solution**: Update FFMPEG_PATH in spec files
```bash
where ffmpeg  # Find FFmpeg location
# Update spec file with correct path
```

### Issue: PyInstaller not found
**Solution**: Install it
```bash
pip install pyinstaller
```

---

## 🎉 You're Ready!

### Build Workflow Summary

1. **Prepare environment**:
   - Install PyInstaller
   - For GPU build: Install PyTorch CUDA
   - For CPU build: Install PyTorch CPU

2. **Update spec files**:
   - Verify SDK_PATH
   - Verify FFMPEG_PATH

3. **Build**:
   ```bash
   build_all.bat
   # Select option 3 (build both)
   ```

4. **Test both versions**:
   - GPU on dev machine + clean machine
   - CPU on clean machine without Python

5. **Create distribution**:
   - ZIP both dist folders
   - Upload ZIPs + documentation

6. **Release**:
   - Share download links
   - Include VERSION_COMPARISON.md
   - Help users choose right version

---

## 📞 Need Help?

### Build Issues
See `BUILD_GUIDE.md` for detailed troubleshooting

### Size Issues
- GPU should be ~700 MB
- CPU should be ~800 MB
- If larger, check PyTorch version (CPU vs CUDA)

### Runtime Issues
- Test on clean machine without Python
- Check antivirus didn't block files
- Verify all files extracted from ZIP

---

## 🚀 Next Steps

1. **Build both versions**: `build_all.bat`
2. **Test thoroughly**: Especially CPU version on clean machine
3. **Create ZIPs**: Both distribution packages
4. **Upload**: To GitHub releases or your platform
5. **Announce**: Share with photogrammetry community!

---

**All build files created and ready! 🎉**

**Build command**: `build_all.bat`  
**Expected time**: 15-25 minutes  
**Output**: Two distribution-ready versions

---

**Questions?** See `BUILD_GUIDE.md` for complete instructions.

**Ready to build?** Run `build_all.bat` now!
