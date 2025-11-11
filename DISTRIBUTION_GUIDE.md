# 360FrameTools Distribution Guide

This guide explains how to package and distribute the 360FrameTools application for Windows.

---

## Overview of Distribution Options

### Current Status
- **Built Executable:** `dist/360ToolkitGS-GPU/` (~5 GB)
- **Components:**
  - `360ToolkitGS-GPU.exe` - Main application
  - `_internal/` folder - Dependencies (PyQt6, OpenCV, SDK, FFmpeg)
  - Total size: ~5 GB uncompressed

---

## Option 1: Portable ZIP (Easiest)

**What it is:** Compress the entire folder into a .zip file for distribution.

### Steps:
1. Build the application:
   ```powershell
   pyinstaller 360FrameTools.spec
   ```

2. Add user files to dist folder:
   ```powershell
   Copy-Item "install_pytorch_gpu.bat" "dist\360ToolkitGS-GPU\"
   Copy-Item "README_GPU_VERSION.md" "dist\360ToolkitGS-GPU\README.txt"
   ```

3. Create ZIP archive:
   ```powershell
   Compress-Archive -Path "dist\360ToolkitGS-GPU\*" -DestinationPath "360FrameTools-GPU-v1.0.0-Portable.zip" -CompressionLevel Optimal
   ```

### Result:
- **File:** `360FrameTools-GPU-v1.0.0-Portable.zip` (~2-3 GB compressed)
- **Distribution:** Upload to GitHub Releases, Google Drive, etc.
- **User experience:**
  1. Download ZIP
  2. Extract anywhere
  3. Run `360ToolkitGS-GPU.exe`
  4. Run `install_pytorch_gpu.bat` for masking support

### Pros:
- ✅ No installer needed
- ✅ Can run from USB drive
- ✅ Easy to create
- ✅ No admin rights required

### Cons:
- ❌ No Start Menu shortcuts
- ❌ Users must extract manually
- ❌ Large download (2-3 GB)

---

## Option 2: Windows Installer (Professional)

**What it is:** Create a professional `.exe` installer using Inno Setup (like commercial software).

### Steps:

#### 1. Download Inno Setup (Free)
- Visit: https://jrsoftware.org/isdl.php
- Download **Inno Setup 6** (latest version)
- Install with default settings

#### 2. Build the installer:
```powershell
.\build_installer.bat
```

This will:
1. Verify Inno Setup is installed
2. Check if `dist/360ToolkitGS-GPU/` exists
3. Compile installer (takes 5-10 minutes)
4. Output: `installer_output/360FrameTools-GPU-Setup-v1.0.0.exe`

### Result:
- **File:** `360FrameTools-GPU-Setup-v1.0.0.exe` (~2-3 GB)
- **User experience:**
  1. Download installer
  2. Run installer (requires admin rights)
  3. Application installed to `C:\Program Files\360FrameTools\`
  4. Start Menu shortcuts created
  5. Desktop icon (optional)
  6. Uninstaller in Control Panel

### Installer Features:
- ✅ Professional installation wizard
- ✅ Start Menu shortcuts
- ✅ Desktop icon (optional)
- ✅ Uninstaller in Windows Apps & Features
- ✅ Version checking
- ✅ Checks for Visual C++ Redistributable
- ✅ Post-install option to run PyTorch installer
- ✅ LZMA2 compression (~40% smaller than ZIP)

### Pros:
- ✅ Professional appearance
- ✅ Automatic Start Menu integration
- ✅ Proper uninstaller
- ✅ Better compression
- ✅ Can check for dependencies (VC++ Redistributable)

### Cons:
- ❌ Requires Inno Setup to build
- ❌ Requires admin rights to install
- ❌ Longer build time (compression)

---

## Option 3: MSI Package (Enterprise)

**What it is:** Windows Installer Package (.msi) for enterprise deployment.

### Tools:
- **WiX Toolset:** https://wixtoolset.org/
- **Advanced Installer:** https://www.advancedinstaller.com/ (commercial)

### When to use:
- Corporate environments
- Group Policy deployment
- System administrators
- Need Windows Installer features (repair, rollback)

### Pros:
- ✅ Native Windows Installer
- ✅ GPO deployment support
- ✅ Repair/rollback functionality
- ✅ Better enterprise integration

### Cons:
- ❌ Complex to create
- ❌ Requires WiX Toolset or commercial software
- ❌ Larger file size than Inno Setup

---

## Option 4: Single-File Executable

**What it is:** Pack everything into one giant .exe file (no `_internal` folder).

### How to enable:
Edit `360FrameTools.spec`:
```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,  # Include these
    a.zipfiles,  # Include these
    a.datas,     # Include these
    [],
    name='360ToolkitGS-GPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico',
    onefile=True,  # <-- Add this
)
```

### Result:
- **File:** `360ToolkitGS-GPU.exe` (~5 GB single file)

### Pros:
- ✅ Only one file to distribute
- ✅ Clean, no folders

### Cons:
- ❌ Huge file size (5 GB)
- ❌ Slower startup (extracts to temp on each run)
- ❌ Uses temp disk space on every launch
- ❌ Antivirus may flag large single executables

---

## Recommended Approach

### For Public Release:
**Use Inno Setup Installer** (Option 2)

**Distribution plan:**
1. Build executable: `pyinstaller 360FrameTools.spec`
2. Create installer: `.\build_installer.bat`
3. Upload to GitHub Releases:
   - **File:** `360FrameTools-GPU-Setup-v1.0.0.exe` (~2-3 GB)
   - **Alt:** `360FrameTools-GPU-v1.0.0-Portable.zip` (for users without admin rights)

### For Development/Testing:
**Use Portable ZIP** (Option 1)

Quick distribution to testers without formal installer.

---

## File Size Breakdown

### Uncompressed (5 GB):
- PyQt6: ~1.5 GB (Qt6 DLLs, plugins)
- OpenCV: ~500 MB (opencv_world DLL + dependencies)
- NumPy: ~300 MB
- Insta360 SDK: ~200 MB
- FFmpeg: ~80 MB
- Python runtime: ~50 MB
- Other dependencies: ~2.4 GB

### Compressed:
- **ZIP (Optimal):** ~2.5-3 GB
- **Inno Setup (LZMA2 Ultra):** ~2-2.5 GB (best compression)
- **7-Zip (Ultra):** ~2.3 GB

---

## Reducing Size (Optional)

If 5 GB is too large, you can:

### 1. Remove Unused Qt6 Plugins
PyQt6 bundles many plugins you may not need:
```python
# In 360FrameTools.spec, add to excludes:
excludes += [
    'PyQt6.QtWebEngine',      # ~500 MB (web browser component)
    'PyQt6.QtQuick',          # ~200 MB (QML engine)
    'PyQt6.Qt3D',             # ~150 MB (3D rendering)
    'PyQt6.QtMultimedia',     # ~100 MB (audio/video)
]
```

**Potential savings:** ~1 GB

### 2. Use UPX Compression
Compress DLLs with UPX (built into PyInstaller):
```python
# In 360FrameTools.spec:
exe = EXE(..., upx=True, upx_exclude=[])
```

**Potential savings:** ~500 MB (but slower startup)

### 3. External Dependencies
Don't bundle FFmpeg/SDK, require users to install:
- User installs Insta360 Studio (SDK included)
- User installs FFmpeg separately
- Application detects installed versions

**Savings:** ~300 MB (but worse user experience)

---

## Signing the Executable (Optional)

### Why sign?
- ✅ Windows SmartScreen won't block
- ✅ Users trust the application
- ✅ Professional appearance

### How to sign:
1. Get code signing certificate (~$80-300/year):
   - Sectigo, DigiCert, SSL.com
2. Use `signtool.exe`:
   ```cmd
   signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\360ToolkitGS-GPU\360ToolkitGS-GPU.exe
   ```

### Without signing:
- Windows SmartScreen shows "Unknown Publisher" warning
- Users must click "More info" → "Run anyway"
- Common for open-source software

---

## Testing Checklist

Before distribution, test on **clean Windows machine**:

- [ ] Extract/Install application
- [ ] Run executable - UI opens
- [ ] Check Start Menu shortcuts (installer only)
- [ ] Test Stage 1 (Extraction)
- [ ] Test Stage 2 (Cubemap generation)
- [ ] Run `install_pytorch_gpu.bat`
- [ ] Test Stage 3 (Masking) after PyTorch installation
- [ ] Check uninstaller (installer only)

---

## Quick Reference Commands

```powershell
# Build executable
pyinstaller 360FrameTools.spec

# Create portable ZIP
Compress-Archive -Path "dist\360ToolkitGS-GPU\*" -DestinationPath "360FrameTools-GPU-v1.0.0-Portable.zip"

# Create installer (requires Inno Setup)
.\build_installer.bat

# Test executable
cd dist\360ToolkitGS-GPU
.\360ToolkitGS-GPU.exe
```

---

## Questions?

- **Q: Which option should I use?**
  - **A:** Inno Setup installer for public release, ZIP for quick testing.

- **Q: Why is it 5 GB?**
  - **A:** PyQt6 bundles entire Qt6 framework (~1.5 GB) + OpenCV (~500 MB) + other dependencies.

- **Q: Can I make it smaller?**
  - **A:** Yes, exclude unused Qt modules (see "Reducing Size" section).

- **Q: Do I need to sign the executable?**
  - **A:** No, but it improves trust (Windows won't show SmartScreen warning).

- **Q: What about macOS/Linux?**
  - **A:** PyInstaller supports all platforms. Run the same spec file on macOS/Linux to create native builds.

---

**Ready to distribute!** 🚀
