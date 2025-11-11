# Building Windows .exe - Complete Guide

## Insta360 SDK Licensing Analysis

### ✅ **YOU CAN LEGALLY BUNDLE THE SDK IN YOUR .EXE**

Based on the official Insta360 SDK License Agreement (Section 3.1):

> **(b) You may copy and distribute the SDK as part of the Application Software and upload the Application Software to any third party application store.**

### Key Legal Points:

#### ✅ **ALLOWED:**
1. **Bundle SDK with your application** (.exe)
2. **Distribute the .exe** to end users
3. **Upload to application stores** (Microsoft Store, etc.)
4. **Free or Commercial use** - The license is **royalty-free**
5. **Allow end users to download** your application

#### ❌ **NOT ALLOWED:**
1. **Distribute SDK separately** (standalone basis) - Section 4.1(d)
2. **Reverse engineer** SDK source code
3. **Modify or create derivatives** of SDK itself
4. **Use GPL/Open Source licenses** that would make SDK code open - Section 4.2
5. **Use "Insta360" in your app name** without written permission - Section 9.4

### ⚖️ **License Type: Royalty-Free, Non-Exclusive**

From Section 3.1:
> "Insta360 grants you a **limited, royalty-free, non-transferable, revocable and non-exclusive license**"

**This means:**
- ✅ **No fees to pay** (royalty-free)
- ✅ **Can be commercial or open source** (your choice)
- ✅ **SDK stays bundled** with your app (non-transferable)
- ⚠️ **Insta360 can revoke** license if you violate terms (revocable)

### 📋 **Your Obligations:**

1. **Attribution** (Section 7.2):
   - Make clear **you** (not Insta360) developed the application
   - Don't imply Insta360 endorses your app

2. **Privacy Policy** (Section 16):
   - Must have a privacy policy
   - Inform users that Insta360 SDK is used
   - Get user consent before initializing SDK

3. **User Protection** (Section 7.4):
   - Comply with applicable laws
   - Protect user privacy and personal information

4. **No GPL Contamination** (Section 4.2):
   - Can't combine SDK with GPL-licensed code in a way that would make SDK GPL
   - Your app code can be GPL, but SDK must remain separate

### 🎯 **Recommended License for Your App:**

Since SDK is proprietary (not GPL-compatible), best options:

1. **MIT License** ✅ (Best choice)
   - Permissive, compatible with proprietary SDK
   - Allows commercial use
   - Simple, well-understood

2. **Apache 2.0** ✅
   - Similar to MIT, with patent protection
   - Compatible with proprietary SDK

3. **Proprietary/Commercial** ✅
   - Keep your code closed-source
   - Fully compatible with SDK

4. **GPL v3** ⚠️ (Problematic)
   - Would require SDK source code to be open (not allowed)
   - Avoid unless SDK is dynamically loaded at runtime

---

## Building Windows .exe with PyInstaller

### Overview

We'll use **PyInstaller** to bundle your Python application into a standalone .exe with all dependencies.

### Step 1: Install PyInstaller

```powershell
pip install pyinstaller
```

### Step 2: Understanding Dependency Bundling

#### **Python Dependencies (Auto-bundled by PyInstaller):**
- ✅ **PyTorch** (~2.8 GB) - Bundled automatically
- ✅ **OpenCV** - Bundled automatically
- ✅ **NumPy** - Bundled automatically
- ✅ **Ultralytics (YOLOv8)** - Bundled automatically
- ✅ **PyQt6** - Bundled automatically

#### **External Binaries (Require special handling):**
- ⚠️ **FFmpeg** (.exe) - Must be bundled manually
- ⚠️ **Insta360 SDK** (MediaSDKTest.exe + DLLs) - Must be bundled manually
- ⚠️ **YOLOv8 models** (.pt files) - Must be bundled manually

### Step 3: Create PyInstaller Spec File

Create `360FrameTools.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Paths
sdk_path = Path('C:/Users/User/Documents/Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1/MediaSDK-3.0.5-20250619-win64/MediaSDK')
ffmpeg_path = Path('C:/Program Files (x86)/ffmpeg/bin')

block_cipher = None

# Data files to include
datas = [
    # Insta360 SDK - ENTIRE DIRECTORY
    (str(sdk_path / 'bin'), 'sdk/bin'),
    (str(sdk_path / 'modelfile'), 'sdk/modelfile'),
    
    # FFmpeg executable
    (str(ffmpeg_path / 'ffmpeg.exe'), 'ffmpeg'),
    
    # YOLOv8 models (if you want to bundle them)
    # ('yolov8n-seg.pt', 'models'),
    # ('yolov8s-seg.pt', 'models'),
    # ('yolov8m-seg.pt', 'models'),
]

# Binary files (DLLs, etc.)
binaries = []

# Hidden imports (modules not auto-detected)
hiddenimports = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'cv2',
    'numpy',
    'torch',
    'torchvision',
    'ultralytics',
    'piexif',
]

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='360FrameTools',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging, False for GUI-only
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico',  # Your app icon (optional)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='360FrameTools',
)
```

### Step 4: Update Code for Bundled Paths

You need to update your code to find bundled resources. Add this helper function:

Create `src/utils/resource_path.py`:

```python
"""
Resource path helper for PyInstaller bundled applications
"""
import sys
from pathlib import Path

def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and for PyInstaller
    
    Args:
        relative_path: Relative path to resource (e.g., 'sdk/bin/MediaSDKTest.exe')
    
    Returns:
        Absolute path to resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Running in development mode
        base_path = Path(__file__).parent.parent.parent
    
    return base_path / relative_path
```

### Step 5: Update SDK Extractor

In `src/extraction/sdk_extractor.py`, update SDK detection:

```python
from ..utils.resource_path import get_resource_path

def _detect_sdk(self):
    """Detect Insta360 MediaSDK installation"""
    
    # Try bundled SDK first (for .exe distribution)
    bundled_sdk = get_resource_path('sdk/bin/MediaSDKTest.exe')
    if bundled_sdk.exists():
        self.sdk_path = bundled_sdk.parent
        self.sdk_executable = bundled_sdk
        logger.info(f"Using bundled SDK: {self.sdk_executable}")
        return True
    
    # Fallback to development SDK path
    # ... existing detection code ...
```

### Step 6: Update FFmpeg Path

In `src/extraction/frame_extractor.py`:

```python
from ..utils.resource_path import get_resource_path

def _find_ffmpeg(self):
    """Find ffmpeg executable"""
    
    # Try bundled ffmpeg first
    bundled_ffmpeg = get_resource_path('ffmpeg/ffmpeg.exe')
    if bundled_ffmpeg.exists():
        return str(bundled_ffmpeg)
    
    # Fallback to system PATH
    return shutil.which('ffmpeg')
```

### Step 7: Build the .exe

```powershell
# One-folder distribution (recommended for testing)
pyinstaller 360FrameTools.spec

# The output will be in: dist/360FrameTools/360FrameTools.exe
```

**Output structure:**
```
dist/360FrameTools/
├── 360FrameTools.exe        # Main executable
├── _internal/               # All dependencies
│   ├── torch/               # PyTorch library
│   ├── cv2.dll              # OpenCV
│   ├── Qt6*.dll             # PyQt6
│   ├── sdk/                 # Insta360 SDK
│   │   ├── bin/
│   │   │   ├── MediaSDKTest.exe
│   │   │   └── *.dll
│   │   └── modelfile/
│   │       └── *.ins
│   └── ffmpeg/
│       └── ffmpeg.exe
└── ... (other DLLs and dependencies)
```

### Step 8: Test the Build

```powershell
cd dist/360FrameTools
.\360FrameTools.exe
```

### Step 9: Create Installer (Optional)

Use **Inno Setup** to create a professional installer:

1. Download Inno Setup: https://jrsoftware.org/isinfo.php
2. Create `installer.iss`:

```ini
[Setup]
AppName=360FrameTools
AppVersion=1.0.0
DefaultDirName={autopf}\360FrameTools
DefaultGroupName=360FrameTools
OutputDir=installer_output
OutputBaseFilename=360FrameTools_Setup
Compression=lzma2
SolidCompression=yes

[Files]
Source: "dist\360FrameTools\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\360FrameTools"; Filename: "{app}\360FrameTools.exe"
Name: "{autodesktop}\360FrameTools"; Filename: "{app}\360FrameTools.exe"

[Run]
Filename: "{app}\360FrameTools.exe"; Description: "Launch 360FrameTools"; Flags: nowait postinstall skipifsilent
```

3. Compile with Inno Setup → Creates `360FrameTools_Setup.exe`

---

## Dependency Size Breakdown

**Expected final .exe size:**

| Component | Size | Notes |
|-----------|------|-------|
| **PyTorch (CUDA)** | ~2.8 GB | Largest dependency |
| **Python runtime** | ~50 MB | Embedded Python |
| **PyQt6** | ~150 MB | GUI framework |
| **OpenCV** | ~50 MB | Image processing |
| **Ultralytics** | ~100 MB | YOLOv8 + dependencies |
| **NumPy, etc.** | ~50 MB | Scientific libraries |
| **Insta360 SDK** | ~200 MB | SDK binaries + models |
| **FFmpeg** | ~80 MB | Video processing |
| **Your code** | ~5 MB | Application logic |
| **TOTAL** | **~3.5 GB** | Full distribution |

### Size Optimization Options:

1. **Don't bundle YOLOv8 models** - Download on first run (~100 MB saved)
2. **Use PyTorch CPU version** for CPU-only build (~2.5 GB → ~300 MB)
3. **One-file vs one-folder**:
   - One-folder: Faster startup, easier debugging
   - One-file: Single .exe, slower startup (extracts to temp)

---

## Alternative: One-File Executable

For a single .exe file (slower startup):

```python
# In 360FrameTools.spec, change EXE section:
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,      # Include binaries
    a.zipfiles,      # Include zipfiles
    a.datas,         # Include data files
    [],
    name='360FrameTools',
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
)
```

**Trade-offs:**
- ✅ Single file distribution
- ❌ Slower startup (extracts to %TEMP%)
- ❌ Larger file size (~3.5 GB single file)
- ❌ Antivirus may flag (unpacking behavior)

---

## Troubleshooting

### Issue: "SDK not found" in .exe

**Solution**: Verify SDK is bundled:
```python
# Add debug logging
bundled_sdk = get_resource_path('sdk/bin/MediaSDKTest.exe')
logger.info(f"Looking for SDK at: {bundled_sdk}")
logger.info(f"SDK exists: {bundled_sdk.exists()}")
```

### Issue: FFmpeg not working

**Solution**: Bundle ffmpeg dependencies:
```python
# In spec file, add:
datas = [
    (str(ffmpeg_path / 'ffmpeg.exe'), 'ffmpeg'),
    (str(ffmpeg_path / 'ffprobe.exe'), 'ffmpeg'),  # If needed
]
```

### Issue: CUDA not detected in .exe

**Solution**: Ensure CUDA DLLs are bundled:
- PyInstaller should auto-detect torch CUDA DLLs
- If not, manually add to binaries:
```python
binaries = [
    ('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/*.dll', 'cuda'),
]
```

### Issue: .exe too large

**Solutions**:
1. Don't bundle models - download on demand
2. Use CPU-only PyTorch
3. Exclude unnecessary torch components
4. Use UPX compression (already enabled)

---

## Distribution Checklist

✅ Test .exe on clean Windows machine (without Python installed)
✅ Test SDK extraction with bundled MediaSDKTest.exe
✅ Test FFmpeg video processing
✅ Test GPU masking (CUDA detection)
✅ Create README with license attribution
✅ Include Insta360 SDK attribution in About dialog
✅ Create privacy policy (required by SDK license)
✅ Test on Windows 10 and Windows 11

---

## Legal Compliance

### Required Attribution in Your App:

Add to "About" dialog or README:

```
This application uses the Insta360 Camera SDK.
SDK © Arashi Vision Inc. All rights reserved.
SDK License: https://www.insta360.com/sdk/license

PyTorch: BSD License
OpenCV: Apache 2.0 License
FFmpeg: LGPL/GPL (depending on build)
```

### Privacy Policy Requirements:

Must include statement like:

```
This application uses the Insta360 SDK to process Insta360 camera files.
The SDK may collect anonymous usage statistics to improve functionality.
For more information, see Insta360's Privacy Policy:
https://www.insta360.com/support/supportcourse?post_id=20166
```

---

## Summary

✅ **SDK Bundling**: LEGAL and ALLOWED (royalty-free license)
✅ **Commercial Use**: ALLOWED (no fees)
✅ **Open Source**: ALLOWED (with MIT/Apache license, NOT GPL)
✅ **Distribution**: ALLOWED (as part of your application)
✅ **Dependencies**: All can be bundled in .exe
✅ **Final Size**: ~3.5 GB (due to PyTorch CUDA)

**Next Steps**:
1. Create resource path helper
2. Update SDK/FFmpeg detection code
3. Create .spec file
4. Build with PyInstaller
5. Test on clean machine
6. Create installer (optional)
7. Add license attributions
