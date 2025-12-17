# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for 360ToolkitGS
ONNX Version - Lightweight (No PyTorch)

This version:
- Bundles SDK, FFmpeg, and ONNX Runtime
- EXCLUDES PyTorch and Ultralytics (saves ~6GB)
- Bundles ONNX models directly
- Final size: ~500 MB
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

# PyQt6 path (in 360toolkit-cpu conda environment)
pyqt6_path = Path(r'C:\Users\User\miniconda3\envs\360toolkit-cpu\Lib\site-packages\PyQt6')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Insta360 SDK path
SDK_PATH = Path('C:/Users/User/Documents/Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1/MediaSDK-3.0.5-20250619-win64/MediaSDK')

# FFmpeg path
FFMPEG_PATH = Path('C:/Program Files (x86)/ffmpeg/bin')

# BUILD TYPE
BUILD_TYPE = 'ONNX'

# ============================================================================

block_cipher = None

# Data files to include
datas = [
    # Insta360 SDK
    (str(SDK_PATH / 'bin'), 'sdk/bin'),
    (str(SDK_PATH / 'modelfile'), 'sdk/modelfile'),
    
    # FFmpeg
    (str(FFMPEG_PATH / 'ffmpeg.exe'), 'ffmpeg'),
    (str(FFMPEG_PATH / 'ffprobe.exe'), 'ffmpeg'),
    
    # ONNX Models (Bundle all exported models)
    ('yolov8n-seg.onnx', '.'),
    ('yolov8s-seg.onnx', '.'),
    ('yolov8m-seg.onnx', '.'),
    
    # PyQt6
    (str(pyqt6_path), 'PyQt6'),
]

# Binaries
binaries = []

# Excludes - CRITICAL for size reduction
excludes = [
    # PyTorch (Huge savings ~4GB)
    'torch',
    'torchvision',
    'torchaudio',
    'caffe2',
    '_C',
    
    # Ultralytics (Not needed for ONNX inference, ~500MB)
    'ultralytics',
    
    # Heavy scientific libs (save ~800MB)
    # NOTE: sympy removed from excludes - needed by onnxruntime
    'IPython',
    'jupyter',
    'notebook',
    'matplotlib',
    'scipy',
    'pandas',
    'networkx',
    'sklearn',
    'seaborn',
    
    # PyQt5 (we use PyQt6)
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    
    # Unused Qt modules (save ~100MB)
    'PyQt6.Qt3DAnimation',
    'PyQt6.Qt3DCore',
    'PyQt6.Qt3DExtras',
    'PyQt6.Qt3DInput',
    'PyQt6.Qt3DLogic',
    'PyQt6.Qt3DRender',
    'PyQt6.QtBluetooth',
    'PyQt6.QtCharts',
    'PyQt6.QtDataVisualization',
    'PyQt6.QtDesigner',
    'PyQt6.QtHelp',
    'PyQt6.QtLocation',
    'PyQt6.QtMultimedia',
    'PyQt6.QtMultimediaWidgets',
    'PyQt6.QtNfc',
    'PyQt6.QtOpenGL',
    'PyQt6.QtOpenGLWidgets',
    'PyQt6.QtPdf',
    'PyQt6.QtPdfWidgets',
    'PyQt6.QtPositioning',
    'PyQt6.QtPrintSupport',
    'PyQt6.QtQml',
    'PyQt6.QtQuick',
    'PyQt6.QtQuick3D',
    'PyQt6.QtQuickWidgets',
    'PyQt6.QtRemoteObjects',
    'PyQt6.QtSensors',
    'PyQt6.QtSerialPort',
    'PyQt6.QtSpatialAudio',
    'PyQt6.QtSql',
    'PyQt6.QtStateMachine',
    'PyQt6.QtSvg',
    'PyQt6.QtSvgWidgets',
    'PyQt6.QtTest',
    'PyQt6.QtTextToSpeech',
    'PyQt6.QtWebChannel',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebEngineQuick',
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebSockets',
    'PyQt6.QtXml',
    
    # Testing/debugging
    'pytest',
    'unittest',
    'pdb',
    'pydoc',
]

print(f"\n{'='*70}")
print(f"Building 360ToolkitGS - {BUILD_TYPE} Version")
print(f"{'='*70}")
print(f"PyTorch: EXCLUDED (Using ONNX Runtime)")
print(f"SDK: BUNDLED")
print(f"FFmpeg: BUNDLED")
print(f"Expected size: ~500 MB")
print(f"{'='*70}\n")

# Hidden imports
hiddenimports = collect_submodules('PyQt6') + [
    # Core
    'numpy',
    'cv2',
    'PIL',
    'piexif',
    
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
]

# Collect ONNX Runtime dependencies (single comprehensive collection)
print("Collecting ONNX Runtime dependencies...")
import os
import glob
try:
    import onnxruntime
    ort_path = os.path.dirname(onnxruntime.__file__)
    print(f"ONNX Runtime found at: {ort_path}")
    
    # Collect all ONNX Runtime files
    tmp_ret = collect_all('onnxruntime')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
    
    # CRITICAL: Explicitly bundle all DLLs to root AND preserve structure
    capi_path = os.path.join(ort_path, 'capi')
    if os.path.exists(capi_path):
        for dll in glob.glob(os.path.join(capi_path, '*.dll')):
            dll_name = os.path.basename(dll)
            # Add to root (for direct access)
            binaries.append((dll, '.'))
            # Keep original structure too
            binaries.append((dll, 'onnxruntime/capi'))
            print(f"  Bundling: {dll_name}")
    
    # Also check for .libs folder (pip install pattern)
    libs_pattern = os.path.join(os.path.dirname(ort_path), 'onnxruntime.libs')
    if os.path.exists(libs_pattern):
        for dll in glob.glob(os.path.join(libs_pattern, '*.dll')):
            binaries.append((dll, '.'))
            print(f"  Bundling: {os.path.basename(dll)}")
            
    # NUCLEAR OPTION: Bundle all DLLs from Conda Library/bin
    # This fixes missing dependencies like zlib.dll, libpng.dll, etc.
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        library_bin = os.path.join(conda_prefix, 'Library', 'bin')
        if os.path.exists(library_bin):
            print(f"Bundling DLLs from Conda Library/bin: {library_bin}")
            # Common dependencies that might be missing
            target_dlls = ['zlib.dll', 'libpng16.dll', 'jpeg.dll', 'tiff.dll', 'libwebp.dll', 'openblas.dll', 'libiomp5md.dll']
            for dll_name in target_dlls:
                dll_path = os.path.join(library_bin, dll_name)
                if os.path.exists(dll_path):
                    binaries.append((dll_path, '.'))
                    print(f"  Bundling dependency: {dll_name}")
            
            # Also bundle everything if we are desperate (optional, but safer to be specific first)
            # for dll in glob.glob(os.path.join(library_bin, '*.dll')):
            #     binaries.append((dll, '.'))

    # Bundle critical CUDA DLLs required by onnxruntime-gpu
    cuda_env_candidates = [
        os.environ.get('CUDA_PATH'),
        os.environ.get('CUDA_PATH_V12_2'),
        os.environ.get('CUDA_PATH_V12_1'),
        os.environ.get('CUDA_PATH_V11_8')
    ]
    cuda_patterns = [
        'cudnn*.dll',
        'cublas*.dll',
        'cublasLt*.dll',
        'cufft*.dll',
        'curand*.dll',
        'cusolver*.dll',
        'cusparse*.dll',
        'nvrtc*.dll'
    ]
    for cuda_root in [p for p in cuda_env_candidates if p]:
        cuda_bin = os.path.join(cuda_root, 'bin')
        if not os.path.exists(cuda_bin):
            continue
        print(f"Bundling CUDA DLLs from: {cuda_bin}")
        for pattern in cuda_patterns:
            for dll in glob.glob(os.path.join(cuda_bin, pattern)):
                dll_name = os.path.basename(dll)
                binaries.append((dll, '.'))
                print(f"  CUDA DLL: {dll_name}")

    print("ONNX Runtime bundled successfully!")
except ImportError as e:
    print(f"WARNING: ONNX Runtime not found - {e}")
    print("Install with: pip install onnxruntime")

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=['runtime_hook_onnx.py'],  # Add ONNX DLL path setup
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================================
# FILTER DUPLICATE DLLs (Fix for huge size)
# ============================================================================
# PyInstaller sometimes picks up the SDK DLLs (OpenCV CUDA) and puts them in root
# even though they are already in sdk/bin. We remove them from the root binaries.

print("Filtering binaries to remove duplicates and SDK files from root...")
original_count = len(a.binaries)
duplicate_patterns = ('opencv_cuda',)
essential_cuda_patterns = ('cublas', 'cufft', 'npp')

def should_remove_binary(entry):
    """Only drop known duplicate OpenCV CUDA DLLs, keep CUDA runtimes."""
    name_lower = entry[0].lower()
    source_lower = str(entry[1]).replace('\\', '/').lower()

    if any(pattern in name_lower for pattern in essential_cuda_patterns):
        return False

    if 'sdk/bin' in source_lower:
        return False

    return any(pattern in name_lower for pattern in duplicate_patterns)

a.binaries = [entry for entry in a.binaries if not should_remove_binary(entry)]
filtered_count = len(a.binaries)
print(f"Removed {original_count - filtered_count} files from root binaries.")
# ============================================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='360ToolkitGS-ONNX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='360ToolkitGS-ONNX',
)
