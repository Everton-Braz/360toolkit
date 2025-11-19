# -*- mode: python ; coding: utf-8 -*-
#
# 360FrameTools - MINIMAL SPEC (Optimized for Size)
# 
# OPTIMIZATIONS:
# - CPU-only PyTorch recommended (saves ~1.5GB vs GPU version)
# - Excluded unused torch modules (saves ~500MB)
# - Removed torchvision (not used, saves ~500MB)
# - Total potential savings: ~2.5GB
#
# For CPU-only build (RECOMMENDED):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# For GPU build (larger binary):
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
#
# Community solutions for PyTorch bundling:
# - GitHub pytorch/pytorch#56362
# - GitHub pyinstaller/pyinstaller#8348
# - GitHub pyinstaller/pyinstaller#8211

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

# ============================================================================
# PATHS
# ============================================================================

# SDK Path - CORRECTED
SDK_PATH = Path(r"C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1\MediaSDK-3.0.5-20250619-win64\MediaSDK")

# FFmpeg Path - CORRECTED
FFMPEG_EXE = Path(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe")

# Get package paths
import site
site_packages = site.getsitepackages()[0]
torch_path = Path(site_packages) / 'torch'

# PyQt6 - Check both base Python and Conda environments
try:
    import PyQt6
    pyqt6_path = Path(PyQt6.__file__).parent
    print(f"[OK] Found PyQt6 at: {pyqt6_path}")
except ImportError:
    print("[ERROR] PyQt6 not found!")
    sys.exit(1)

print("\n" + "="*70)
print("Building 360ToolkitGS - MINIMAL VERSION (PyTorch GPU Bundled)")
print("="*70)
print(f"Python: {sys.version.split()[0]}")
print(f"Site-packages: {site_packages}")
print(f"PyTorch: {torch_path}")
print(f"Build type: ONE-DIR (required for PyTorch DLLs)")
print("="*70 + "\n")

# ============================================================================
# DATA FILES
# ============================================================================

datas = [
    # Insta360 SDK
    (str(SDK_PATH / 'bin'), 'sdk/bin'),
    (str(SDK_PATH / 'modelfile'), 'sdk/modelfile'),
    
    # PyQt6
    (str(pyqt6_path), 'PyQt6'),
]

# FFmpeg - Bundle if found
if FFMPEG_EXE.exists():
    datas.append((str(FFMPEG_EXE), 'ffmpeg'))
    print("[OK] FFmpeg found and will be bundled")
else:
    print("[WARN] FFmpeg not found - user will need to install separately")

# Collect ONLY what we need
torch_datas = collect_data_files('torch')
ultralytics_datas = collect_data_files('ultralytics')

datas += torch_datas
datas += ultralytics_datas

print(f"[OK] Collected {len(torch_datas)} torch data files")
print(f"[OK] Collected {len(ultralytics_datas)} ultralytics data files")

# ============================================================================
# BINARY FILES (DLLs) - CRITICAL FOR PYTORCH
# ============================================================================

binaries = []

# SOLUTION #1: Manual PyTorch DLL bundling
torch_lib_path = torch_path / 'lib'
if torch_lib_path.exists():
    dll_files = list(torch_lib_path.glob('*.dll'))
    for dll in dll_files:
        binaries.append((str(dll), 'torch/lib'))
    print(f"[OK] Bundled {len(dll_files)} PyTorch DLLs from torch/lib")

# CRITICAL FIX #1: Bundle MSVC Runtime DLLs (REQUIRED - WinError 1114 fix)
# PyTorch depends on Visual C++ Redistributable DLLs
# Search in Windows System32 and Python directory
msvc_dll_names = [
    'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
    'msvcp140_1.dll', 'msvcp140_2.dll',  # May be needed by some PyTorch builds
]

msvc_search_paths = [
    Path(r"C:\Windows\System32"),
    Path(sys.prefix),  # Python installation directory
    Path(sys.prefix) / 'Library' / 'bin',  # Conda path
]

msvc_dlls_found = 0
for dll_name in msvc_dll_names:
    found = False
    for search_path in msvc_search_paths:
        dll_path = search_path / dll_name
        if dll_path.exists():
            binaries.append((str(dll_path), 'torch/lib'))  # Put with torch DLLs
            msvc_dlls_found += 1
            found = True
            break
    if not found:
        print(f"[WARNING] MSVC DLL not found: {dll_name} (may cause runtime errors)")

print(f"[OK] Bundled {msvc_dlls_found} MSVC Runtime DLLs (CRITICAL for PyTorch)")

# CRITICAL FIX #2: Bundle CUDA DLLs from system (only for GPU builds)
# OPTIMIZATION: Skip CUDA bundling if using CPU-only PyTorch
try:
    import torch
    has_cuda = torch.cuda.is_available()
    torch_version_has_cuda = '+cu' in torch.__version__ or 'cuda' in torch.__version__
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {has_cuda}")
    print(f"[INFO] CUDA in version string: {torch_version_has_cuda}")
except:
    has_cuda = False
    torch_version_has_cuda = False
    print("[INFO] Could not detect PyTorch CUDA status")

cuda_dlls_found = 0

if has_cuda or torch_version_has_cuda:
    print("[INFO] GPU-enabled PyTorch detected - bundling CUDA DLLs")
    cuda_paths = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"),
        Path(r"C:\Users\User\miniconda3\Library\bin"),
    ]
    
    cuda_dll_names = [
        'cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll',
        'cufft64_*.dll', 'curand64_*.dll', 'cusparse64_*.dll',
        'cusolver64_*.dll', 'cudnn64_*.dll', 'cudnn_*64_*.dll',
        'nvrtc64_*.dll', 'nvrtc-builtins64_*.dll',
    ]
    
    for cuda_path in cuda_paths:
        if cuda_path.exists():
            for pattern in cuda_dll_names:
                for dll in cuda_path.glob(pattern):
                    binaries.append((str(dll), 'torch/lib'))  # Put in torch/lib so runtime hook finds them
                    cuda_dlls_found += 1
    
    print(f"[OK] Bundled {cuda_dlls_found} CUDA DLLs (required for GPU masking)")
else:
    print("[INFO] CPU-only PyTorch detected - skipping CUDA DLL bundling (saves ~1GB)")

# Collect other binaries
torch_binaries = collect_dynamic_libs('torch')
ultralytics_binaries = collect_dynamic_libs('ultralytics')
cv2_binaries = collect_dynamic_libs('cv2')

binaries += torch_binaries
binaries += ultralytics_binaries
binaries += cv2_binaries

print(f"[OK] Collected {len(torch_binaries)} torch binaries")
print(f"[OK] Collected {len(cv2_binaries)} cv2 binaries")

# ============================================================================
# HIDDEN IMPORTS - MINIMAL (only what we use)
# ============================================================================

hiddenimports = [
    # Python standard library (DO NOT use collect_submodules for stdlib - it breaks)
    'unittest', 'unittest.mock',
    'collections', 'collections.abc',
    'email', 'email.mime',
    'json', 'logging', 'os', 'sys', 'io', 'pathlib',
    'threading', 'queue', 'subprocess', 'shutil', 'glob',
    'tempfile', 'datetime', 'time', 'struct',
    'base64', 'hashlib', 'random', 'math',
    
    # PyQt6
    'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    'PyQt6.sip', 'PyQt6.QtNetwork', 'PyQt6.QtPrintSupport',
    
    # OpenCV
    'cv2', 'cv2.dnn',
    
    # NumPy
    'numpy', 'numpy.core', 'numpy.core._multiarray_umath',
    
    # Application modules (NOT masking - it imports torch)
    'src', 'src.ui', 'src.extraction', 'src.transforms',
    'src.pipeline', 'src.config',
    
    # TORCH AND ULTRALYTICS: DO NOT ADD TO HIDDEN IMPORTS
    # They will be bundled as data files and imported lazily at runtime
    # Adding them here causes PyInstaller to analyze and import during build,
    # triggering WinError 1114. Instead, they're bundled via collect_data_files()
    # and loaded by the runtime hook.
]

# Add PyQt6 submodules (safe via collect_submodules)
hiddenimports += collect_submodules('PyQt6')

excludes = [
    # Exclude EVERYTHING we don't need (reduces size massively)
    'matplotlib', 'scipy', 'IPython', 'jupyter',
    'notebook', 'pandas', 'sklearn', 'seaborn',
    'tensorflow', 'keras', 'onnx', 'onnxruntime',
    'pytorch_lightning', 'lightning_fabric', 'transformers',
    'gradio', 'streamlit', 'fastapi', 'uvicorn',
    'test', 'tests', 'pytest',  # Remove 'unittest' - we NEED it!
    'tkinter', 'PySide2', 'PySide6', 'wx',
    
    # OPTIMIZATION: Exclude unused PyTorch modules (reduces binary by ~500MB)
    'torch.distributed',  # Distributed training (not used)
    'torch.jit',  # JIT compilation (not used)
    'torch.nn.quantized',  # Quantization (not used)
    'torch.onnx',  # ONNX export (not used)
    'torch.autograd.profiler',  # Profiling (not used)
    'torch.utils.tensorboard',  # TensorBoard (not used)
    'torch.cuda.amp',  # Automatic mixed precision (not used)
    'torch.distributed.rpc',  # RPC framework (not used)
    'torch.distributed.pipeline',  # Pipeline parallelism (not used)
    
    # OPTIMIZATION: Exclude torchvision (removed from requirements)
    'torchvision',
    'torchvision.models',
    'torchvision.datasets',
    'torchvision.transforms',
]

print(f"[OK] Configured {len(hiddenimports)} hidden imports")
print(f"[OK] Excluding {len(excludes)} unnecessary packages")

# ============================================================================
# RUNTIME HOOKS
# ============================================================================

# CRITICAL: Runtime hooks must be absolute paths
import os
# In spec files, use SPECPATH instead of __file__
current_dir = SPECPATH

runtime_hooks = [
    os.path.join(current_dir, 'runtime_hook_pytorch.py'),
    os.path.join(current_dir, 'runtime_hook_sdk.py'),
]

print(f"[OK] Using {len(runtime_hooks)} runtime hooks")
print(f"  - {runtime_hooks[0]}")
print(f"  - {runtime_hooks[1]}\n")

# ============================================================================
# ANALYSIS
# ============================================================================

# ============================================================================
# ANALYSIS
# ============================================================================

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludedimports=['torch', 'ultralytics'],  # CRITICAL: Don't analyze these during build
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================================================
# EXECUTABLE - ONE-DIR MODE
# ============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='360ToolkitGS-FULL',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='360ToolkitGS-FULL',
)

print("\n" + "="*70)
print("SPEC FILE READY - Build will start now...")
print("="*70 + "\n")
