# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for 360ToolkitGS
Universal GPU Version - RTX 30/40/50 Series Compatible

This version:
- Bundles PyTorch with CUDA 12.8 (supports sm_50 through sm_120)
- RTX 30xx (sm_86), RTX 40xx (sm_89), RTX 50xx (sm_120) - ALL NATIVE
- Bundles Ultralytics YOLOv8 + SAM for hybrid masking
- Bundles Insta360 SDK + FFmpeg + SphereSfM/COLMAP
- GPU auto-detection with graceful CPU fallback
"""

import sys
import os
import glob
import shutil
import atexit
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ============================================================================
# PRE-BUILD: Patch torch source files BEFORE PyInstaller collects them.
#
# Why: PyInstaller's PYZ builder compiles from original source during Analysis.
# Modifying a.pure entries afterward does NOT affect the compiled bytecode in
# PYZ. The ONLY reliable way is to patch the source files themselves.
#
# Files patched:
# 1. torch/__init__.py - Split "from torch import (30 items)" into individual
#    imports so FrozenImporter sets parent attrs after each one
# 2. torch/distributed/__init__.py - Replace with minimal stub (avoids circular
#    import from "import torch" and "import pdb" at module level)
# 3. torch/distributed/rpc/__init__.py - Replace with minimal stub
# 4. torch/_jit_internal.py - Guard bare "import torch.distributed.rpc"
# 5. torch/_library/infer_schema.py - Guard distributed.is_available()
# 6. torch/nn/parallel/__init__.py - Guard DDP import
#
# All originals are backed up and restored after build (via atexit).
# ============================================================================

CONDA_PREFIX = os.environ.get('CONDA_PREFIX', '')
SITE_PACKAGES = os.path.join(CONDA_PREFIX, 'Lib', 'site-packages') if CONDA_PREFIX else ''
_TORCH_DIR = os.path.join(SITE_PACKAGES, 'torch') if SITE_PACKAGES else ''

_backed_up_files = []  # List of (original_path, backup_path)

def _restore_all():
    """Restore all patched files from backups."""
    for orig, bak in _backed_up_files:
        if os.path.exists(bak):
            shutil.move(bak, orig)
            print(f"  [RESTORE] {orig}")

atexit.register(_restore_all)

def _patch_file(rel_path, new_content=None, replacements=None):
    """Patch a file in the torch package. Either replace entirely or apply text subs."""
    target = os.path.join(_TORCH_DIR, rel_path)
    if not os.path.exists(target):
        print(f"  [SKIP] {rel_path} not found")
        return False
    
    backup = target + '.360bak'
    shutil.copy2(target, backup)
    _backed_up_files.append((target, backup))
    
    if new_content is not None:
        with open(target, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  [PATCH] {rel_path} (full replacement)")
        return True
    
    if replacements:
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        changed = False
        for old_text, new_text in replacements:
            if old_text in content:
                content = content.replace(old_text, new_text)
                changed = True
            else:
                print(f"  [WARN] Text not found in {rel_path}: {old_text[:60]}...")
        if changed:
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [PATCH] {rel_path} (text replacements)")
        return changed
    return False

if _TORCH_DIR and os.path.isdir(_TORCH_DIR):
    print("\n[PRE-BUILD] Patching torch source files for frozen app...")
    
    # 1. torch/__init__.py - Split the big from-import block
    _patch_file('__init__.py', replacements=[
        ('from torch import (\n'
         '    __config__ as __config__,\n'
         '    __future__ as __future__,\n'
         '    _awaits as _awaits,\n'
         '    accelerator as accelerator,\n'
         '    autograd as autograd,\n'
         '    backends as backends,\n'
         '    cpu as cpu,\n'
         '    cuda as cuda,\n'
         '    distributed as distributed,\n'
         '    distributions as distributions,\n'
         '    fft as fft,\n'
         '    futures as futures,\n'
         '    hub as hub,\n'
         '    jit as jit,\n'
         '    linalg as linalg,\n'
         '    mps as mps,\n'
         '    mtia as mtia,\n'
         '    multiprocessing as multiprocessing,\n'
         '    nested as nested,\n'
         '    nn as nn,\n'
         '    optim as optim,\n'
         '    overrides as overrides,\n'
         '    profiler as profiler,\n'
         '    sparse as sparse,\n'
         '    special as special,\n'
         '    testing as testing,\n'
         '    types as types,\n'
         '    utils as utils,\n'
         '    version as version,\n'
         '    xpu as xpu,\n'
         ')\n',
         # Individual imports - each sets parent attr immediately in FrozenImporter
         'import torch.__config__ as __config__  # PATCHED: split for FrozenImporter\n'
         'import torch.__future__ as __future__\n'
         'import torch._awaits as _awaits\n'
         'import torch.accelerator as accelerator\n'
         'import torch.autograd as autograd\n'
         'import torch.backends as backends\n'
         'import torch.cpu as cpu\n'
         'import torch.cuda as cuda\n'
         'import torch.distributed as distributed\n'
         'import torch.distributions as distributions\n'
         'import torch.fft as fft\n'
         'import torch.futures as futures\n'
         'import torch.hub as hub\n'
         'import torch.jit as jit\n'
         'import torch.linalg as linalg\n'
         'import torch.mps as mps\n'
         'import torch.mtia as mtia\n'
         'import torch.multiprocessing as multiprocessing\n'
         'import torch.nested as nested\n'
         'import torch.nn as nn\n'
         'import torch.optim as optim\n'
         'import torch.overrides as overrides\n'
         'import torch.profiler as profiler\n'
         'import torch.sparse as sparse\n'
         'import torch.special as special\n'
         'import torch.testing as testing\n'
         'import torch.types as types\n'
         'import torch.utils as utils\n'
         'import torch.version as version\n'
         'import torch.xpu as xpu\n'),
    ])
    
    # 2. torch/distributed/__init__.py - Minimal stub
    _patch_file(os.path.join('distributed', '__init__.py'), new_content='''\
# PATCHED: Minimal stub for frozen app (single-GPU inference only)
# Original does: import pdb, import torch (circular!), torch._C._c10d_init()
# This stub avoids all circular imports and just returns False for is_available()

def is_available():
    return False

def is_initialized():
    return False

def is_mpi_available():
    return False

def is_nccl_available():
    return False

def is_gloo_available():
    return False

def is_torchelastic_launched():
    return False
''')
    
    # 3. torch/distributed/rpc/__init__.py - Minimal stub
    _patch_file(os.path.join('distributed', 'rpc', '__init__.py'), new_content='''\
# PATCHED: Minimal stub for frozen app
def is_available():
    return False
''')
    
    # 4. torch/_jit_internal.py - Guard bare import
    _patch_file('_jit_internal.py', replacements=[
        ('import torch.distributed.rpc\n',
         'try:  # PATCHED for frozen app\n    import torch.distributed.rpc\nexcept (ImportError, ModuleNotFoundError):\n    pass\n'),
        ('if torch.distributed.rpc.is_available():',
         'if getattr(getattr(getattr(torch, "distributed", None), "rpc", None), "is_available", lambda: False)():  # PATCHED'),
    ])
    
    # 5. torch/_library/infer_schema.py
    _patch_file(os.path.join('_library', 'infer_schema.py'), replacements=[
        ('if torch.distributed.is_available():',
         'if getattr(getattr(torch, "distributed", None), "is_available", lambda: False)():  # PATCHED'),
    ])
    
    # 6. torch/nn/parallel/__init__.py
    _patch_file(os.path.join('nn', 'parallel', '__init__.py'), replacements=[
        ('from torch.nn.parallel.distributed import DistributedDataParallel\n',
         'try:  # PATCHED for frozen app\n    from torch.nn.parallel.distributed import DistributedDataParallel\nexcept (ImportError, RuntimeError):\n    pass\n'),
    ])
    
    print(f"  Total: {len(_backed_up_files)} files patched (will restore after build)\n")
else:
    print("[WARN] Torch directory not found, skipping source patches")

# ============================================================================
# CONFIGURATION - Update these paths for your system
# ============================================================================

# Conda environment path (auto-detect)
CONDA_PREFIX = os.environ.get('CONDA_PREFIX', '')
SITE_PACKAGES = os.path.join(CONDA_PREFIX, 'Lib', 'site-packages') if CONDA_PREFIX else ''

# Insta360 SDK path
SDK_PATH = Path(r'C:\Users\Everton-PC\Documents\Windows_CameraSDK-2.1.1_MediaSDK-3.1.0\MediaSDK-3.1.0.0-20250904-win64\MediaSDK')

# FFmpeg path
FFMPEG_EXE = Path(r'C:\Users\Everton-PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe')
FFPROBE_EXE = FFMPEG_EXE.parent / 'ffprobe.exe'

# SphereSfM / COLMAP binaries (bundled in project)
SPHERE_SFM_DIR = Path('bin/SphereSfM')
COLMAP_DIR = Path('bin/COLMAP-3.11.1')

# Build configuration
BUILD_NAME = '360ToolkitGS'
BUILD_VERSION = '1.3.0'

# ============================================================================
# PyQt6 path
# ============================================================================
pyqt6_path = Path(SITE_PACKAGES) / 'PyQt6' if SITE_PACKAGES else None

block_cipher = None

print(f"\n{'='*70}")
print(f"Building {BUILD_NAME} v{BUILD_VERSION} - Universal GPU Edition")
print(f"{'='*70}")
print(f"Python: {sys.version}")
print(f"Conda: {CONDA_PREFIX}")
print(f"SDK: {SDK_PATH}")
print(f"FFmpeg: {FFMPEG_EXE}")
print(f"PyQt6: {pyqt6_path}")
print(f"Target GPUs: RTX 30xx (sm_86), 40xx (sm_89), 50xx (sm_120) - ALL NATIVE")
print(f"{'='*70}\n")

# ============================================================================
# DATA FILES
# ============================================================================
datas = []

# Insta360 SDK
if SDK_PATH.exists():
    sdk_bin = SDK_PATH / 'bin'
    sdk_models = SDK_PATH / 'models'
    if sdk_bin.exists():
        datas.append((str(sdk_bin), 'sdk/bin'))
        print(f"[OK] SDK bin: {sdk_bin}")
    if sdk_models.exists():
        datas.append((str(sdk_models), 'sdk/models'))
        print(f"[OK] SDK models: {sdk_models}")
else:
    print(f"[WARN] SDK not found at {SDK_PATH}")

# FFmpeg
if FFMPEG_EXE.exists():
    datas.append((str(FFMPEG_EXE), 'ffmpeg'))
    print(f"[OK] FFmpeg: {FFMPEG_EXE}")
    if FFPROBE_EXE.exists():
        datas.append((str(FFPROBE_EXE), 'ffmpeg'))
        print(f"[OK] FFprobe: {FFPROBE_EXE}")
else:
    print(f"[WARN] FFmpeg not found at {FFMPEG_EXE}")

# SphereSfM binaries
if SPHERE_SFM_DIR.exists():
    datas.append((str(SPHERE_SFM_DIR), 'bin/SphereSfM'))
    print(f"[OK] SphereSfM: {SPHERE_SFM_DIR}")

# COLMAP binaries
if COLMAP_DIR.exists():
    datas.append((str(COLMAP_DIR), 'bin/COLMAP-3.11.1'))
    print(f"[OK] COLMAP: {COLMAP_DIR}")

# YOLO models (bundle if present)
for model_file in ['yolov8m-seg.pt', 'yolov8m.pt', 'yolov8n-seg.pt', 'yolov8n-seg.onnx', 'yolov8s-seg.onnx', 'yolov8m-seg.onnx']:
    if os.path.exists(model_file):
        datas.append((model_file, '.'))
        print(f"[OK] Model: {model_file}")

# SAM model
if os.path.exists('sam_vit_b_01ec64.pth'):
    datas.append(('sam_vit_b_01ec64.pth', '.'))
    print(f"[OK] SAM model: sam_vit_b_01ec64.pth")

# PyQt6
if pyqt6_path and pyqt6_path.exists():
    datas.append((str(pyqt6_path), 'PyQt6'))
    print(f"[OK] PyQt6: {pyqt6_path}")

# ============================================================================
# BINARIES (DLLs)
# ============================================================================
binaries = []

# Collect ALL PyTorch DLLs from torch/lib as binaries (go to root _internal/)
# AND as datas to torch/lib (so torch's _load_dll_libraries() finds them)
# PyInstaller deduplicates binaries by filename, so we use datas for torch/lib
print("\nCollecting PyTorch DLLs...")
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        all_dlls = glob.glob(os.path.join(torch_lib, '*.dll'))
        for dll in all_dlls:
            binaries.append((dll, '.'))  # Root for general DLL search
        # CRITICAL: Also copy ALL DLLs to torch/lib as DATA entries
        # This prevents PyInstaller dedup from removing asmjit.dll, fbgemm.dll, etc.
        for dll in all_dlls:
            datas.append((dll, 'torch/lib'))
        print(f"  Total: {len(all_dlls)} DLLs from torch/lib (to root + torch/lib)")
    print(f"  PyTorch CUDA: {torch.version.cuda}")
except ImportError:
    print("[WARN] PyTorch not found")

# Conda Library/bin dependencies  
if CONDA_PREFIX:
    library_bin = os.path.join(CONDA_PREFIX, 'Library', 'bin')
    if os.path.exists(library_bin):
        needed_dlls = [
            'zlib.dll', 'libpng16.dll', 'jpeg62.dll', 'tiff.dll',
            'libwebp.dll', 'openblas.dll',
            # NOTE: Do NOT include libiomp5md.dll from conda - torch provides its
            # own LLVM version which has different exports. Using Intel's smaller
            # version causes WinError 127 when loading fbgemm.dll.
            # MKL / BLAS / LAPACK (needed by numpy)
            'libblas.dll', 'liblapack.dll', 'libcblas.dll',
            'mkl_rt.2.dll', 'mkl_core.2.dll', 'mkl_intel_thread.2.dll',
            'mkl_def.2.dll', 'mkl_avx2.2.dll', 'mkl_avx512.2.dll',
            'mkl_mc3.2.dll', 'mkl_sequential.2.dll',
            'mkl_vml_def.2.dll', 'mkl_vml_avx2.2.dll',
            'mkl_vml_avx512.2.dll', 'mkl_vml_cmpt.2.dll', 'mkl_vml_mc3.2.dll',
        ]
        for dll_name in needed_dlls:
            dll_path = os.path.join(library_bin, dll_name)
            if os.path.exists(dll_path):
                binaries.append((dll_path, '.'))
                # Also add as DATA to bypass PyInstaller dedup (liblapack.dll issue)
                datas.append((dll_path, '.'))
                print(f"  Conda DLL: {dll_name}")

# ============================================================================
# EXCLUDES (size optimization) 
# ============================================================================
excludes = [
    # PyQt5 (we use PyQt6)
    'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    
    # Unused Qt6 modules (save ~200MB)
    'PyQt6.Qt3DAnimation', 'PyQt6.Qt3DCore', 'PyQt6.Qt3DExtras',
    'PyQt6.Qt3DInput', 'PyQt6.Qt3DLogic', 'PyQt6.Qt3DRender',
    'PyQt6.QtBluetooth', 'PyQt6.QtCharts', 'PyQt6.QtDataVisualization',
    'PyQt6.QtDesigner', 'PyQt6.QtHelp', 'PyQt6.QtLocation',
    'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets', 'PyQt6.QtNfc',
    'PyQt6.QtOpenGL', 'PyQt6.QtOpenGLWidgets', 'PyQt6.QtPdf',
    'PyQt6.QtPdfWidgets', 'PyQt6.QtPositioning', 'PyQt6.QtPrintSupport',
    'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtQuick3D',
    'PyQt6.QtQuickWidgets', 'PyQt6.QtRemoteObjects', 'PyQt6.QtSensors',
    'PyQt6.QtSerialPort', 'PyQt6.QtSpatialAudio', 'PyQt6.QtSql',
    'PyQt6.QtStateMachine', 'PyQt6.QtSvg', 'PyQt6.QtSvgWidgets',
    'PyQt6.QtTest', 'PyQt6.QtTextToSpeech', 'PyQt6.QtWebChannel',
    'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineQuick',
    'PyQt6.QtWebEngineWidgets', 'PyQt6.QtWebSockets', 'PyQt6.QtXml',
    
    # Heavy libs not needed at runtime
    'IPython', 'jupyter', 'notebook', 'matplotlib',
    'pandas', 'sklearn', 'seaborn',
    
    # Testing/debugging
    'pytest', 'pydoc',
    
    # Development tools
    'black', 'isort', 'flake8', 'mypy',
    
    # NOTE: torch.distributed is NOT excluded. It is INCLUDED in the PYZ archive
    # but its __init__.py is PYZ-patched to a 5-line stub (is_available()→False).
    # This way `from torch import distributed` succeeds naturally through
    # FrozenImporter, torch.distributed becomes an attribute on torch, and
    # no circular imports occur because the stub doesn't import torch.
]

# ============================================================================
# HIDDEN IMPORTS
# ============================================================================
hiddenimports = [
    # Core
    'numpy', 'numpy.core', 'numpy.core._multiarray_umath',
    'numpy.linalg', 'numpy.linalg._umath_linalg',
    'numpy.fft', 'numpy.random', 'cv2', 'PIL', 'piexif',
    
    # Stdlib modules needed at runtime
    'readline',
    
    # PyQt6
    'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.sip',
    
    # PyTorch (for masking GPU support)
    'torch', 'torch.cuda', 'torch.nn', 'torch.nn.functional',
    'torchvision', 'torchvision.ops',
    
    # Ultralytics YOLOv8
    'ultralytics', 'ultralytics.nn', 'ultralytics.utils',
    'ultralytics.models', 'ultralytics.engine',
    
    # SAM (Segment Anything)
    'segment_anything',
    
    # SfM/COLMAP
    'pycolmap', 'scipy', 'scipy.spatial', 'scipy.spatial.transform',
    
    # Our modules
    'src', 'src.main', 'src.ui', 'src.ui.main_window',
    'src.pipeline', 'src.pipeline.batch_orchestrator',
    'src.extraction', 'src.transforms', 'src.masking',
    'src.config', 'src.config.defaults', 'src.config.settings',
    'src.premium', 'src.premium.sphere_sfm_integration',
    'src.premium.pose_transfer_integration',
    'src.utils',
]

# Add all PyQt6 submodules available 
try:
    hiddenimports += collect_submodules('PyQt6')
except Exception:
    pass

# Add ultralytics submodules
try:
    hiddenimports += collect_submodules('ultralytics')
except Exception:
    pass

# CRITICAL: Collect ALL torch submodules so PyInstaller's FrozenImporter
# can resolve torch's own internal imports (from torch import nn, etc.)
# Without this, torch's __init__.py fails with circular import errors.
# torch.distributed is INCLUDED — its __init__.py is PYZ-patched to a
# minimal stub so the import succeeds without circular imports.
try:
    torch_submodules = collect_submodules('torch')
    hiddenimports += torch_submodules
    print(f"[OK] torch: {len(torch_submodules)} submodules collected")
except Exception as e:
    print(f"[WARN] torch submodules: {e}")

# Collect torch data files for CUDA support
try:
    torch_datas = collect_data_files('torch')
    # Filter out torch/distributed/ data files.
    # These create an on-disk directory that acts as a namespace package,
    # conflicting with PYZ archive imports. The post-build patches handle
    # all references to torch.distributed in torch's source code.
    pre_filter = len(torch_datas)
    torch_datas = [(src, dst) for src, dst in torch_datas 
                   if not (dst.startswith('torch\\distributed\\') or
                           dst.startswith('torch/distributed/') or
                           dst == 'torch\\distributed' or
                           dst == 'torch/distributed' or
                           # Also catch nested paths like torch\distributed\optim
                           '\\distributed\\' in dst.replace('/', '\\'))
                   or ('include\\' in dst.replace('/', '\\') or 
                       'csrc\\' in dst.replace('/', '\\') or 
                       'testing\\' in dst.replace('/', '\\'))]
    print(f"[OK] torch data files: {len(torch_datas)} (filtered {pre_filter - len(torch_datas)} distributed)")
    datas += torch_datas
except Exception:
    pass

# Collect numpy binaries (MKL/BLAS dependencies)
try:
    np_datas, np_bins = collect_all('numpy')[:2]
    datas += np_datas
    binaries += np_bins
    print(f"[OK] numpy: {len(np_bins)} binaries collected")
except Exception as e:
    print(f"[WARN] numpy collect: {e}")

# Collect cv2 (OpenCV) binaries
try:
    cv2_datas, cv2_bins = collect_all('cv2')[:2]
    datas += cv2_datas
    binaries += cv2_bins
    print(f"[OK] cv2: {len(cv2_bins)} binaries collected")
except Exception as e:
    print(f"[WARN] cv2 collect: {e}")

# ============================================================================
# ANALYSIS
# ============================================================================
a = Analysis(
    ['src/main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=['runtime_hook_fullgpu.py'],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================================
# FILTER DUPLICATE / CONFLICTING DLLs
# ============================================================================
print("\nFiltering duplicate binaries...")
original_count = len(a.binaries)

# Track seen DLLs to prevent duplicates
seen_dlls = set()
filtered_binaries = []

for entry in a.binaries:
    name_lower = entry[0].lower()
    
    # Skip known problematic duplicates
    if 'opencv_cuda' in name_lower:
        continue
    
    # Deduplicate by filename
    base_name = os.path.basename(name_lower)
    if base_name in seen_dlls:
        continue
    seen_dlls.add(base_name)
    filtered_binaries.append(entry)

a.binaries = filtered_binaries
print(f"Filtered: {original_count} -> {len(a.binaries)} binaries")

# ============================================================================
# PACKAGE
# ============================================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=BUILD_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Don't UPX CUDA DLLs - causes crashes
    console=True,  # Keep console for log output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # Don't UPX - CUDA DLLs crash when compressed
    upx_exclude=[],
    name=BUILD_NAME,
)

# ============================================================================
# POST-BUILD: Fix DLL placement (PyInstaller deduplicates by filename)
# Some critical DLLs end up in wrong locations. Copy them to where they're needed.
# ============================================================================
import shutil

dist_internal = os.path.join('dist', BUILD_NAME, '_internal')
torch_lib_dest = os.path.join(dist_internal, 'torch', 'lib')

print(f"\n{'='*70}")
print("Post-build: Fixing DLL placement...")

# DLLs that must be in torch/lib/ for torch's _load_dll_libraries()
# CRITICAL: Some DLLs like libiomp5md.dll exist in both conda Library/bin and
# torch/lib/ but are DIFFERENT versions. torch needs its own larger LLVM version.
torch_lib_fixes = {}
try:
    import torch as _torch
    _src_torch_lib = os.path.join(os.path.dirname(_torch.__file__), 'lib')
    # DLLs missing from torch/lib due to dedup AND version-mismatched DLLs
    fix_dlls = ['cudart64_12.dll', 'libiomp5md.dll', 'nvrtc64_120_0.dll',
                'nvrtc-builtins64_128.dll', 'nvJitLink_120_0.dll',
                'caffe2_nvrtc.dll', 'cublas64_12.dll', 'cublasLt64_12.dll']
    for dll_name in fix_dlls:
        src = os.path.join(_src_torch_lib, dll_name)
        if os.path.exists(src):
            torch_lib_fixes[dll_name] = src
except ImportError:
    pass

for dll_name, src_path in torch_lib_fixes.items():
    dst = os.path.join(torch_lib_dest, dll_name)
    # Always overwrite - the deduped version might be the wrong one (e.g. libiomp5md.dll)
    shutil.copy2(src_path, dst)
    print(f"  [FIX] Copied {dll_name} -> torch/lib/ ({os.path.getsize(src_path):,} bytes)")

# DLLs that must be in root _internal/ for numpy BLAS
if CONDA_PREFIX:
    library_bin = os.path.join(CONDA_PREFIX, 'Library', 'bin')
    root_fixes = ['liblapack.dll', 'libblas.dll', 'libcblas.dll']
    for dll_name in root_fixes:
        src = os.path.join(library_bin, dll_name)
        dst = os.path.join(dist_internal, dll_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  [FIX] Copied {dll_name} -> _internal/")

print("Post-build DLL fixes complete.")

# ============================================================================
# POST-BUILD: Fix SDK DLL placement
# PyInstaller deduplicates common DLLs (msvcp140.dll, vcruntime140.dll, etc.)
# and removes them from sdk/bin/ because they already exist in _internal/.
# The MediaSDK subprocess needs them IN its own bin/ directory.
# Also restore OpenCV CUDA DLLs that PyInstaller may skip.
# ============================================================================
sdk_bin_dist = os.path.join(dist_internal, 'sdk', 'bin')
if os.path.isdir(sdk_bin_dist) and SDK_PATH.exists():
    sdk_bin_orig = str(SDK_PATH / 'bin')
    if os.path.isdir(sdk_bin_orig):
        restored_count = 0
        for dll_file in os.listdir(sdk_bin_orig):
            if dll_file.lower().endswith('.dll'):
                src = os.path.join(sdk_bin_orig, dll_file)
                dst = os.path.join(sdk_bin_dist, dll_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  [FIX] Restored SDK DLL: {dll_file}")
                    restored_count += 1
        if restored_count > 0:
            print(f"  [FIX] Restored {restored_count} missing SDK DLLs (dedup fix)")
        else:
            print(f"  [OK] All SDK DLLs present")

# ============================================================================
# POST-BUILD: Clean up torch/distributed/ directory
#
# collect_data_files may create on-disk torch/distributed/ data files
# (.pyi stubs, .json, etc.). This directory acts as a namespace package
# that can shadow the PYZ modules. Delete it to ensure FrozenImporter
# serves our patched distributed stub from the PYZ archive.
# ============================================================================
torch_dist_dir = os.path.join(dist_internal, 'torch', 'distributed')
if os.path.isdir(torch_dist_dir):
    shutil.rmtree(torch_dist_dir)
    print(f"  [CLEAN] Removed torch/distributed/ directory (prevents namespace package shadowing)")

print(f"\n{'='*70}")
print(f"Build spec created for {BUILD_NAME} v{BUILD_VERSION}")
print(f"GPU Support: RTX 30xx (sm_86), 40xx (sm_89), 50xx (sm_120)")
print(f"CUDA 12.8: Full native GPU on all RTX generations")
print(f"{'='*70}")

# ============================================================================
# POST-BUILD: Restore patched torch source files
# ============================================================================
print("\n[POST-BUILD] Restoring torch source files...")
_restore_all()
print("  Torch sources restored to original state.\n")
