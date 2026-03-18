"""
Runtime hook for 360ToolkitGS.
Sets up paths for ONNX Runtime CUDA, SDK, FFmpeg, and optional PyTorch assets.
"""

import os
import sys
import glob

# Fix OpenMP DLL conflict (libomp.dll vs libiomp5md.dll)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

_DLL_DIR_HANDLES = []


def _remember_dll_directory(path: str) -> None:
    """Keep add_dll_directory handles alive for the lifetime of the process."""
    if not hasattr(os, 'add_dll_directory'):
        return
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
    except (OSError, FileNotFoundError):
        pass


def _has_full_torch_runtime(base_path: str) -> bool:
    """Return True only when a real torch package is bundled, not just torch/lib."""
    torch_dir = os.path.join(base_path, 'torch')
    if not os.path.isdir(torch_dir):
        return False

    package_markers = [
        os.path.join(torch_dir, '__init__.py'),
        os.path.join(torch_dir, '__init__.pyc'),
        os.path.join(torch_dir, 'version.py'),
        os.path.join(torch_dir, 'version.pyc'),
    ]
    if any(os.path.exists(marker) for marker in package_markers):
        return True

    return bool(glob.glob(os.path.join(torch_dir, '_C*.pyd')))

# ==============================================================================
# CRITICAL: Preload MKL/BLAS DLLs for numpy in frozen app (Python 3.10+ Windows)
# numpy._umath_linalg needs MKL DLLs, but frozen apps restrict DLL search paths.
# We must add DLL directories AND preload key DLLs before numpy loads.
# ==============================================================================
if getattr(sys, 'frozen', False) and sys.platform == 'win32':
    _base = sys._MEIPASS
    _has_torch_runtime = _has_full_torch_runtime(_base)
    # Add _internal to DLL search path
    if hasattr(os, 'add_dll_directory'):
        _remember_dll_directory(_base)
        _numpy_libs = os.path.join(_base, 'numpy.libs')
        if os.path.isdir(_numpy_libs):
            _remember_dll_directory(_numpy_libs)
        # Also add torch/lib subdir
        _torch_lib = os.path.join(_base, 'torch', 'lib')
        if _has_torch_runtime and os.path.isdir(_torch_lib):
            _remember_dll_directory(_torch_lib)
    
    # Preload MKL DLLs in correct order via ctypes
    import ctypes
    _mkl_dlls = [
        'mkl_core.2.dll',
        'mkl_intel_thread.2.dll',
        'mkl_rt.2.dll',
        'mkl_def.2.dll',
        'mkl_avx2.2.dll',
        # NOTE: Do NOT load libiomp5md.dll here - torch provides its own LLVM
        # version in torch/lib/ which has different exports than Intel's
        'libblas.dll',
        'liblapack.dll',
    ]
    for _dll in _mkl_dlls:
        _dll_path = os.path.join(_base, _dll)
        if os.path.exists(_dll_path):
            try:
                ctypes.CDLL(_dll_path)
            except OSError:
                pass
    
    # ======================================================================
    # CRITICAL: Preload PyTorch DLLs in dependency order BEFORE any torch import.
    # In PyInstaller frozen apps, torch's own DLL loading code may fail because
    # the search paths are different from a normal Python installation.
    # By loading these DLLs into the process early, they'll be found when
    # torch._C.pyd tries to resolve its dependencies.
    # ======================================================================
    _torch_lib = os.path.join(_base, 'torch', 'lib')
    if _has_torch_runtime and os.path.isdir(_torch_lib):
        # Load in dependency order: base deps first, then torch libs
        _torch_dlls = [
            # CUDA runtime (no torch dependency, pure NVIDIA)
            'cudart64_12.dll',
            'nvToolsExt64_1.dll',
            # cuDNN
            'cudnn64_9.dll',
            # cuBLAS
            'cublasLt64_12.dll',
            'cublas64_12.dll',
            # NVRTC
            'nvrtc64_120_0.dll',
            'nvJitLink_120_0.dll',
            'nvrtc-builtins64_128.dll',
            # Intel OpenMP (torch version)
            'libiomp5md.dll',
            # Torch core (dependency chain: c10 -> torch_cpu -> c10_cuda -> torch_cuda)
            'c10.dll',
            'torch_cpu.dll',
            'c10_cuda.dll',
            'torch_cuda.dll',
            # caffe2 NVRTC bridge
            'caffe2_nvrtc.dll',
        ]
        _loaded = 0
        for _dll in _torch_dlls:
            _dll_path = os.path.join(_torch_lib, _dll)
            if os.path.exists(_dll_path):
                try:
                    ctypes.CDLL(_dll_path)
                    _loaded += 1
                except OSError:
                    pass  # Non-critical: some DLLs may fail but torch might still work

    # ======================================================================
    # CRITICAL: Preload ONNX Runtime + CUDA dependency DLLs via ctypes.
    # Same pattern as MKL/torch above: force DLLs into the process module
    # table so that onnxruntime_pybind11_state.pyd's implicit deps resolve.
    # ======================================================================
    _ort_capi = os.path.join(_base, 'onnxruntime', 'capi')
    if os.path.isdir(_ort_capi):
        _onnx_cuda_dlls = [
            # CUDA runtime deps from _internal root
            (_base, 'cudart64_12.dll'),
            (_base, 'zlibwapi.dll'),
            (_base, 'cublasLt64_12.dll'),
            (_base, 'cublas64_12.dll'),
            (_base, 'cufft64_11.dll'),
            (_base, 'cudnn64_9.dll'),
            (_base, 'cudnn_ops64_9.dll'),
            (_base, 'cudnn_cnn64_9.dll'),
            (_base, 'cudnn_adv64_9.dll'),
            (_base, 'cudnn_graph64_9.dll'),
            (_base, 'cudnn_heuristic64_9.dll'),
            (_base, 'cudnn_engines_precompiled64_9.dll'),
            (_base, 'cudnn_engines_runtime_compiled64_9.dll'),
            # ONNX Runtime core from capi/
            (_ort_capi, 'onnxruntime.dll'),
            (_ort_capi, 'onnxruntime_providers_shared.dll'),
            (_ort_capi, 'onnxruntime_providers_cuda.dll'),
            (_ort_capi, 'onnxruntime_providers_tensorrt.dll'),
        ]
        _ort_loaded = 0
        for _dir, _dll in _onnx_cuda_dlls:
            _dll_path = os.path.join(_dir, _dll)
            if os.path.exists(_dll_path):
                try:
                    ctypes.CDLL(_dll_path)
                    _ort_loaded += 1
                except OSError:
                    pass
        if _ort_loaded:
            print(f"[OK] Preloaded {_ort_loaded} CUDA/ONNX DLLs via ctypes")

# ==============================================================================
# torch.distributed: INCLUDED in PYZ archive but PYZ-patched to a minimal stub.
# The stub's is_available() returns False, so no distributed code runs.
# No runtime stubs, no meta_path finders, no sys.modules hacks needed.
# The patched __init__.py is compiled into PYZ by the spec file's a.pure
# patching step, so FrozenImporter loads the stub directly.
# ==============================================================================
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

def setup_bundled_paths():
    """Configure paths for bundled executables and DLLs."""
    
    # Determine base path (PyInstaller sets _MEIPASS when bundled)
    if getattr(sys, 'frozen', False):
        # Running as bundled executable
        base_path = sys._MEIPASS
        app_path = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = base_path
    
    # Key directories
    internal_path = base_path
    sdk_bin_path = os.path.join(base_path, 'sdk', 'bin')
    ffmpeg_path = os.path.join(base_path, 'ffmpeg')
    onnx_capi_path = os.path.join(base_path, 'onnxruntime', 'capi')
    torch_lib_path = os.path.join(base_path, 'torch', 'lib')
    has_torch_runtime = _has_full_torch_runtime(base_path)
    numpy_libs_path = os.path.join(base_path, 'numpy.libs')
    numpy_core_path = os.path.join(base_path, 'numpy', 'core')
    numpy_linalg_path = os.path.join(base_path, 'numpy', 'linalg')
    
    # Build PATH with all DLL locations
    dll_paths = [
        internal_path,          # Main binaries + MKL DLLs
        sdk_bin_path,           # Insta360 SDK
        ffmpeg_path,            # FFmpeg
        onnx_capi_path,         # ONNX Runtime CUDA
        numpy_libs_path,        # NumPy/OpenBLAS runtime DLLs
        numpy_core_path,        # NumPy core
        numpy_linalg_path,      # NumPy linalg
    ]
    if has_torch_runtime:
        dll_paths.insert(1, torch_lib_path)
    
    # Add CUDA from system if available
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
    ]
    
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            dll_paths.append(cuda_path)
            break
    
    # Prepend to PATH
    existing_path = os.environ.get('PATH', '')
    new_paths = [p for p in dll_paths if os.path.exists(p)]
    os.environ['PATH'] = os.pathsep.join(new_paths) + os.pathsep + existing_path
    
    # Windows: Add DLL directories explicitly (Python 3.8+)
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        for p in new_paths:
            _remember_dll_directory(p)
    
    return base_path, app_path

def setup_torch_env(base_path: str):
    """Set environment variables for PyTorch CUDA performance when bundled.
    
    IMPORTANT: Do NOT import torch here! In PyInstaller frozen apps, importing
    torch in a runtime hook can cause C-extension double-initialization errors
    ('function already has a docstring', 'partially initialized module').
    
    Instead, we set environment variables that torch will pick up when it's
    actually imported by the application code (in _detect_gpu or _test_torch_cuda).
    """
    torch_lib_path = os.path.join(base_path, 'torch', 'lib')
    if not (_has_full_torch_runtime(base_path) and os.path.isdir(torch_lib_path)):
        os.environ['_360TK_RUNTIME_HOOK'] = '1'
        print("[OK] ONNX-only runtime detected; skipping PyTorch environment setup")
        return False

    # Set memory allocation strategy for CUDA
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Allow TF32 via env var (torch will also set programmatically on import)
    os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
    
    # Signal that the hook ran (app can check this)
    os.environ['_360TK_RUNTIME_HOOK'] = '1'
    
    print("[OK] PyTorch environment configured (import deferred to app startup)")
    return True

def setup_onnx_cuda():
    """Defer ONNX Runtime probing until normal application startup."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    ort_capi_path = os.path.join(base_path, 'onnxruntime', 'capi')
    if os.path.isdir(ort_capi_path):
        print("[OK] ONNX Runtime packaged (probe deferred to app startup)")
        return True

    print("[WARN] ONNX Runtime packaged files not found")
    return False

def setup_sdk_path():
    """Set SDK environment variable."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        sdk_path = os.path.join(base_path, 'sdk')
        if os.path.exists(sdk_path):
            os.environ['INSTA360_SDK_PATH'] = sdk_path
            return True
    return False

# Execute setup when this hook runs
try:
    base_path, app_path = setup_bundled_paths()
    setup_sdk_path()
    setup_torch_env(base_path)
    setup_onnx_cuda()
    
    print("[OK] Runtime hook complete - DLL paths configured")
        
except Exception as e:
    print(f"[WARN] Runtime hook error: {e}")
