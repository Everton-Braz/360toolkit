"""
Runtime hook for 360ToolkitGS - Full GPU Version
Sets up paths for PyTorch CUDA, ONNX Runtime CUDA, SDK, and FFmpeg
"""

import os
import sys
import types
import builtins

# ==============================================================================
# CRITICAL: Handle torch.distributed imports in frozen PyInstaller apps
# PyTorch 2.11+ accesses torch.distributed.rpc during its own initialization
# This causes circular import issues in frozen apps where distributed is missing
# ==============================================================================

# Pre-create stub modules in sys.modules
def _create_distributed_stubs():
    """Create fake torch.distributed.* modules before torch imports."""
    
    # Base distributed module
    distributed = types.ModuleType('torch.distributed')
    distributed.__file__ = '<stub:torch.distributed>'
    distributed.__path__ = []
    distributed.__package__ = 'torch.distributed'
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    distributed.is_mpi_available = lambda: False
    distributed.is_nccl_available = lambda: False
    distributed.is_gloo_available = lambda: False
    distributed.get_world_size = lambda *args, **kwargs: 1
    distributed.get_rank = lambda *args, **kwargs: 0
    distributed.barrier = lambda *args, **kwargs: None
    
    # RPC submodule (torch._jit_internal accesses this)
    rpc = types.ModuleType('torch.distributed.rpc')
    rpc.__file__ = '<stub:torch.distributed.rpc>'
    rpc.__path__ = []
    rpc.is_available = lambda: False
    distributed.rpc = rpc
    
    # Register in sys.modules
    sys.modules['torch.distributed'] = distributed
    sys.modules['torch.distributed.rpc'] = rpc
    
    # Pre-create all common submodules
    submodules = [
        'c10d_logger', 'elastic', 'fsdp', 'launch', 'nn', 'optim',
        'pipeline', 'rendezvous', 'tensor', 'algorithms', 'autograd',
        'checkpoint', 'constants', 'distributed_c10d', 'utils',
        '_sharding_spec', '_sharded_tensor', '_shard',
        'elastic.multiprocessing', 'elastic.multiprocessing.redirects'
    ]
    
    for submod in submodules:
        fullname = f'torch.distributed.{submod}'
        if fullname not in sys.modules:
            mod = types.ModuleType(fullname)
            mod.__file__ = f'<stub:{fullname}>'
            mod.__path__ = []
            mod.is_available = lambda: False
            sys.modules[fullname] = mod
    
    return distributed

_fake_distributed = _create_distributed_stubs()

# Custom import hook to intercept any remaining torch.distributed.* imports
class TorchDistributedFinder:
    """Meta path finder that returns stubs for torch.distributed submodules."""
    
    def find_module(self, fullname, path=None):
        if fullname.startswith('torch.distributed.'):
            return TorchDistributedLoader()
        return None

class TorchDistributedLoader:
    """Loader that returns stub modules for torch.distributed submodules."""
    
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        
        mod = types.ModuleType(fullname)
        mod.__file__ = f'<stub:{fullname}>'
        mod.__path__ = []
        mod.__loader__ = self
        mod.is_available = lambda: False
        mod.is_initialized = lambda: False
        sys.modules[fullname] = mod
        return mod

# Install at the FRONT of meta_path
sys.meta_path.insert(0, TorchDistributedFinder())

# Patch builtins.__import__ to inject 'distributed' attr into torch module
_original_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    
    # After torch is imported, ensure it has 'distributed' attribute
    if name == 'torch' or (name.startswith('torch.') and 'torch' in sys.modules):
        torch_mod = sys.modules.get('torch')
        if torch_mod is not None and not hasattr(torch_mod, 'distributed'):
            torch_mod.distributed = _fake_distributed
    
    return result

builtins.__import__ = _patched_import

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
    
    # Build PATH with all DLL locations
    dll_paths = [
        internal_path,          # Main binaries + PyTorch CUDA DLLs
        sdk_bin_path,           # Insta360 SDK
        ffmpeg_path,            # FFmpeg
        onnx_capi_path,         # ONNX Runtime CUDA
    ]
    
    # Add CUDA from system if available
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin',
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
            try:
                os.add_dll_directory(p)
            except (OSError, FileNotFoundError):
                pass
    
    return base_path, app_path

def setup_torch_cuda():
    """Configure PyTorch for optimal GPU performance."""
    try:
        import torch
        
        if torch.cuda.is_available():
            # Enable TF32 for RTX 30/40/50 series
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"[GPU] PyTorch CUDA: {device_name} (CUDA {cuda_version})")
            return True
    except Exception as e:
        print(f"[WARN] PyTorch CUDA setup: {e}")
    
    return False

def setup_onnx_cuda():
    """Configure ONNX Runtime for GPU inference."""
    try:
        import onnxruntime as ort
        
        providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in providers:
            print(f"[GPU] ONNX Runtime CUDA available")
            return True
        elif 'TensorrtExecutionProvider' in providers:
            print(f"[GPU] ONNX Runtime TensorRT available")
            return True
        else:
            print(f"[CPU] ONNX Runtime providers: {providers}")
    except Exception as e:
        print(f"[WARN] ONNX Runtime setup: {e}")
    
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
    torch_gpu = setup_torch_cuda()
    onnx_gpu = setup_onnx_cuda()
    
    if torch_gpu and onnx_gpu:
        print("[OK] Full GPU acceleration enabled (PyTorch + ONNX)")
    elif torch_gpu:
        print("[OK] GPU acceleration: PyTorch only")
    elif onnx_gpu:
        print("[OK] GPU acceleration: ONNX only")
    else:
        print("[INFO] CPU mode (no GPU acceleration)")
        
except Exception as e:
    print(f"[WARN] Runtime hook error: {e}")
