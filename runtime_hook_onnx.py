"""
Runtime hook for 360ToolkitGS bundled application
Ensures all DLLs (ONNX Runtime, SDK, FFmpeg, CUDA) are properly loaded

This hook runs BEFORE any Python imports in the bundled app.
"""
import os
import sys
from pathlib import Path

def setup_bundled_paths():
    """
    Configure DLL search paths for the bundled application.
    Adds paths for: ONNX Runtime, Insta360 SDK, FFmpeg, CUDA
    """
    if not getattr(sys, 'frozen', False):
        return  # Not running as bundled executable
    
    # Base directories
    exe_dir = Path(sys.executable).parent
    internal_dir = exe_dir / '_internal'
    
    # All paths that might contain DLLs we need
    dll_paths = [
        # Root directory (where we spray critical DLLs)
        exe_dir,
        internal_dir,
        
        # ONNX Runtime
        internal_dir / 'onnxruntime' / 'capi',
        internal_dir / 'onnxruntime.libs',
        
        # Insta360 SDK
        exe_dir / 'sdk' / 'bin',
        internal_dir / 'sdk' / 'bin',
        
        # FFmpeg
        exe_dir / 'ffmpeg',
        internal_dir / 'ffmpeg',
    ]
    
    # Add all existing paths to system PATH
    current_path = os.environ.get('PATH', '')
    new_paths = []
    
    for path in dll_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in current_path:
                new_paths.append(path_str)
                
            # Windows 10+ DLL search path (more reliable than PATH)
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(path_str)
                except (OSError, RuntimeError):
                    pass  # Ignore errors (e.g., path already added)
    
    # Prepend new paths to PATH environment variable
    if new_paths:
        os.environ['PATH'] = os.pathsep.join(new_paths) + os.pathsep + current_path


def preload_critical_dlls():
    """
    Pre-load critical DLLs to ensure they're found before Python imports.
    This helps resolve DLL dependency chains.
    """
    if not getattr(sys, 'frozen', False):
        return
    
    import ctypes
    
    exe_dir = Path(sys.executable).parent
    internal_dir = exe_dir / '_internal'
    
    # DLLs to pre-load (in dependency order)
    critical_dlls = [
        # ONNX Runtime core
        'onnxruntime.dll',
        'onnxruntime_providers_shared.dll',
        
        # CUDA providers (optional, may not exist)
        'onnxruntime_providers_cuda.dll',
        'onnxruntime_providers_tensorrt.dll',
    ]
    
    # Search locations
    search_paths = [
        exe_dir,
        internal_dir,
        internal_dir / 'onnxruntime' / 'capi',
    ]
    
    for dll_name in critical_dlls:
        for search_path in search_paths:
            dll_path = search_path / dll_name
            if dll_path.exists():
                try:
                    ctypes.CDLL(str(dll_path))
                except OSError:
                    pass  # DLL load failed, but continue
                break  # Found and attempted to load, move to next DLL


def setup_sdk_path():
    """
    Set environment variable for Insta360 SDK model files.
    """
    if not getattr(sys, 'frozen', False):
        return
    
    exe_dir = Path(sys.executable).parent
    
    # SDK model file paths
    model_paths = [
        exe_dir / 'sdk' / 'modelfile',
        exe_dir / '_internal' / 'sdk' / 'modelfile',
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            os.environ['INSTA360_SDK_MODELPATH'] = str(model_path)
            break


# ============================================================================
# Run all setup functions immediately when this hook loads
# ============================================================================
setup_bundled_paths()
preload_critical_dlls()
setup_sdk_path()
