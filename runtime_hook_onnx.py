"""
Runtime hook for ONNX Runtime DLL loading
Ensures onnxruntime can find its DLLs in the bundled application
"""
import os
import sys
from pathlib import Path

def setup_onnx_dlls():
    """Add ONNX Runtime DLL paths to system PATH"""
    if getattr(sys, 'frozen', False):
        # Running as bundled executable
        exe_dir = Path(sys.executable).parent
        internal_dir = exe_dir / '_internal'
        
        # Possible ONNX DLL locations
        onnx_paths = [
            internal_dir / 'onnxruntime' / 'capi',
            internal_dir / 'onnxruntime.libs',  # pip install location
            internal_dir,
            exe_dir,
        ]
        
        # Add all existing paths to system PATH
        for path in onnx_paths:
            if path.exists():
                path_str = str(path)
                if path_str not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = path_str + os.pathsep + os.environ.get('PATH', '')
                    
                # Also add to DLL search path (Windows)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(path_str)
                    except:
                        pass

    # Pre-load onnxruntime.dll to help resolution
    try:
        import ctypes
        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).parent
            # Try root
            dll_path = base_dir / 'onnxruntime.dll'
            if dll_path.exists():
                ctypes.CDLL(str(dll_path))
            
            # Try _internal/onnxruntime/capi
            dll_path_2 = base_dir / '_internal' / 'onnxruntime' / 'capi' / 'onnxruntime.dll'
            if dll_path_2.exists():
                ctypes.CDLL(str(dll_path_2))
    except Exception as e:
        pass

# Run setup immediately
setup_onnx_dlls()
