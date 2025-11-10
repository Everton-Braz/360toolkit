"""
PyInstaller Runtime Hook for PyTorch DLL Loading
Solves: OSError [WinError 1114] DLL initialization routine failed

This hook runs BEFORE your application starts and fixes PyTorch DLL paths.
Based on community solutions from GitHub issues #56362, #8348, #8211, #4085
"""
import os
import sys
import ctypes

print("[PyTorch Hook] Starting runtime hook execution...")

# Check if running in PyInstaller frozen mode
if hasattr(sys, '_MEIPASS'):
    # _MEIPASS is the temp folder where PyInstaller extracts files
    base_path = sys._MEIPASS
    print(f"[PyTorch Hook] Base path: {base_path}")
    
    # CRITICAL FIX #1: Pre-load dependency DLLs in correct order
    # Windows DLL loading is order-dependent - load dependencies FIRST
    torch_lib = os.path.join(base_path, '_internal', 'torch', 'lib')
    if not os.path.exists(torch_lib):
        torch_lib = os.path.join(base_path, 'torch', 'lib')
    
    if os.path.exists(torch_lib):
        print(f"[PyTorch Hook] Torch lib found: {torch_lib}")
        
        # Pre-load critical DLLs in dependency order
        # These must load BEFORE c10.dll
        critical_dlls = [
            # MSVC Runtime (REQUIRED - from your research)
            'msvcp140.dll',
            'vcruntime140.dll', 
            'vcruntime140_1.dll',
            # Intel MKL/OpenMP
            'libiomp5md.dll',
            # CUDA Runtime (if GPU build)
            'cudart64_110.dll',
            'cudart64_12.dll',
        ]
        
        for dll_name in critical_dlls:
            dll_path = os.path.join(torch_lib, dll_name)
            if os.path.exists(dll_path):
                try:
                    ctypes.CDLL(dll_path)
                    print(f"[PyTorch Hook] Pre-loaded: {dll_name}")
                except Exception as e:
                    print(f"[PyTorch Hook] WARNING: Could not pre-load {dll_name}: {e}")
    
    # CRITICAL FIX #2: Add PyTorch DLL directories to PATH
    torch_dll_paths = [
        os.path.join(base_path, '_internal', 'torch', 'lib'),
        os.path.join(base_path, 'torch', 'lib'),
        os.path.join(base_path, '_internal', 'torch', 'bin'),
        os.path.join(base_path, 'torch', 'bin'),
        base_path,  # Also add root (for MSVC DLLs)
        os.path.join(base_path, '_internal'),
    ]
    
    # Add all existing paths to system PATH BEFORE torch import
    existing_path = os.environ.get('PATH', '')
    new_paths = [p for p in torch_dll_paths if os.path.exists(p)]
    
    if new_paths:
        # Prepend torch paths to ensure they're found FIRST
        os.environ['PATH'] = os.pathsep.join(new_paths) + os.pathsep + existing_path
        print(f"[PyTorch Hook] Added {len(new_paths)} paths to PATH")
        for p in new_paths:
            print(f"  - {p}")
    
    # Set torch home to prevent lookups outside bundle
    os.environ['TORCH_HOME'] = os.path.join(base_path, 'torch')
    os.environ['KMP_WARNINGS'] = '0'
    
    # CRITICAL FIX #3: Add Python modules to sys.path for imports
    # PyInstaller extracts modules to _internal, but they might not be in sys.path
    python_paths_to_add = [
        base_path,
        os.path.join(base_path, '_internal'),
    ]
    
    for py_path in python_paths_to_add:
        if py_path not in sys.path and os.path.exists(py_path):
            sys.path.insert(0, py_path)  # Insert at beginning for priority
            print(f"[PyTorch Hook] Added to sys.path: {py_path}")
    
    # Verify torch module is importable
    torch_module_path = os.path.join(base_path, '_internal', 'torch')
    if not os.path.exists(torch_module_path):
        torch_module_path = os.path.join(base_path, 'torch')
    
    if os.path.exists(torch_module_path):
        print(f"[PyTorch Hook] Torch module location: {torch_module_path}")
        if os.path.exists(os.path.join(torch_module_path, '__init__.py')):
            print("[PyTorch Hook] Torch __init__.py found - module should be importable")
        else:
            print("[PyTorch Hook] WARNING: Torch __init__.py NOT found!")
    
    print("[PyTorch Hook] Runtime environment configured successfully!")
else:
    print("[PyTorch Hook] Not running in frozen mode, skipping hook")
