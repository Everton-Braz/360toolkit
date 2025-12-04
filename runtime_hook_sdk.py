"""
PyInstaller Runtime Hook for SDK Path Detection
Fixes SDK path issue when running frozen app on different computers

Ensures SDK is loaded from _internal folder, not hardcoded dev machine path
Also sets up CUDA/GPU environment for MediaSDK
"""
import os
import sys
from pathlib import Path

if hasattr(sys, '_MEIPASS'):
    # Running as frozen PyInstaller app
    base_path = Path(sys._MEIPASS)
    
    # Look for SDK in _internal folder (bundled location)
    sdk_locations = [
        base_path / 'sdk' / 'MediaSDK-3.0.5-20250619-win64',
        base_path / '_internal' / 'sdk' / 'MediaSDK-3.0.5-20250619-win64',
        base_path / 'sdk',
        base_path / '_internal' / 'sdk',
    ]
    
    sdk_found = None
    for sdk_path in sdk_locations:
        if sdk_path.exists():
            sdk_found = sdk_path
            break
    
    if sdk_found:
        # Override SDK path for the application
        os.environ['INSTA360_SDK_PATH'] = str(sdk_found)
        print(f"[SDK Hook] Found bundled SDK at: {sdk_found}")
        
        # CRITICAL: Add SDK bin to PATH for DLL loading
        sdk_bin = sdk_found / 'bin'
        if sdk_bin.exists():
            current_path = os.environ.get('PATH', '')
            os.environ['PATH'] = str(sdk_bin) + os.pathsep + current_path
            print(f"[SDK Hook] Added SDK bin to PATH: {sdk_bin}")
            
            # Add SDK bin to DLL search path (Windows)
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(str(sdk_bin))
                    print(f"[SDK Hook] Added SDK bin to DLL search path")
                except Exception as e:
                    print(f"[SDK Hook] Failed to add DLL directory: {e}")
    else:
        print(f"[SDK Hook] WARNING: SDK not found in bundle")
        print(f"[SDK Hook] Searched locations:")
        for loc in sdk_locations:
            print(f"  - {loc}")

# CRITICAL FIX: Ensure System32 is in PATH for CUDA drivers (nvcuda.dll)
# PyInstaller sometimes strips System32 from PATH, breaking GPU detection
system_root = os.environ.get('SystemRoot', r'C:\Windows')
system32 = Path(system_root) / 'System32'
syswow64 = Path(system_root) / 'SysWOW64'

if system32.exists():
    current_path = os.environ.get('PATH', '')
    if str(system32) not in current_path:
        os.environ['PATH'] = str(system32) + os.pathsep + current_path
        print(f"[SDK Hook] Added System32 to PATH for CUDA drivers: {system32}")
        
        # Check for CUDA driver
        nvcuda_dll = system32 / 'nvcuda.dll'
        if nvcuda_dll.exists():
            print(f"[SDK Hook] CUDA driver found: {nvcuda_dll}")
        else:
            print(f"[SDK Hook] WARNING: CUDA driver (nvcuda.dll) not found in System32")
            print(f"[SDK Hook] GPU stitching will NOT work - SDK will report 'no device found'")
            print(f"[SDK Hook] Please ensure NVIDIA GPU drivers are installed")

# Also add SysWOW64 for 32-bit compatibility (though SDK is 64-bit)
if syswow64.exists():
    current_path = os.environ.get('PATH', '')
    if str(syswow64) not in current_path:
        os.environ['PATH'] = str(syswow64) + os.pathsep + current_path
