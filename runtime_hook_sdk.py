"""
PyInstaller Runtime Hook for SDK Path Detection
Fixes SDK path issue when running frozen app on different computers

Ensures SDK is loaded from _internal folder, not hardcoded dev machine path
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
    else:
        print(f"[SDK Hook] WARNING: SDK not found in bundle")
        print(f"[SDK Hook] Searched locations:")
        for loc in sdk_locations:
            print(f"  - {loc}")
