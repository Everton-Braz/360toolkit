#!/usr/bin/env python3
"""Monitor and start the PyInstaller build."""

import subprocess
import sys
import time
import os
from pathlib import Path

BUILD_DIR = Path(r"C:\Users\User\Documents\APLICATIVOS\360ToolKit")
os.chdir(BUILD_DIR)

print("=" * 60)
print("360FrameTools PyInstaller Build Monitor")
print("=" * 60)
print(f"Working directory: {BUILD_DIR}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Clean old artifacts
print("[1/4] Cleaning old artifacts...")
try:
    import shutil
    if (BUILD_DIR / "build").exists():
        shutil.rmtree(BUILD_DIR / "build", ignore_errors=True)
    if (BUILD_DIR / "dist").exists():
        shutil.rmtree(BUILD_DIR / "dist", ignore_errors=True)
    print("  ✓ Cleaned")
except Exception as e:
    print(f"  Warning: {e}")

print()
print("[2/4] Running PyInstaller...")
print(f"  Command: pyinstaller 360FrameTools_MINIMAL.spec --clean --onedir")
print(f"  (This takes 15-30 minutes, please be patient...)")
print()

try:
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", 
         "360FrameTools_MINIMAL.spec", "--clean", "--onedir"],
        cwd=BUILD_DIR,
        capture_output=False,  # Show live output
        timeout=3600  # 1 hour timeout
    )
    
    if result.returncode == 0:
        print()
        print("[3/4] Build completed successfully!")
        print()
        
        dist_path = BUILD_DIR / "dist"
        if dist_path.exists():
            print("[4/4] Verifying dist folder...")
            contents = list(dist_path.iterdir())
            for item in contents:
                size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024*1024) if item.is_dir() else item.stat().st_size / (1024*1024)
                print(f"  - {item.name:<30} ({size_mb:>10.1f} MB)")
            
            exe_path = dist_path / "360ToolkitGS-FULL" / "360FrameTools.exe"
            if exe_path.exists():
                print()
                print(f"✓ Executable ready: {exe_path}")
                print(f"  File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            else:
                print()
                print("WARNING: 360FrameTools.exe not found in dist folder")
        else:
            print("ERROR: dist folder not created")
    else:
        print(f"ERROR: Build failed with exit code {result.returncode}")
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("ERROR: Build timed out after 1 hour")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print()
print("=" * 60)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
