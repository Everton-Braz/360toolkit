#!/usr/bin/env python3
"""
Test script to verify 360toolkit build functionality.
Tests: SDK detection, FFmpeg detection, ONNX masking
Run this after building to ensure everything works.
"""

import os
import sys
import subprocess
from pathlib import Path

def get_app_path():
    """Get the application path (works for both dev and frozen)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        return Path(__file__).parent

def test_ffmpeg():
    """Test FFmpeg availability."""
    print("\n" + "="*60)
    print("TEST 1: FFmpeg Detection")
    print("="*60)
    
    app_path = get_app_path()
    
    # Check bundled location
    bundled_ffmpeg = app_path / 'ffmpeg' / 'ffmpeg.exe'
    bundled_ffprobe = app_path / 'ffmpeg' / 'ffprobe.exe'
    
    # Also check _internal for frozen app
    if not bundled_ffmpeg.exists():
        bundled_ffmpeg = app_path / '_internal' / 'ffmpeg' / 'ffmpeg.exe'
        bundled_ffprobe = app_path / '_internal' / 'ffmpeg' / 'ffprobe.exe'
    
    ffmpeg_found = False
    ffprobe_found = False
    
    if bundled_ffmpeg.exists():
        print(f"✅ FFmpeg found: {bundled_ffmpeg}")
        ffmpeg_found = True
        # Test execution
        try:
            result = subprocess.run([str(bundled_ffmpeg), '-version'], 
                                    capture_output=True, text=True, timeout=10)
            version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
            print(f"   Version: {version_line}")
        except Exception as e:
            print(f"   ⚠️ Could not get version: {e}")
    else:
        print(f"❌ FFmpeg not found at: {bundled_ffmpeg}")
    
    if bundled_ffprobe.exists():
        print(f"✅ FFprobe found: {bundled_ffprobe}")
        ffprobe_found = True
    else:
        print(f"❌ FFprobe not found at: {bundled_ffprobe}")
    
    return ffmpeg_found and ffprobe_found


def test_sdk():
    """Test Insta360 SDK availability."""
    print("\n" + "="*60)
    print("TEST 2: Insta360 SDK Detection")
    print("="*60)
    
    app_path = get_app_path()
    
    # Check bundled location
    sdk_exe = app_path / 'sdk' / 'bin' / 'MediaSDKTest.exe'
    
    # Also check _internal for frozen app
    if not sdk_exe.exists():
        sdk_exe = app_path / '_internal' / 'sdk' / 'bin' / 'MediaSDKTest.exe'
    
    sdk_found = False
    
    if sdk_exe.exists():
        print(f"✅ SDK executable found: {sdk_exe}")
        sdk_found = True
        
        # Check for model files
        model_dir = sdk_exe.parent.parent / 'modelfile'
        if model_dir.exists():
            models = list(model_dir.glob('*.ins'))
            print(f"   Found {len(models)} model files:")
            for model in models[:5]:  # Show first 5
                print(f"      - {model.name}")
        else:
            print(f"   ⚠️ Model directory not found: {model_dir}")
    else:
        print(f"❌ SDK not found at: {sdk_exe}")
    
    return sdk_found


def test_onnx():
    """Test ONNX Runtime and model loading."""
    print("\n" + "="*60)
    print("TEST 3: ONNX Runtime & Model Detection")
    print("="*60)
    
    # Test import
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime imported successfully")
        print(f"   Version: {ort.__version__}")
        print(f"   Available providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"❌ ONNX Runtime import failed: {e}")
        return False
    
    # Test model loading
    app_path = get_app_path()
    
    model_paths = [
        app_path / 'yolov8s-seg.onnx',
        app_path / '_internal' / 'yolov8s-seg.onnx',
        Path('.') / 'yolov8s-seg.onnx',
    ]
    
    model_found = False
    for model_path in model_paths:
        if model_path.exists():
            print(f"✅ ONNX model found: {model_path}")
            model_found = True
            
            # Test loading
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    providers=['CPUExecutionProvider']
                )
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape
                print(f"   Model loaded successfully!")
                print(f"   Input: {input_name}, Shape: {input_shape}")
            except Exception as e:
                print(f"   ⚠️ Model loading failed: {e}")
                return False
            break
    
    if not model_found:
        print(f"❌ No ONNX model found in expected locations")
        return False
    
    return True


def test_opencv():
    """Test OpenCV availability."""
    print("\n" + "="*60)
    print("TEST 4: OpenCV Detection")
    print("="*60)
    
    try:
        import cv2
        print(f"✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
        
        # Test basic functionality
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        print(f"   Basic operations: Working")
        return True
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ OpenCV test failed: {e}")
        return False


def test_pyqt6():
    """Test PyQt6 availability."""
    print("\n" + "="*60)
    print("TEST 5: PyQt6 Detection")
    print("="*60)
    
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QT_VERSION_STR
        print(f"✅ PyQt6 imported successfully")
        print(f"   Qt Version: {QT_VERSION_STR}")
        return True
    except ImportError as e:
        print(f"❌ PyQt6 import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("360toolkit Build Functionality Test")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"Frozen: {getattr(sys, 'frozen', False)}")
    if getattr(sys, 'frozen', False):
        print(f"MEIPASS: {sys._MEIPASS}")
    
    results = {
        'FFmpeg': test_ffmpeg(),
        'SDK': test_sdk(),
        'ONNX': test_onnx(),
        'OpenCV': test_opencv(),
        'PyQt6': test_pyqt6(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 All tests passed! Build is functional.")
    else:
        print("⚠️ Some tests failed. Check the output above.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
