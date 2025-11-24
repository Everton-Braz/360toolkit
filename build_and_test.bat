@echo off
REM Build and Test Script for Simplified 360ToolkitGS
REM This script runs all tests and provides a build report

echo ============================================================
echo 360ToolkitGS - Simplified Version Build and Test
echo ============================================================
echo.

echo [1/5] Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    exit /b 1
)
echo.

echo [2/5] Checking required packages...
python -c "import cv2; print('  OpenCV:', cv2.__version__)"
python -c "import numpy; print('  NumPy:', numpy.__version__)"
python -c "import PyQt6; print('  PyQt6: OK')" 2>nul || echo   PyQt6: Not installed
echo.

echo [3/5] Checking ONNX Runtime (optional)...
python -c "import onnxruntime; print('  ONNX Runtime:', onnxruntime.__version__)" 2>nul || echo   ONNX Runtime: Not installed (optional)
echo.

echo [4/5] Running optimization tests...
python test_optimizations.py
if errorlevel 1 (
    echo.
    echo WARNING: Some tests failed. Review output above.
    echo.
) else (
    echo.
    echo SUCCESS: All tests passed!
    echo.
)

echo [5/5] Checking build requirements...
echo.
echo To build the executable:
echo   1. Install PyInstaller: pip install pyinstaller
echo   2. For ONNX version: pip install onnxruntime
echo   3. Export models: python export_yolo_to_onnx.py
echo   4. Build: pyinstaller 360FrameTools_ONNX.spec -y
echo.

echo ============================================================
echo Build check complete!
echo ============================================================
echo.
echo Next steps:
echo   - If tests passed: Ready to build
echo   - If tests failed: Fix issues above first
echo   - For ONNX build: Follow steps in QUICK_START_ONNX.md
echo.

pause
