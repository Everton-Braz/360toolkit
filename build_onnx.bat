@echo off
REM Simple ONNX Build Script for 360ToolkitGS
REM This script activates the correct environment and builds

echo ============================================================
echo Building 360ToolkitGS - ONNX Version
echo ============================================================
echo.

REM Activate 360toolkit-cpu conda environment
echo Activating 360toolkit-cpu environment...
call C:\Users\User\miniconda3\Scripts\activate.bat 360toolkit-cpu
if errorlevel 1 (
    echo ERROR: Failed to activate environment
    echo Please create it first: conda create -n 360toolkit-cpu python=3.10
    pause
    exit /b 1
)

echo.
echo Checking Python version...
python --version
echo.

REM Check PyInstaller
echo Checking PyInstaller...
python -c "import PyInstaller; print('PyInstaller installed')" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Check ONNX Runtime
echo Checking ONNX Runtime...
python -c "import onnxruntime; print(f'ONNX Runtime {onnxruntime.__version__} installed')" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: ONNX Runtime not installed!
    echo Installing ONNX Runtime...
    pip install onnxruntime
    if errorlevel 1 (
        echo ERROR: Failed to install ONNX Runtime
        pause
        exit /b 1
    )
    echo ONNX Runtime installed successfully!
)

echo.
echo Checking ONNX model files...
if not exist "yolov8n-seg.onnx" (
    echo WARNING: yolov8n-seg.onnx not found
    echo Please export models first: python export_yolo_to_onnx.py
    echo Or download from: https://github.com/ultralytics/assets/releases
)

echo.
echo Starting PyInstaller build...
echo.

REM Clean previous build
if exist "build" rmdir /s /q "build"
if exist "dist\360ToolkitGS-ONNX" rmdir /s /q "dist\360ToolkitGS-ONNX"

REM Build with PyInstaller
pyinstaller 360ToolkitGS-ONNX.spec -y

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Build failed!
    echo ============================================================
    echo.
    echo Check the error messages above.
    echo Common issues:
    echo   - Missing dependencies (pip install -r requirements.txt)
    echo   - SDK path incorrect in spec file
    echo   - FFmpeg path incorrect in spec file
    echo   - ONNX models missing
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Build completed successfully!
echo ============================================================
echo.
echo Output location: dist\360ToolkitGS-ONNX\
echo Executable: dist\360ToolkitGS-ONNX\360ToolkitGS.exe
echo.
echo To test: cd dist\360ToolkitGS-ONNX ^&^& 360ToolkitGS.exe
echo.
pause
