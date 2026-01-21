@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM 360ToolkitGS Build Script
REM ONNX Version - Lightweight Build without PyTorch
REM ============================================================================

echo.
echo ========================================================================
echo 360ToolkitGS - Build Script (ONNX Version)
echo ========================================================================
echo.
echo This build:
echo   - Uses ONNX Runtime (CPU + optional GPU)
echo   - EXCLUDES PyTorch (saves ~6 GB)
echo   - Bundles Insta360 SDK, FFmpeg, YOLOv8 ONNX models
echo   - Target size: ~400-600 MB
echo   - Works on any Windows PC
echo.

REM Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)
python --version

REM Check PyInstaller
echo.
echo [2/6] Checking PyInstaller...
python -c "import PyInstaller; print(f'PyInstaller {PyInstaller.__version__}')"
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Check ONNX Runtime
echo.
echo [3/6] Checking ONNX Runtime...
python -c "import onnxruntime; print(f'ONNX Runtime {onnxruntime.__version__}'); print(f'Providers: {onnxruntime.get_available_providers()}')"
if errorlevel 1 (
    echo.
    echo ONNX Runtime not found. Installing...
    echo.
    echo Choose version:
    echo   1 = CPU only (smaller, works everywhere)
    echo   2 = GPU (CUDA) - requires NVIDIA GPU + CUDA installed
    echo.
    set /p ONNX_CHOICE="Enter choice (1 or 2): "
    if "!ONNX_CHOICE!"=="2" (
        pip install onnxruntime-gpu
    ) else (
        pip install onnxruntime
    )
)

REM Export ONNX models
echo.
echo [4/6] Checking ONNX models...
if not exist "yolov8s-seg.onnx" (
    echo ONNX models not found. Exporting from PyTorch models...
    echo This requires ultralytics package (one-time export)
    python export_onnx_models.py
    if not exist "yolov8s-seg.onnx" (
        echo.
        echo WARNING: Could not export ONNX models.
        echo Make sure ultralytics is installed: pip install ultralytics
        echo Or download pre-exported models manually.
        pause
    )
) else (
    echo Found ONNX models:
    for %%f in (yolov8*-seg.onnx) do echo   - %%f
)

REM Clean previous build
echo.
echo [5/6] Cleaning previous build...
if exist "build\360ToolkitGS" rmdir /s /q "build\360ToolkitGS"
if exist "dist\360ToolkitGS" rmdir /s /q "dist\360ToolkitGS"
echo Done.

REM Build
echo.
echo [6/6] Building with PyInstaller...
echo.
python -m PyInstaller 360ToolkitGS-Build.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo BUILD FAILED
    echo ========================================================================
    echo Check the error messages above.
    pause
    exit /b 1
)

REM Calculate size
echo.
echo ========================================================================
echo BUILD SUCCESSFUL
echo ========================================================================
echo.
echo Output: dist\360ToolkitGS\

REM Get folder size
for /f "tokens=3" %%a in ('dir "dist\360ToolkitGS" /s /-c ^| find "File(s)"') do set SIZE=%%a
echo Size: !SIZE! bytes

REM Convert to MB
set /a SIZE_MB=!SIZE!/1048576
echo Size: !SIZE_MB! MB (approximately)
echo.

REM Test if it runs
echo Testing executable...
dist\360ToolkitGS\360ToolkitGS.exe --version >nul 2>&1
if errorlevel 1 (
    echo NOTE: Quick test completed. Run the app to fully test.
) else (
    echo Quick test: OK
)

echo.
echo ========================================================================
echo Next Steps:
echo ========================================================================
echo 1. Test the app: dist\360ToolkitGS\360ToolkitGS.exe
echo 2. Create ZIP for distribution: scripts\create_portable_zip.bat
echo 3. Upload to Gumroad: https://evertonbraz.gumroad.com/l/360toolkit
echo.
pause
