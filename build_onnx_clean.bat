@echo off
REM ============================================================================
REM Clean Build Script with ONNX Runtime Fix
REM ============================================================================

echo ============================================================
echo 360ToolkitGS - Clean ONNX Build
echo ============================================================
echo.

REM Activate conda environment
call C:\Users\User\miniconda3\Scripts\activate.bat 360toolkit-cpu
if errorlevel 1 (
    echo ERROR: Failed to activate 360toolkit-cpu environment
    pause
    exit /b 1
)

echo Environment: 360toolkit-cpu
python --version
echo.

REM Ensure ONNX Runtime is installed
echo Checking ONNX Runtime installation (GPU build)...
python -c "import onnxruntime as ort, sys; print('ONNX Runtime version:', ort.__version__); print('Providers:', ort.get_available_providers()); sys.exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)"
set GPU_PROVIDER_STATUS=%ERRORLEVEL%
if "%GPU_PROVIDER_STATUS%"=="0" goto HAVE_CUDA

echo.
echo Installing ONNX Runtime GPU (CUDA-enabled)...
pip install --upgrade onnxruntime-gpu==1.23.2
if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)

echo.
echo Verifying GPU-enabled ONNX Runtime...
python -c "import onnxruntime as ort, sys; print('ONNX Runtime version:', ort.__version__); print('Providers:', ort.get_available_providers()); sys.exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)"
set GPU_PROVIDER_STATUS=%ERRORLEVEL%
if "%GPU_PROVIDER_STATUS%"=="0" goto HAVE_CUDA

echo ERROR: CUDAExecutionProvider still unavailable. Check CUDA Toolkit installation.
pause
exit /b 1

:HAVE_CUDA

echo.
echo Verifying ONNX Runtime DLLs...
python -c "import os, onnxruntime; ort_dir = os.path.dirname(onnxruntime.__file__); capi = os.path.join(ort_dir, 'capi'); print('ONNX capi folder:', capi); print('DLLs exist:', os.path.exists(capi))"

echo.
echo Cleaning previous build...
if exist "build" rmdir /s /q "build"
if exist "dist\360ToolkitGS-ONNX" rmdir /s /q "dist\360ToolkitGS-ONNX"

echo.
echo Starting PyInstaller build...
pyinstaller 360ToolkitGS-ONNX.spec -y --clean

if errorlevel 1 (
    echo.
    echo ============================================================
    echo BUILD FAILED!
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo BUILD SUCCESSFUL!
echo ============================================================
echo.
echo Output: dist\360ToolkitGS-ONNX\360ToolkitGS-ONNX.exe
echo.
echo Verifying ONNX DLLs in build...
dir "dist\360ToolkitGS-ONNX\_internal\onnxruntime*.dll" 2>nul
if errorlevel 1 (
    echo WARNING: onnxruntime DLLs not found in _internal!
) else (
    echo OK: ONNX Runtime DLLs found
)

echo.
pause
