@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM Build Script for 360ToolkitGS - ONNX Version
REM ============================================================================

echo.
echo ========================================================================
echo Building 360ToolkitGS - ONNX Version
echo ========================================================================
echo.
echo This version:
echo   - Uses ONNX Runtime (Lightweight)
echo   - EXCLUDES PyTorch (Saves ~6GB)
echo   - Bundles SDK, FFmpeg, and ONNX Models
echo   - Expected size: ~500 MB
echo   - Works on any PC (CPU/GPU)
echo.

REM Activate the 360toolkit-cpu conda environment
echo Activating 360toolkit-cpu conda environment...
call C:\Users\User\miniconda3\Scripts\activate.bat 360toolkit-cpu
if errorlevel 1 (
    echo ERROR: Failed to activate 360toolkit-cpu environment
    pause
    exit /b 1
)

echo Environment activated: 360toolkit-cpu
echo.

REM Verify ONNX Runtime is installed
echo Checking ONNX Runtime installation...
python -c "import onnxruntime; print(f'ONNX Runtime {onnxruntime.__version__} installed')"
if errorlevel 1 (
    echo.
    echo ERROR: ONNX Runtime not found!
    echo Please install: pip install onnxruntime
    pause
    exit /b 1
)

REM Check if ONNX models exist
if not exist "yolov8s-seg.onnx" (
    echo.
    echo WARNING: ONNX models not found in current directory!
    echo Attempting to export them now...
    python export_onnx.py
    if not exist "yolov8s-seg.onnx" (
        echo ERROR: Failed to export ONNX models.
        pause
        exit /b 1
    )
)

echo.
echo Starting build...
echo.

REM Clean previous build
if exist "build\360ToolkitGS-ONNX" rmdir /s /q "build\360ToolkitGS-ONNX"
if exist "dist\360ToolkitGS-ONNX" rmdir /s /q "dist\360ToolkitGS-ONNX"

REM Build with PyInstaller
pyinstaller 360ToolkitGS-ONNX.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Applying DLL Fixes (DLL Spraying)
echo ========================================================================
echo.

set "DIST_DIR=dist\360ToolkitGS-ONNX"
set "INTERNAL_DIR=%DIST_DIR%\_internal"

REM 1. Ensure vcomp140.dll is present (OpenMP support for OpenCV/SDK)
if exist "%CONDA_PREFIX%\Library\bin\vcomp140.dll" (
    echo Copying vcomp140.dll from Conda...
    copy /y "%CONDA_PREFIX%\Library\bin\vcomp140.dll" "%INTERNAL_DIR%\"
)

REM 2. Fix SDK Dependencies (Fixes Exit Code 3221225781 - DLL not found)
echo Fixing SDK dependencies...
set "SDK_ROOT=C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1\MediaSDK-3.0.5-20250619-win64\MediaSDK"

REM CRITICAL: Copy MediaSDK.dll to _internal root (SDK exe depends on it via PATH)
if exist "%SDK_ROOT%\bin\MediaSDK.dll" (
    echo - Copying MediaSDK.dll to _internal root...
    copy /y "%SDK_ROOT%\bin\MediaSDK.dll" "%INTERNAL_DIR%\" >nul
    echo - Copying MediaSDK.dll to sdk\bin...
    copy /y "%SDK_ROOT%\bin\MediaSDK.dll" "%INTERNAL_DIR%\sdk\bin\" >nul
) else (
    echo WARNING: MediaSDK.dll not found in SDK installation!
)

REM Copy VC++ Runtimes to SDK bin folder (for SDK dependencies)
echo - Spraying MSVC runtimes to sdk\bin...
for %%f in (msvcp140.dll msvcp140_1.dll msvcp140_2.dll vcruntime140.dll vcruntime140_1.dll concrt140.dll vcomp140.dll ucrtbase.dll) do (
    if exist "%INTERNAL_DIR%\%%f" copy /y "%INTERNAL_DIR%\%%f" "%INTERNAL_DIR%\sdk\bin\" >nul 2>&1
)

REM 3. Fix ONNX Runtime Dependencies (Fixes python310.dll initialization)
echo Fixing ONNX Runtime dependencies...

REM Copy Python DLL to ONNX capi folder
if exist "%INTERNAL_DIR%\python310.dll" (
    echo - Copying python310.dll to onnxruntime\capi...
    copy /y "%INTERNAL_DIR%\python310.dll" "%INTERNAL_DIR%\onnxruntime\capi\" >nul
) else (
    echo WARNING: python310.dll not found!
)

REM Copy MSVC runtimes to ONNX capi folder (fixes "DLL initialization failed")
echo - Spraying MSVC runtimes to onnxruntime\capi...
for %%f in (msvcp140.dll msvcp140_1.dll msvcp140_2.dll vcruntime140.dll vcruntime140_1.dll concrt140.dll vcomp140.dll ucrtbase.dll) do (
    if exist "%INTERNAL_DIR%\%%f" copy /y "%INTERNAL_DIR%\%%f" "%INTERNAL_DIR%\onnxruntime\capi\" >nul 2>&1
)

REM Copy python3.dll to ONNX capi if present (some ONNX builds need it)
if exist "%INTERNAL_DIR%\python3.dll" (
    echo - Copying python3.dll to onnxruntime\capi...
    copy /y "%INTERNAL_DIR%\python3.dll" "%INTERNAL_DIR%\onnxruntime\capi\" >nul
)

echo.
echo ========================================================================
echo Build Complete!
echo ========================================================================
echo.
echo Output location: dist\360ToolkitGS-ONNX\
echo Executable: dist\360ToolkitGS-ONNX\360ToolkitGS-ONNX.exe
echo.

REM Copy additional files to distribution
echo Copying additional files...
copy README_CPU_VERSION.md "dist\360ToolkitGS-ONNX\README.txt" 2>nul
copy LICENSE "dist\360ToolkitGS-ONNX\" 2>nul

REM Get folder size
for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-ONNX" ^| find "File(s)"') do set size=%%a
echo Total size: %size% bytes
echo.

echo ========================================================================
echo Testing build...
echo ========================================================================
echo.

REM Test if executable exists
if not exist "dist\360ToolkitGS-ONNX\360ToolkitGS-ONNX.exe" (
    echo ERROR: Executable not found!
    pause
    exit /b 1
)

echo Executable found: OK
echo.
echo IMPORTANT: This version uses ONNX Runtime and does NOT require PyTorch.
echo.

pause
