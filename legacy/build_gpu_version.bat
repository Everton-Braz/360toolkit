@echo off
REM ============================================================================
REM Build Script for 360ToolkitGS - GPU Version
REM ============================================================================

echo.
echo ========================================================================
echo Building 360ToolkitGS - GPU FULL Version
echo ========================================================================
echo.
echo This version:
echo   - Includes PyTorch GPU with CUDA 11.8 (bundled)
echo   - Bundles SDK and FFmpeg
echo   - Expected size: ~8-10 GB
echo   - Requires: NVIDIA GPU with CUDA 11.8 drivers
echo.

REM Activate the instantsplat conda environment (has PyTorch GPU + CUDA 11.8)
echo Activating instantsplat environment...
call C:\Users\User\miniconda3\Scripts\activate.bat instantsplat
if errorlevel 1 (
    echo ERROR: Failed to activate instantsplat environment
    echo Please ensure conda environment 'instantsplat' exists
    pause
    exit /b 1
)

echo Environment activated successfully
python --version
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

REM Check if SDK path exists
if not exist "C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1\MediaSDK-3.0.5-20250619-win64\MediaSDK" (
    echo ERROR: SDK path not found!
    echo Please update SDK_PATH in 360FrameTools.spec
    pause
    exit /b 1
)

REM Check if FFmpeg exists
if not exist "C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe" (
    echo WARNING: FFmpeg not found at default location
    echo Please update FFMPEG_PATH in 360FrameTools.spec
    pause
)

echo.
echo Starting build...
echo.

REM Clean previous build
if exist "build" rmdir /s /q "build"
if exist "dist\360ToolkitGS-FULL" rmdir /s /q "dist\360ToolkitGS-FULL"

REM Build with PyInstaller - Use FULL spec for GPU version
pyinstaller 360FrameTools_FULL.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Build Complete!
echo ========================================================================
echo.
echo Output location: dist\360ToolkitGS-FULL\
echo Executable: dist\360ToolkitGS-FULL\360ToolkitGS-FULL.exe
echo.

REM Copy additional files to distribution
echo Copying additional files...
copy install_pytorch_gpu.bat "dist\360ToolkitGS-FULL\" 2>nul
copy README_GPU_VERSION.md "dist\360ToolkitGS-FULL\README.txt" 2>nul
copy LICENSE "dist\360ToolkitGS-FULL\" 2>nul

echo.
echo Distribution includes:
echo   - 360ToolkitGS-FULL.exe (with PyTorch GPU bundled)
echo   - _internal\ (dependencies: SDK, FFmpeg, PyTorch, CUDA, etc.)
echo.

REM Get folder size
for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-FULL" ^| find "File(s)"') do set size=%%a
echo Total size: %size% bytes
echo.

echo ========================================================================
echo Testing build...
echo ========================================================================
echo.

REM Test if executable exists
if not exist "dist\360ToolkitGS-FULL\360ToolkitGS-FULL.exe" (
    echo ERROR: Executable not found!
    pause
    exit /b 1
)

echo Executable found: OK
echo.
echo IMPORTANT: This is the FULL version with PyTorch GPU bundled
echo            Test on machine with NVIDIA GPU and CUDA 11.8
echo.
echo To distribute:
echo   1. Compress dist\360ToolkitGS-FULL\ to .zip
echo   2. Upload to your distribution platform
echo   3. Users need: NVIDIA GPU with CUDA 11.8 drivers
echo.
pause
