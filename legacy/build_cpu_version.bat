@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM Build Script for 360ToolkitGS - CPU Version
REM ============================================================================

echo.
echo ========================================================================
echo Building 360ToolkitGS - CPU Version
echo ========================================================================
echo.
echo This version:
echo   - Uses PyTorch CPU (fully bundled)
echo   - Bundles SDK and FFmpeg
echo   - Expected size: ~6-8 GB (CPU inference)
echo   - Works on any PC (no GPU required)
echo.

REM Activate the 360toolkit-cpu conda environment
echo Activating 360toolkit-cpu conda environment...
call C:\Users\User\miniconda3\Scripts\activate.bat 360toolkit-cpu
if errorlevel 1 (
    echo ERROR: Failed to activate 360toolkit-cpu environment
    echo Please ensure the environment exists: conda create -n 360toolkit-cpu python=3.10
    pause
    exit /b 1
)

echo Environment activated: 360toolkit-cpu
echo.

REM Verify PyTorch CPU is installed
echo Checking PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch not found in 360toolkit-cpu environment!
    echo.
    echo Please install PyTorch CPU:
    echo   conda activate 360toolkit-cpu
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo.
    echo Build cancelled. Please install PyTorch CPU first.
    pause
    exit /b 1
)

REM Verify PyTorch is CPU version (not CUDA)
echo Verifying PyTorch CPU version...
python -c "import torch, sys; print('DEBUG: Version =', torch.__version__); print('DEBUG: Has +cpu =', '+cpu' in torch.__version__); sys.exit(1 if '+cpu' not in torch.__version__ else 0)"
echo DEBUG: ERRORLEVEL after Python = !ERRORLEVEL!
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo WARNING: PyTorch CUDA version detected!
    echo This will bundle the large CUDA version (~8 GB)
    echo.
    echo For CPU version, the environment should have PyTorch CPU.
    echo Please check: pip list ^| findstr torch
    echo.
    echo Continue anyway? (Y/N)
    set /p continue_cuda="Continue with CUDA PyTorch? "
    if /i not "!continue_cuda!"=="Y" (
        echo Build cancelled.
        pause
        exit /b 1
    )
)

echo PyTorch CPU detected - Proceeding with CPU build
echo.

REM Check if SDK path exists
if not exist "C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1\MediaSDK-3.0.5-20250619-win64\MediaSDK" (
    echo ERROR: SDK path not found!
    echo Please update SDK_PATH in 360ToolkitGS-CPU.spec
    pause
    exit /b 1
)

REM Check if FFmpeg exists
if not exist "C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe" (
    echo WARNING: FFmpeg not found at default location
    echo Please update FFMPEG_PATH in 360ToolkitGS-CPU.spec
    pause
)

echo.
echo Starting build...
echo.

REM Clean previous build
if exist "build" rmdir /s /q "build"
if exist "dist\360ToolkitGS-CPU" rmdir /s /q "dist\360ToolkitGS-CPU"

REM Build with PyInstaller
pyinstaller 360ToolkitGS-CPU.spec

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
echo Output location: dist\360ToolkitGS-CPU\
echo Executable: dist\360ToolkitGS-CPU\360ToolkitGS-CPU.exe
echo.

REM Copy additional files to distribution
echo Copying additional files...
copy README_CPU_VERSION.md "dist\360ToolkitGS-CPU\README.txt" 2>nul
copy LICENSE "dist\360ToolkitGS-CPU\" 2>nul

echo.
echo Distribution includes:
echo   - 360ToolkitGS-CPU.exe
echo   - README.txt (user guide)
echo   - _internal\ (dependencies: PyTorch CPU, SDK, FFmpeg, etc.)
echo.

REM Get folder size
for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-CPU" ^| find "File(s)"') do set size=%%a
echo Total size: %size% bytes
echo.

echo ========================================================================
echo Testing build...
echo ========================================================================
echo.

REM Test if executable exists
if not exist "dist\360ToolkitGS-CPU\360ToolkitGS-CPU.exe" (
    echo ERROR: Executable not found!
    pause
    exit /b 1
)

echo Executable found: OK
echo.
echo IMPORTANT: Test the application on a clean machine WITHOUT Python
echo            to ensure all dependencies are bundled correctly.
echo.
echo This version should work immediately without any setup!
echo.
echo To distribute:
echo   1. Compress dist\360ToolkitGS-CPU\ to .zip
echo   2. Upload to your distribution platform
echo   3. Users just extract and run - no setup required!
echo.
pause
