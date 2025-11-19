@echo off
REM ============================================================================
REM 360FrameTools - GPU Build Script (Windows)
REM Builds the GPU-accelerated version with CUDA support
REM ============================================================================

echo.
echo ====================================================================
echo 360FrameTools - GPU BUILD (CUDA-Enabled)
echo ====================================================================
echo.
echo This will build the GPU version with CUDA support for faster masking.
echo Binary size: ~2.3 GB (vs ~780 MB for CPU version)
echo Masking speed: ~5 minutes for 1000 images (vs ~10 minutes on CPU)
echo.
echo Requirements:
echo - NVIDIA GPU with CUDA support
echo - CUDA Toolkit installed
echo - PyTorch with CUDA support
echo.

REM Check if PyTorch is installed
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [ERROR] PyTorch not found!
    echo.
    echo Please install PyTorch with CUDA support:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    echo.
    echo Or for CUDA 12.1:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] PyTorch detected. Checking CUDA support...
echo.

python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo [WARNING] CUDA not available!
    echo.
    echo PyTorch is installed but cannot access GPU.
    echo This may be because:
    echo   1. No NVIDIA GPU detected
    echo   2. CUDA Toolkit not installed
    echo   3. GPU drivers outdated
    echo.
    echo You can continue, but the build will use CPU for inference.
    echo For GPU support, install CUDA Toolkit from:
    echo   https://developer.nvidia.com/cuda-downloads
    echo.
    pause
) else (
    echo [OK] CUDA is available!
    python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
    echo.
)

echo.
echo [INFO] Starting GPU build with PyInstaller...
echo [INFO] This may take 10-20 minutes depending on your system.
echo.

REM Build with PyInstaller
pyinstaller 360FrameTools_GPU.spec

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo Check the output above for errors.
    echo.
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo BUILD COMPLETE!
echo ====================================================================
echo.
echo Binary location: dist\360FrameTools\
echo Binary size: ~2.3 GB (with CUDA support)
echo.
echo To test the build:
echo   cd dist\360FrameTools
echo   360FrameTools.exe
echo.
echo To verify GPU is working:
echo   - Run the application
echo   - Check Stage 3 (Masking) settings
echo   - GPU should be detected and enabled
echo   - Monitor GPU usage with: nvidia-smi
echo.

pause
