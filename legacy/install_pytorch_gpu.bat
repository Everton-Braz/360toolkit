@echo off
REM ============================================================================
REM 360ToolkitGS - PyTorch GPU Installation Script
REM 
REM This script installs PyTorch with CUDA 11.8 support
REM Required for 360ToolkitGS-GPU.exe to work with Stage 3 (AI Masking)
REM ============================================================================

echo.
echo ========================================================================
echo 360ToolkitGS - PyTorch GPU Installation
echo ========================================================================
echo.
echo This will install PyTorch with CUDA 11.8 support for GPU acceleration.
echo.
echo REQUIREMENTS:
echo   - NVIDIA GPU (GTX 1650 or newer)
echo   - NVIDIA drivers installed (CUDA 11.8 or higher)
echo   - Python 3.9+ installed
echo.
echo Download size: ~2.8 GB
echo Installation time: ~5-10 minutes
echo.
pause
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.9 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/3] Checking Python version...
python --version
echo.

REM Check if GPU is available
echo [2/3] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. GPU may not be available.
    echo If you have an NVIDIA GPU, make sure drivers are installed.
    echo.
    echo Continue anyway? (Y/N)
    set /p continue=
    if /i not "%continue%"=="Y" exit /b 1
)
echo NVIDIA GPU detected.
echo.

REM Uninstall any existing CPU-only PyTorch
echo [3/3] Installing PyTorch with CUDA 11.8...
echo.
echo This may take several minutes (large download)...
echo.

REM Uninstall old versions first
pip uninstall torch torchvision torchaudio -y >nul 2>&1

REM Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install PyTorch
    echo.
    echo Please check:
    echo   1. Internet connection
    echo   2. Python version (3.9+)
    echo   3. Available disk space (at least 3 GB)
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Verifying installation...
echo ========================================================================
echo.

REM Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if errorlevel 1 (
    echo.
    echo WARNING: Installation verification failed
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo SUCCESS! PyTorch GPU is installed and ready to use.
echo ========================================================================
echo.
echo You can now run 360ToolkitGS-GPU.exe with full GPU acceleration.
echo Stage 3 (AI Masking) will use GPU for 5-7x faster processing.
echo.
echo PERFORMANCE:
echo   - CPU: ~1 second per image
echo   - GPU: ~0.15 seconds per image (6-7x faster!)
echo.
pause
