@echo off
REM ============================================================
REM 360toolkit - GPU Support Fixer
REM Uninstalls CPU-only PyTorch and installs CUDA-enabled version
REM ============================================================

echo.
echo ============================================================
echo 360toolkit - GPU Support Fixer
echo ============================================================
echo.
echo This script will:
echo 1. Uninstall CPU-only PyTorch
echo 2. Install CUDA-enabled PyTorch (CUDA 12.4)
echo 3. Install/update ONNX Runtime with CUDA support
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo Step 1: Removing CPU-only PyTorch...
echo ============================================================
pip uninstall -y torch torchvision torchaudio

echo.
echo ============================================================
echo Step 2: Installing CUDA-enabled PyTorch (CUDA 12.4)...
echo ============================================================
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo ============================================================
echo Step 3: Installing ONNX Runtime with CUDA support...
echo ============================================================
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu

echo.
echo ============================================================
echo Testing GPU availability...
echo ============================================================
python check_gpu_compatibility.py

echo.
echo ============================================================
echo Done! GPU support should now be enabled.
echo ============================================================
echo.
echo If CUDA is still not available:
echo 1. Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
echo 2. Check if your GPU is supported (requires compute capability 3.5+)
echo.
pause
