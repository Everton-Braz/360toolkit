@echo off
REM ============================================================
REM Update PyTorch for RTX 50-series (Blackwell) GPU Support
REM ============================================================
REM This script installs PyTorch nightly build with CUDA 12.4
REM Required for RTX 5090/5080/5070 Ti and other RTX 50-series
REM ============================================================

echo.
echo ============================================================
echo 360toolkit - PyTorch RTX 50-series GPU Support Updater
echo ============================================================
echo.
echo This will install PyTorch nightly build with CUDA 12.4 support
echo Required for: RTX 5090, RTX 5080, RTX 5070 Ti, RTX 5070
echo.
echo Compatible with: RTX 30/40/50 series, GTX 1000+ series
echo.
pause

echo.
echo Removing potentially incompatible PyTorch versions...
pip uninstall -y torch torchvision

echo.
echo Installing PyTorch nightly with CUDA 12.8 support (for RTX 50-series)...
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Close and reopen 360toolkit
echo 2. Check log for "GPU compatibility verified" message
echo 3. GPU masking should now work without errors
echo.
pause
