@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM 360ToolkitGS Build - Run this by double-clicking!
REM Full GPU Build (PyTorch + CUDA 12.8)
REM ============================================================================

echo.
echo ========================================================================
echo 360ToolkitGS - Full GPU Build
echo ========================================================================
echo.

cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM Activate the correct conda environment with CUDA PyTorch
echo Activating 360pipeline environment...
call conda activate 360pipeline
if errorlevel 1 (
    echo ERROR: Could not activate 360pipeline environment.
    echo Run: conda activate 360pipeline
    pause
    exit /b 1
)

REM Verify PyTorch CUDA works
echo Verifying PyTorch CUDA...
python -c "import torch; assert torch.cuda.is_available() and 'cu' in torch.__version__, 'No CUDA'; print(f'PyTorch {torch.__version__} CUDA OK')"
if errorlevel 1 (
    echo ERROR: PyTorch CUDA not available in 360pipeline env!
    echo Fix: pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)
echo [OK] PyTorch CUDA verified

REM Clean previous build
echo.
echo Cleaning previous build...
if exist "dist\360ToolkitGS" rmdir /s /q "dist\360ToolkitGS"
if exist "build\360ToolkitGS" rmdir /s /q "build\360ToolkitGS"
echo [OK] Cleaned

REM Run PyInstaller
echo.
echo ========================================================================
echo Starting PyInstaller build...
echo This may take 5-10 minutes. Please wait...
echo ========================================================================
echo.

python -m PyInstaller 360ToolkitGS.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo BUILD FAILED! Check errors above.
    echo ========================================================================
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo BUILD SUCCESSFUL!
echo ========================================================================
echo.
echo Output: dist\360ToolkitGS\
echo.

REM Show folder size
for /f "tokens=3" %%a in ('dir "dist\360ToolkitGS" /s /-c ^| find "File(s)"') do (
    set /a SIZE_MB=%%a/1048576
    echo Total size: approximately !SIZE_MB! MB
)

echo.
echo Next steps:
echo   1. Test: dist\360ToolkitGS\360ToolkitGS.exe
echo   2. Create ZIP: create_release_zip.bat
echo   3. Upload to Gumroad
echo.

pause
