@echo off
REM ============================================================================
REM Master Build Script for 360ToolkitGS
REM Builds both GPU and CPU versions
REM ============================================================================

echo.
echo ========================================================================
echo 360ToolkitGS - Master Build Script
echo ========================================================================
echo.
echo This script will build BOTH versions:
echo   1. GPU Version (~700 MB) - PyTorch excluded
echo   2. CPU Version (~800 MB) - PyTorch bundled
echo.
echo Prerequisites:
echo   - PyInstaller installed
echo   - Insta360 SDK at configured path
echo   - FFmpeg at configured path
echo.

REM Check for PyInstaller
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

echo.
echo ========================================================================
echo Build Options
echo ========================================================================
echo.
echo 1. Build GPU version only
echo 2. Build CPU version only
echo 3. Build BOTH versions
echo 4. Exit
echo.
set /p choice="Select option (1-4): "

if "%choice%"=="4" exit /b 0

REM Clean old builds
echo.
echo Cleaning old builds...
if exist "build" rmdir /s /q "build"

if "%choice%"=="1" goto build_gpu
if "%choice%"=="2" goto build_cpu
if "%choice%"=="3" goto build_both
echo Invalid option!
pause
exit /b 1

:build_both
echo.
echo ========================================================================
echo Building BOTH versions...
echo ========================================================================
call :build_gpu_func
call :build_cpu_func
goto summary

:build_gpu
echo.
echo ========================================================================
echo Building GPU version only...
echo ========================================================================
call :build_gpu_func
goto summary

:build_cpu
echo.
echo ========================================================================
echo Building CPU version only...
echo ========================================================================
call :build_cpu_func
goto summary

REM ============================================================================
REM GPU Build Function
REM ============================================================================
:build_gpu_func
echo.
echo --- Building GPU Version ---
echo.

if exist "dist\360ToolkitGS-GPU" rmdir /s /q "dist\360ToolkitGS-GPU"

echo Running PyInstaller for GPU version...
pyinstaller 360FrameTools.spec

if errorlevel 1 (
    echo ERROR: GPU build failed!
    pause
    exit /b 1
)

echo Copying additional files...
copy install_pytorch_gpu.bat "dist\360ToolkitGS-GPU\" 2>nul
copy README_GPU_VERSION.md "dist\360ToolkitGS-GPU\README.txt" 2>nul
copy LICENSE "dist\360ToolkitGS-GPU\" 2>nul

echo.
echo GPU Version Build Complete!
echo Location: dist\360ToolkitGS-GPU\
echo.
goto :eof

REM ============================================================================
REM CPU Build Function
REM ============================================================================
:build_cpu_func
echo.
echo --- Building CPU Version ---
echo.

REM Check PyTorch CPU is installed
python -c "import torch; exit(0 if '+cpu' in torch.__version__ or not torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: PyTorch CUDA detected! CPU version will be large (~3.5 GB)
    echo.
    echo Recommended: Install PyTorch CPU first:
    echo   pip uninstall torch torchvision torchaudio -y
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo.
    set /p continue_anyway="Continue with CUDA PyTorch? (Y/N): "
    if /i not "%continue_anyway%"=="Y" (
        echo Skipping CPU build.
        goto :eof
    )
)

if exist "dist\360ToolkitGS-CPU" rmdir /s /q "dist\360ToolkitGS-CPU"

echo Running PyInstaller for CPU version...
pyinstaller 360ToolkitGS-CPU.spec

if errorlevel 1 (
    echo ERROR: CPU build failed!
    pause
    exit /b 1
)

echo Copying additional files...
copy README_CPU_VERSION.md "dist\360ToolkitGS-CPU\README.txt" 2>nul
copy LICENSE "dist\360ToolkitGS-CPU\" 2>nul

echo.
echo CPU Version Build Complete!
echo Location: dist\360ToolkitGS-CPU\
echo.
goto :eof

REM ============================================================================
REM Summary
REM ============================================================================
:summary
echo.
echo ========================================================================
echo Build Summary
echo ========================================================================
echo.

if exist "dist\360ToolkitGS-GPU" (
    for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-GPU" ^| find "File(s)"') do set gpu_size=%%a
    echo GPU Version: dist\360ToolkitGS-GPU\
    echo   - Size: %gpu_size% bytes
    echo   - Executable: 360ToolkitGS-GPU.exe
    echo   - Includes: install_pytorch_gpu.bat, README.txt
    echo.
)

if exist "dist\360ToolkitGS-CPU" (
    for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-CPU" ^| find "File(s)"') do set cpu_size=%%a
    echo CPU Version: dist\360ToolkitGS-CPU\
    echo   - Size: %cpu_size% bytes
    echo   - Executable: 360ToolkitGS-CPU.exe
    echo   - Includes: README.txt
    echo.
)

echo ========================================================================
echo Next Steps
echo ========================================================================
echo.
echo 1. Test both versions on your system
echo 2. Test on clean machine without Python (important!)
echo 3. Create distribution packages:
echo.
echo    Option A - Portable ZIP (easiest):
if exist "dist\360ToolkitGS-GPU" (
    echo      .\create_portable_zip.bat
)
echo.
echo    Option B - Windows Installer (professional):
echo      Download Inno Setup: https://jrsoftware.org/isdl.php
echo      Then run: .\build_installer.bat
echo.
echo 4. See DISTRIBUTION_GUIDE.md for detailed instructions
echo.
echo ========================================================================
echo Distribution Ready!
echo ========================================================================
echo.
pause

