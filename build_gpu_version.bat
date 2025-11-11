@echo off
REM ============================================================================
REM Build Script for 360ToolkitGS - GPU Version
REM ============================================================================

echo.
echo ========================================================================
echo Building 360ToolkitGS - GPU Version
echo ========================================================================
echo.
echo This version:
echo   - Excludes PyTorch (user installs separately)
echo   - Bundles SDK and FFmpeg
echo   - Expected size: ~700 MB
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
if exist "dist\360ToolkitGS-GPU" rmdir /s /q "dist\360ToolkitGS-GPU"

REM Build with PyInstaller
pyinstaller 360FrameTools.spec

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
echo Output location: dist\360ToolkitGS-GPU\
echo Executable: dist\360ToolkitGS-GPU\360ToolkitGS-GPU.exe
echo.

REM Copy additional files to distribution
echo Copying additional files...
copy install_pytorch_gpu.bat "dist\360ToolkitGS-GPU\"
copy README_GPU_VERSION.md "dist\360ToolkitGS-GPU\README.txt"
copy LICENSE "dist\360ToolkitGS-GPU\" 2>nul

echo.
echo Distribution includes:
echo   - 360ToolkitGS-GPU.exe
echo   - install_pytorch_gpu.bat (for users to install PyTorch)
echo   - README.txt (user guide)
echo   - _internal\ (dependencies: SDK, FFmpeg, etc.)
echo.

REM Get folder size
for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-GPU" ^| find "File(s)"') do set size=%%a
echo Total size: %size% bytes
echo.

echo ========================================================================
echo Testing build...
echo ========================================================================
echo.

REM Test if executable exists
if not exist "dist\360ToolkitGS-GPU\360ToolkitGS-GPU.exe" (
    echo ERROR: Executable not found!
    pause
    exit /b 1
)

echo Executable found: OK
echo.
echo IMPORTANT: Test the application on a clean machine without Python
echo            to ensure all dependencies are bundled correctly.
echo.
echo To distribute:
echo   1. Compress dist\360ToolkitGS-GPU\ to .zip
echo   2. Upload to your distribution platform
echo   3. Include instructions to run install_pytorch_gpu.bat first
echo.
pause
