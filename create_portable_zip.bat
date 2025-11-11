@echo off
REM Create Portable ZIP Distribution for 360FrameTools

echo ====================================================================
echo Creating 360FrameTools Portable ZIP Distribution
echo ====================================================================
echo.

REM Check if dist folder exists
if not exist "dist\360ToolkitGS-GPU" (
    echo ERROR: dist\360ToolkitGS-GPU folder not found!
    echo.
    echo Please build the GPU version first:
    echo   pyinstaller 360FrameTools.spec
    echo.
    pause
    exit /b 1
)

echo Copying user files to dist folder...
copy "install_pytorch_gpu.bat" "dist\360ToolkitGS-GPU\" >nul
copy "README_GPU_VERSION.md" "dist\360ToolkitGS-GPU\README.txt" >nul

echo.
echo Checking folder size...
for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-GPU" ^| find "File(s)"') do set SIZE=%%a
echo Folder contains approximately %SIZE% bytes (~5 GB)
echo.

echo Creating ZIP archive...
echo This may take 5-10 minutes (compressing 5 GB)...
echo.

powershell -Command "Compress-Archive -Path 'dist\360ToolkitGS-GPU\*' -DestinationPath '360FrameTools-GPU-v1.0.0-Portable.zip' -CompressionLevel Optimal -Force"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================================
    echo SUCCESS! Portable ZIP created
    echo ====================================================================
    echo.
    echo File: 360FrameTools-GPU-v1.0.0-Portable.zip
    echo.
    dir "360FrameTools-GPU-v1.0.0-Portable.zip"
    echo.
    echo Expected size: ~2-3 GB (compressed from 5 GB)
    echo.
    echo Ready to distribute!
    echo Users should:
    echo   1. Extract ZIP to any folder
    echo   2. Run 360ToolkitGS-GPU.exe
    echo   3. Run install_pytorch_gpu.bat for masking support
    echo.
) else (
    echo.
    echo ERROR: ZIP creation failed!
    echo.
)

pause
