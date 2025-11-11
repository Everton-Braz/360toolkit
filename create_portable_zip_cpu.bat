@echo off
REM Create Portable ZIP Distribution for 360FrameTools CPU Version

echo ====================================================================
echo Creating 360FrameTools CPU Version Portable ZIP Distribution
echo ====================================================================
echo.

REM Check if dist folder exists
if not exist "dist\360ToolkitGS-CPU" (
    echo ERROR: dist\360ToolkitGS-CPU folder not found!
    echo.
    echo Please build the CPU version first:
    echo   pyinstaller 360ToolkitGS-CPU.spec
    echo.
    pause
    exit /b 1
)

echo Copying user files to dist folder...
copy "README_CPU_VERSION.md" "dist\360ToolkitGS-CPU\README.txt" >nul 2>&1
if not exist "README_CPU_VERSION.md" (
    echo Note: README_CPU_VERSION.md not found, skipping...
)

echo.
echo Checking folder size...
for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-CPU" ^| find "File(s)"') do set SIZE=%%a
echo Folder contains approximately %SIZE% bytes
echo.
echo WARNING: CPU version is LARGE (~10-12 GB) because it includes PyTorch!
echo Compression will take 10-15 minutes.
echo.

set /p continue="Continue with ZIP creation? (Y/N): "
if /i not "%continue%"=="Y" (
    echo.
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Creating ZIP archive...
echo This may take 10-15 minutes (compressing ~10 GB)...
echo.

powershell -Command "Compress-Archive -Path 'dist\360ToolkitGS-CPU\*' -DestinationPath '360FrameTools-CPU-v1.0.0-Portable.zip' -CompressionLevel Optimal -Force"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================================
    echo SUCCESS! Portable ZIP created
    echo ====================================================================
    echo.
    echo File: 360FrameTools-CPU-v1.0.0-Portable.zip
    echo.
    dir "360FrameTools-CPU-v1.0.0-Portable.zip"
    echo.
    echo Expected size: ~4-6 GB (compressed from ~10 GB)
    echo.
    echo Ready to distribute!
    echo Users should:
    echo   1. Extract ZIP to any folder
    echo   2. Run 360ToolkitGS-CPU.exe
    echo   3. All features work immediately (PyTorch bundled)
    echo.
) else (
    echo.
    echo ERROR: ZIP creation failed!
    echo.
)

pause
