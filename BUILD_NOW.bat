@echo off
REM ============================================================================
REM 360ToolkitGS Build - Run this by double-clicking!
REM ============================================================================

echo.
echo ========================================================================
echo 360ToolkitGS - ONNX Build (Lightweight)
echo ========================================================================
echo.

cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM Check if ONNX models exist
if not exist "yolov8s-seg.onnx" (
    echo [!] ONNX models not found. Please run first:
    echo     python export_onnx_models.py
    echo.
    pause
    exit /b 1
)
echo [OK] ONNX models found

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

python -m PyInstaller 360ToolkitGS-Build.spec --noconfirm

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
