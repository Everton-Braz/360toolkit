@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo 360FrameTools Full Build
echo ========================================
echo.
echo Build started: %date% %time%
echo.

cd /d "C:\Users\User\Documents\APLICATIVOS\360ToolKit"

echo [1/4] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo Done.
echo.

echo [2/4] Starting PyInstaller build (this takes 15-30 minutes)...
echo Command: pyinstaller 360FrameTools_MINIMAL.spec --clean --onedir
echo.
pyinstaller 360FrameTools_MINIMAL.spec --clean --onedir

if %errorlevel% equ 0 (
    echo.
    echo [3/4] Build completed successfully!
    echo.
    if exist dist\360ToolkitGS-FULL (
        echo [4/4] Dist folder created. Contents:
        dir /b dist\
        echo.
        echo Executable location:
        echo   dist\360ToolkitGS-FULL\360FrameTools.exe
        echo.
        echo Bundle size:
        for /f %%A in ('powershell -Command "(Get-ChildItem dist -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB"') do echo   %%A GB
    ) else (
        echo ERROR: dist\360ToolkitGS-FULL not found!
    )
) else (
    echo.
    echo ERROR: Build failed with exit code %errorlevel%
)

echo.
echo Build ended: %date% %time%
echo.
pause
