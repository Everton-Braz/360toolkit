@echo off
echo ============================================
echo Rebuilding 360FrameTools with Bug Fixes
echo ============================================
echo.
echo Fixes applied:
echo  1. WinError 32: Added explicit PIL file handle close + 10ms delay
echo  2. Duplicate Stage 2: Added auto-advance flag for stage-only mode
echo.

cd /d C:\Users\User\Documents\APLICATIVOS\360ToolKit

echo Rebuilding executable...
C:\Users\User\miniconda3\envs\instantsplat\python.exe -m PyInstaller 360FrameTools_FULL.spec --noconfirm

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo BUILD SUCCESSFUL!
    echo ============================================
    echo.
    echo Output: dist\360ToolkitGS-FULL\360ToolkitGS-FULL.exe
    echo.
    echo Test Instructions:
    echo  1. Run same 29-frame, 8-camera test
    echo  2. Check logs for:
    echo     - NO WinError 32 errors
    echo     - Stage 2 appears ONLY ONCE (not twice)
    echo     - All 232 PNGs saved successfully
    echo     - Metadata embedded in all images
    echo.
) else (
    echo.
    echo ============================================
    echo BUILD FAILED - See errors above
    echo ============================================
    echo.
)

pause
