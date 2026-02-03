@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM Create Portable ZIP for Distribution
REM ============================================================================

echo.
echo ========================================================================
echo Creating Portable ZIP for Distribution
echo ========================================================================
echo.

set "DIST_DIR=dist\360ToolkitGS"
set "VERSION=1.1.0"
set "ZIP_NAME=360ToolkitGS-v%VERSION%-Windows-x64.zip"

REM Check if build exists
if not exist "%DIST_DIR%\360ToolkitGS.exe" (
    echo ERROR: Build not found at %DIST_DIR%
    echo Run build.bat first.
    pause
    exit /b 1
)

REM Create releases folder
if not exist "releases" mkdir releases

REM Calculate size before zipping
echo Calculating folder size...
for /f "tokens=3" %%a in ('dir "%DIST_DIR%" /s /-c ^| find "File(s)"') do set SIZE_BYTES=%%a
set /a SIZE_MB=%SIZE_BYTES:,=%/1048576 2>nul
echo Build size: approximately %SIZE_MB% MB

REM Create ZIP using PowerShell (built-in, no external tools needed)
echo.
echo Creating ZIP archive...
echo Output: releases\%ZIP_NAME%
echo.

powershell -Command "Compress-Archive -Path '%DIST_DIR%\*' -DestinationPath 'releases\%ZIP_NAME%' -Force"

if errorlevel 1 (
    echo ERROR: Failed to create ZIP
    pause
    exit /b 1
)

REM Get ZIP size
for %%a in (releases\%ZIP_NAME%) do set ZIP_SIZE=%%~za
set /a ZIP_SIZE_MB=%ZIP_SIZE%/1048576

echo.
echo ========================================================================
echo ZIP CREATED SUCCESSFULLY
echo ========================================================================
echo.
echo File: releases\%ZIP_NAME%
echo Size: %ZIP_SIZE_MB% MB
echo.
echo Ready to upload to Gumroad!
echo URL: https://evertonbraz.gumroad.com/l/360toolkit
echo.
echo ========================================================================

REM Open releases folder
explorer releases

pause
