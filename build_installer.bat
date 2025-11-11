@echo off
REM Build 360FrameTools Windows Installer using Inno Setup
REM Download Inno Setup from: https://jrsoftware.org/isdl.php

echo ====================================================================
echo Building 360FrameTools Windows Installer
echo ====================================================================
echo.

REM Check if Inno Setup is installed
set INNO_COMPILER="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if not exist %INNO_COMPILER% (
    echo ERROR: Inno Setup not found!
    echo.
    echo Please download and install Inno Setup from:
    echo https://jrsoftware.org/isdl.php
    echo.
    echo After installation, run this script again.
    pause
    exit /b 1
)

echo Found Inno Setup compiler: %INNO_COMPILER%
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

echo Checking dist folder size...
for /f "tokens=3" %%a in ('dir /s "dist\360ToolkitGS-GPU" ^| find "File(s)"') do set SIZE=%%a
echo Dist folder contains approximately %SIZE% bytes
echo.

REM Create installer output directory
if not exist "installer_output" mkdir installer_output

echo Compiling installer with Inno Setup...
echo This may take 5-10 minutes (compressing 5 GB)...
echo.

%INNO_COMPILER% installer_setup.iss

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================================
    echo SUCCESS! Installer created
    echo ====================================================================
    echo.
    echo Installer location: installer_output\360FrameTools-GPU-Setup-v1.0.0.exe
    echo.
    echo Expected installer size: ~2-3 GB (compressed from 5 GB using LZMA2)
    echo.
    dir installer_output\*.exe
    echo.
    echo Ready to distribute!
    echo.
) else (
    echo.
    echo ERROR: Installer build failed!
    echo Check the error messages above.
    echo.
)

pause
