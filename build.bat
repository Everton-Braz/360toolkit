@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
python scripts\build_release.py build --variant customer-managed --zip %*
exit /b %ERRORLEVEL%
if exist "build\360ToolkitGS" rmdir /s /q "build\360ToolkitGS"
if exist "dist\360ToolkitGS" rmdir /s /q "dist\360ToolkitGS"
echo Done.

REM Build
echo.
echo [6/6] Building with PyInstaller...
echo.
python -m PyInstaller 360ToolkitGS.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo BUILD FAILED
    echo ========================================================================
    echo Check the error messages above.
    pause
    exit /b 1
)

REM Calculate size
echo.
echo ========================================================================
echo BUILD SUCCESSFUL
echo ========================================================================
echo.
echo Output: dist\360ToolkitGS\

REM Get folder size
for /f "tokens=3" %%a in ('dir "dist\360ToolkitGS" /s /-c ^| find "File(s)"') do set SIZE=%%a
echo Size: !SIZE! bytes

REM Convert to MB
set /a SIZE_MB=!SIZE!/1048576
echo Size: !SIZE_MB! MB (approximately)
echo.

REM Test if it runs
echo Testing executable...
dist\360ToolkitGS\360ToolkitGS.exe --version >nul 2>&1
if errorlevel 1 (
    echo NOTE: Quick test completed. Run the app to fully test.
) else (
    echo Quick test: OK
)

echo.
echo ========================================================================
echo Next Steps:
echo ========================================================================
echo 1. Test the app: dist\360ToolkitGS\360ToolkitGS.exe
echo 2. Create ZIP for distribution: scripts\create_portable_zip.bat
echo 3. Upload to Gumroad: https://evertonbraz.gumroad.com/l/360toolkit
echo.
pause
