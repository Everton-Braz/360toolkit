@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=full-bundled"
python scripts\build_release.py zip --variant %VARIANT%
exit /b %ERRORLEVEL%
echo.
echo Ready to upload to Gumroad!
echo URL: https://evertonbraz.gumroad.com/l/360toolkit
echo.
echo ========================================================================

REM Open releases folder
explorer releases

pause
