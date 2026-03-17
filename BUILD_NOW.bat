@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
python scripts\build_release.py build --variant full-bundled --zip %*
exit /b %ERRORLEVEL%
echo   2. Create ZIP: create_release_zip.bat
echo   3. Upload to Gumroad
echo.

pause
