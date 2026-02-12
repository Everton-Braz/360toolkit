@echo off
REM 360toolkit Launch Script - Auto-detects environment and runs app
REM Works on GTX, RTX 30/40/50 series GPUs

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║            360toolkit - Launching Application             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if .venv exists
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found!
    echo.
    echo Creating virtual environment...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

REM Check if requirements are installed
echo.
echo Checking dependencies...
python -c "import PyQt6" 2>nul
if !errorlevel! neq 0 (
    echo ⚠️  Missing dependencies. Installing...
    pip install -q -r requirements.txt
    if !errorlevel! neq 0 (
        echo Failed to install requirements
        pause
        exit /b 1
    )
)

echo ✅ All dependencies available

REM Run GPU setup and diagnostics
echo.
echo Running GPU diagnostics...
python setup_gpu_environment.py
if !errorlevel! neq 0 (
    echo ⚠️  GPU diagnostics completed with warnings
)

REM Launch application
echo.
echo ════════════════════════════════════════════════════════════
echo Starting 360toolkit...
echo ════════════════════════════════════════════════════════════
echo.

python run_app.py

REM If app exits, keep window open
if !errorlevel! neq 0 (
    echo.
    echo ❌ Application exited with error code: !errorlevel!
    echo.
    pause
)

endlocal
