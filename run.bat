@echo off
REM 360FrameTools Launcher Script
REM Windows batch file to start the application

echo ================================================
echo 360FrameTools v1.0.0
echo Photogrammetry Preprocessing Pipeline
echo ================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo ERROR: Dependencies not installed!
    echo Please install dependencies:
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Run application
echo Starting 360FrameTools...
echo.
python -m src.main

REM Keep window open if error occurs
if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
