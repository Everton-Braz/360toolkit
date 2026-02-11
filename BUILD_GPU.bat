@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM 360ToolkitGS Build Script - Full GPU Version
REM PyTorch 2.10 + CUDA 12.8 (RTX 30/40/50 native support)
REM ============================================================================

echo.
echo ========================================================================
echo 360ToolkitGS - GPU Build (PyTorch + CUDA 12.8)
echo ========================================================================
echo.
echo This build:
echo   - Uses PyTorch 2.10 with CUDA 12.8
echo   - RTX 30xx, 40xx, 50xx NATIVE GPU support
echo   - Bundles Ultralytics YOLOv8 + SAM hybrid masking
echo   - Bundles Insta360 SDK, FFmpeg, SphereSfM/COLMAP
echo   - Target size: ~3-4 GB (includes CUDA runtime)
echo.

cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM ============================================================================
REM Step 1: Activate correct conda environment
REM ============================================================================
echo [1/7] Activating 360pipeline conda environment...
call conda activate 360pipeline
if errorlevel 1 (
    echo ERROR: Could not activate 360pipeline conda environment.
    echo Make sure it exists: conda env list
    pause
    exit /b 1
)
echo [OK] Environment: 360pipeline

REM ============================================================================
REM Step 2: Verify PyTorch CUDA
REM ============================================================================
echo.
echo [2/7] Verifying PyTorch CUDA...
python -c "import torch; assert 'cu' in torch.__version__, f'CPU-only torch: {torch.__version__}'; assert torch.cuda.is_available(), 'CUDA not available'; t=torch.tensor([1.0],device='cuda'); assert (t+1).item()==2.0, 'Kernel test failed'; print(f'  PyTorch {torch.__version__} - CUDA {torch.version.cuda}'); print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}'); print('  Kernel test: PASSED')"
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch CUDA verification failed!
    echo.
    echo Fix: Install PyTorch with CUDA support:
    echo   pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu128
    echo.
    pause
    exit /b 1
)
echo [OK] PyTorch CUDA verified

REM ============================================================================
REM Step 3: Verify key dependencies
REM ============================================================================
echo.
echo [3/7] Verifying dependencies...
python -c "import ultralytics; print(f'  Ultralytics: {ultralytics.__version__}')"
if errorlevel 1 (
    echo   Installing ultralytics...
    pip install ultralytics
)
python -c "import segment_anything; print('  SAM: OK')" 2>nul
if errorlevel 1 (
    echo   SAM not installed (optional - hybrid masking disabled)
)
python -c "import PyQt6; print('  PyQt6: OK')"
if errorlevel 1 (
    echo ERROR: PyQt6 not found!
    pause
    exit /b 1
)
python -c "import PyInstaller; print(f'  PyInstaller: {PyInstaller.__version__}')"
if errorlevel 1 (
    echo   Installing PyInstaller...
    pip install pyinstaller
)
echo [OK] Dependencies verified

REM ============================================================================
REM Step 4: Check YOLO/SAM models
REM ============================================================================
echo.
echo [4/7] Checking model files...
set MODELS_FOUND=0
for %%f in (yolov8m-seg.pt yolov8m.pt yolov8n-seg.pt) do (
    if exist "%%f" (
        echo   [OK] %%f
        set /a MODELS_FOUND+=1
    )
)
if exist "sam_vit_b_01ec64.pth" (
    echo   [OK] sam_vit_b_01ec64.pth
    set /a MODELS_FOUND+=1
)
if !MODELS_FOUND! EQU 0 (
    echo   [WARN] No model files found - they will auto-download on first use
)
echo [OK] Model check complete

REM ============================================================================
REM Step 5: Check SDK and FFmpeg
REM ============================================================================
echo.
echo [5/7] Checking external tools...
python -c "from pathlib import Path; sdk=Path(r'C:\Users\Everton-PC\Documents\Windows_CameraSDK-2.1.1_MediaSDK-3.1.0\MediaSDK-3.1.0.0-20250904-win64\MediaSDK'); print(f'  SDK: {\"OK\" if sdk.exists() else \"NOT FOUND\"} ({sdk})')"
python -c "from pathlib import Path; ff=Path(r'C:\Users\Everton-PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe'); print(f'  FFmpeg: {\"OK\" if ff.exists() else \"NOT FOUND\"} ({ff})')"
echo [OK] External tools checked

REM ============================================================================
REM Step 6: Clean previous build
REM ============================================================================
echo.
echo [6/7] Cleaning previous build...
if exist "dist\360ToolkitGS" rmdir /s /q "dist\360ToolkitGS"
if exist "build\360ToolkitGS" rmdir /s /q "build\360ToolkitGS"
echo [OK] Cleaned

REM ============================================================================
REM Step 7: Build with PyInstaller (GPU spec)
REM ============================================================================
echo.
echo ========================================================================
echo [7/7] Starting PyInstaller build...
echo Using spec: 360ToolkitGS.spec (Full GPU)
echo This may take 10-20 minutes. Please wait...
echo ========================================================================
echo.

python -m PyInstaller 360ToolkitGS.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo BUILD FAILED! Check errors above.
    echo ========================================================================
    pause
    exit /b 1
)

REM ============================================================================
REM Post-build verification
REM ============================================================================
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

REM Verify torch is bundled
echo.
echo Verifying GPU support in build...
if exist "dist\360ToolkitGS\_internal\torch\lib\torch_cuda.dll" (
    echo   [OK] PyTorch CUDA DLLs bundled
) else (
    echo   [WARN] torch_cuda.dll not found in build!
)
if exist "dist\360ToolkitGS\_internal\torch\lib\cudart64_12.dll" (
    echo   [OK] CUDA Runtime bundled
) else (
    echo   [WARN] cudart64_12.dll not found in build!
)

echo.
echo ========================================================================
echo Next steps:
echo   1. Test: dist\360ToolkitGS\360ToolkitGS.exe
echo   2. Create ZIP: create_release_zip.bat
echo   3. Upload to Gumroad
echo ========================================================================
echo.

pause
