"""Pre-build verification script for 360ToolkitGS GPU build"""
import os
import sys
from pathlib import Path

print('='*70)
print('PRE-BUILD VERIFICATION')
print('='*70)

errors = []

# 1. PyTorch
try:
    import torch
    print(f'\n[1] PyTorch: {torch.__version__}')
    has_cuda_tag = 'cu' in torch.__version__
    if not has_cuda_tag:
        errors.append('PyTorch is CPU-only! Need cu128 version.')
    print(f'    CUDA build: {"YES" if has_cuda_tag else "NO (CPU ONLY!)"}')
    print(f'    CUDA available: {torch.cuda.is_available()} (v{torch.version.cuda})')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f'    GPU: {gpu_name}')
        print(f'    Compute: sm_{cc[0]}{cc[1]}')
        try:
            t = torch.tensor([1.0], device='cuda')
            result = (t + 1).item()
            if result == 2.0:
                print('    Kernel test: PASS')
            else:
                print(f'    Kernel test: FAIL (got {result})')
                errors.append('CUDA kernel test failed')
        except Exception as e:
            print(f'    Kernel test: FAIL ({e})')
            errors.append(f'CUDA kernel error: {e}')
        
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        dlls = [f for f in os.listdir(torch_lib) if f.endswith('.dll')]
        print(f'    DLLs in torch/lib: {len(dlls)}')
    else:
        errors.append('CUDA not available')
except ImportError:
    print('\n[1] PyTorch: NOT INSTALLED!')
    errors.append('PyTorch not installed')

# 2. Key packages
print()
try:
    import ultralytics
    print(f'[2] Ultralytics: {ultralytics.__version__}')
except ImportError:
    print('[2] Ultralytics: NOT INSTALLED')
    errors.append('Ultralytics not installed')

try:
    import PyQt6
    print('[3] PyQt6: OK')
except ImportError:
    print('[3] PyQt6: NOT INSTALLED')
    errors.append('PyQt6 not installed')

try:
    import segment_anything
    print('[4] SAM: OK')
except ImportError:
    print('[4] SAM: not installed (optional)')

try:
    import PyInstaller
    print(f'[5] PyInstaller: {PyInstaller.__version__}')
except ImportError:
    print('[5] PyInstaller: NOT INSTALLED')
    errors.append('PyInstaller not installed')

# 3. SDK
sdk = Path(r'C:\Users\Everton-PC\Documents\Windows_CameraSDK-2.1.1_MediaSDK-3.1.0\MediaSDK-3.1.0.0-20250904-win64\MediaSDK')
print(f'\n[6] SDK: {"OK" if sdk.exists() else "MISSING"}')
if sdk.exists():
    print(f'    bin/: {"OK" if (sdk/"bin").exists() else "MISSING"}')
    print(f'    models/: {"OK" if (sdk/"models").exists() else "MISSING"}')

# 4. FFmpeg
ff = Path(r'C:\Users\Everton-PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe')
print(f'\n[7] FFmpeg: {"OK" if ff.exists() else "MISSING"}')

# 5. Models
print('\n[8] Models:')
for m in ['yolov8m-seg.pt', 'yolov8m.pt', 'yolov8n-seg.pt', 'sam_vit_b_01ec64.pth']:
    print(f'    {m}: {"OK" if os.path.exists(m) else "missing (will auto-download)"}')

# 6. Environment
print(f'\n[9] CONDA_PREFIX: {os.environ.get("CONDA_PREFIX", "NOT SET")}')
conda = os.environ.get('CONDA_PREFIX', '')
if conda:
    pyqt6_path = Path(conda) / 'Lib' / 'site-packages' / 'PyQt6'
    print(f'    PyQt6 in env: {"OK" if pyqt6_path.exists() else "MISSING"}')

# 7. Spec file
spec = Path('360ToolkitGS.spec')
print(f'\n[10] Spec file: {"OK" if spec.exists() else "MISSING"}')

# Summary
print(f'\n{"="*70}')
if errors:
    print(f'ERRORS FOUND ({len(errors)}):')
    for e in errors:
        print(f'  - {e}')
    print('\nFIX ERRORS BEFORE BUILDING!')
    sys.exit(1)
else:
    print('ALL CHECKS PASSED - Ready to build!')
    print('Run: python -m PyInstaller 360ToolkitGS.spec --noconfirm')
print('='*70)
