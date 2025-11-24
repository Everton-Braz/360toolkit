@echo off
call C:\Users\User\miniconda3\Scripts\activate.bat 360toolkit-cpu

echo Testing PyTorch version check...
python -c "import torch, sys; print('Version:', torch.__version__); sys.exit(1 if '+cpu' not in torch.__version__ else 0)"
echo ERRORLEVEL is: %ERRORLEVEL%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Not CPU version
) else (
    echo SUCCESS: CPU version detected
)
pause
