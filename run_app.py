"""
Launcher script for 360toolkit
Ensures proper Python path setup
"""

import os
# Fix OpenMP conflict: PyTorch (libomp.dll) + NumPy/MKL (libiomp5md.dll)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import multiprocessing
from pathlib import Path


def _ensure_workspace_venv_python() -> None:
    """Re-exec with workspace .venv interpreter when launched via `py`/global Python."""
    project_root = Path(__file__).parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"

    if os.name != "nt":
        return
    if not venv_python.exists():
        return
    if os.environ.get("TOOLKIT_SKIP_VENV_REEXEC") == "1":
        return

    current_exe = Path(sys.executable).resolve()
    target_exe = venv_python.resolve()
    if current_exe == target_exe:
        return

    os.environ["TOOLKIT_SKIP_VENV_REEXEC"] = "1"
    os.execv(str(target_exe), [str(target_exe), str(Path(__file__).resolve()), *sys.argv[1:]])


def _preload_torch_before_qt() -> None:
    """
    Import torch before any PyQt6 modules.

    On some Windows setups, importing PyQt6 first can trigger DLL loader
    conflicts and torch later fails with WinError 1114 (c10.dll).
    """
    try:
        import torch  # noqa: F401
    except Exception:
        pass

# CRITICAL: Must be at the very start for PyInstaller on Windows
# Prevents child processes from spawning new GUI windows
if __name__ == '__main__':
    _ensure_workspace_venv_python()
    multiprocessing.freeze_support()

# Add project root to Python path so 'src' can be imported as a package
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    _preload_torch_before_qt()
    # Import main only after torch preload to avoid PyQt->torch DLL conflict
    from src.main import main
    main()
