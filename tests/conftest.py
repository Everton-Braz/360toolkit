import importlib
import os
import sys
from pathlib import Path


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _add_windows_dll_dirs() -> None:
    if os.name != "nt":
        return

    candidate_dirs = [
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
        Path(sys.prefix) / "Lib" / "site-packages" / "onnxruntime" / "capi",
    ]

    for dll_dir in candidate_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass


def pytest_sessionstart(session) -> None:
    _add_windows_dll_dirs()

    for module_name in ("torch", "onnxruntime", "ultralytics"):
        try:
            importlib.import_module(module_name)
        except Exception:
            # Some CI/dev environments intentionally run without these optional deps.
            pass
