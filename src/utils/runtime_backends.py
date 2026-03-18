"""Helpers for detecting usable runtime backends in dev and frozen builds."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def has_real_module(module_name: str) -> bool:
    """Return True only for real importable modules, not namespace placeholders."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    if spec.loader is None and spec.origin is None:
        return False
    return True


def has_usable_torch_runtime() -> bool:
    """Return True only when torch resolves to a real bundled or installed package."""
    return has_real_module('torch')


def is_usable_torch_module(module: ModuleType) -> bool:
    """Return True when an imported torch module exposes the expected runtime API."""
    return hasattr(module, '__version__') and hasattr(module, 'cuda')


def has_bundled_onnx_runtime() -> bool:
    """Return True when the frozen bundle contains ONNX Runtime package files."""
    if not getattr(sys, 'frozen', False) or not hasattr(sys, '_MEIPASS'):
        return False

    base_path = Path(sys._MEIPASS)
    required = [
        base_path / 'onnxruntime' / 'capi' / 'onnxruntime_pybind11_state.pyd',
        base_path / 'onnxruntime' / 'capi' / 'onnxruntime.dll',
    ]
    return all(path.exists() for path in required)