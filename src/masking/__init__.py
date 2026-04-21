"""Masking module entry points for the supported YOLO and SAM3.cpp backends."""

import importlib.util
import logging
from pathlib import Path

from src.utils.runtime_backends import has_usable_torch_runtime
from src.utils.resource_path import get_base_path

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = has_usable_torch_runtime()
_ONNX_AVAILABLE = importlib.util.find_spec('onnxruntime') is not None


def _default_onnx_model_path() -> str:
    """Resolve the best bundled ONNX model available for lightweight releases."""
    candidates = [
        'yolo26s-seg.onnx',
        'yolov8s-seg.onnx',
        'yolov8n-seg.onnx',
        'yolov8m-seg.onnx',
    ]
    base_candidates = [get_base_path(), Path.cwd()]

    for base_dir in base_candidates:
        for model_name in candidates:
            candidate = base_dir / model_name
            if candidate.exists():
                return str(candidate)

    return candidates[0]


def get_masker(model_path=None, confidence_threshold=0.5, use_gpu=True, prefer_onnx=None):
    """Return the supported YOLO masker, preferring ONNX when possible."""
    if prefer_onnx is None:
        if model_path and str(model_path).endswith('.onnx'):
            prefer_onnx = True
        else:
            prefer_onnx = not _TORCH_AVAILABLE

    if prefer_onnx or not _TORCH_AVAILABLE:
        if not _ONNX_AVAILABLE:
            raise ImportError(
                "No masking backend available. Install either:\n"
                "  - pip install onnxruntime\n"
                "  - pip install torch ultralytics"
            )
        from .onnx_masker import ONNXMasker
        logger.info("Using ONNX Runtime backend for masking")
        return ONNXMasker(
            model_path=model_path or _default_onnx_model_path(),
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
        )

    from .multi_category_masker import MultiCategoryMasker
    logger.info("Using PyTorch/Ultralytics backend for masking")
    return MultiCategoryMasker(
        model_size='small' if model_path is None else None,
        confidence_threshold=confidence_threshold,
        use_gpu=use_gpu,
    )


def __getattr__(name):
    if name == 'MultiCategoryMasker':
        if _TORCH_AVAILABLE:
            try:
                from .multi_category_masker import MultiCategoryMasker
                return MultiCategoryMasker
            except Exception as exc:
                logger.warning("PyTorch masker import failed (%s). Falling back to ONNXMasker.", exc)
        from .onnx_masker import ONNXMasker
        return ONNXMasker

    if name == 'ONNXMasker':
        from .onnx_masker import ONNXMasker
        return ONNXMasker

    if name == 'SAM3ExternalMasker':
        from .sam3_external_masker import SAM3ExternalMasker
        return SAM3ExternalMasker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['MultiCategoryMasker', 'ONNXMasker', 'SAM3ExternalMasker', 'get_masker']
