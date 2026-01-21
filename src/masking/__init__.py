"""
Masking Module
Multi-category object detection and masking for photogrammetry.

This module provides two backends:
1. ONNXMasker - Lightweight, uses ONNX Runtime (recommended for distribution)
2. MultiCategoryMasker - Full-featured, uses PyTorch/Ultralytics (for development)

The module automatically selects the appropriate backend:
- If PyTorch is available: Uses MultiCategoryMasker (more features)
- If only ONNX Runtime is available: Uses ONNXMasker (smaller binary)
"""

import logging

logger = logging.getLogger(__name__)

# Check which backends are available
_TORCH_AVAILABLE = False
_ONNX_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime
    _ONNX_AVAILABLE = True
except ImportError:
    pass


def get_masker(model_path=None, confidence_threshold=0.5, use_gpu=True, prefer_onnx=None):
    """
    Factory function to get the appropriate masker based on available backends.
    
    Args:
        model_path: Path to model file (.pt for PyTorch, .onnx for ONNX)
        confidence_threshold: Detection confidence threshold (0.0-1.0)
        use_gpu: Enable GPU acceleration if available
        prefer_onnx: If True, prefer ONNX even if PyTorch is available.
                     If None, auto-detect based on model_path extension.
    
    Returns:
        Masker instance (ONNXMasker or MultiCategoryMasker)
    """
    # Auto-detect based on model path
    if prefer_onnx is None:
        if model_path and str(model_path).endswith('.onnx'):
            prefer_onnx = True
        else:
            prefer_onnx = not _TORCH_AVAILABLE
    
    if prefer_onnx or not _TORCH_AVAILABLE:
        if _ONNX_AVAILABLE:
            from .onnx_masker import ONNXMasker
            logger.info("Using ONNX Runtime backend for masking")
            return ONNXMasker(
                model_path=model_path or 'yolov8s-seg.onnx',
                confidence_threshold=confidence_threshold,
                use_gpu=use_gpu
            )
        else:
            raise ImportError(
                "No masking backend available. Install either:\n"
                "  - pip install onnxruntime (lightweight, ~20 MB)\n"
                "  - pip install torch ultralytics (full-featured, ~2 GB)"
            )
    else:
        from .multi_category_masker import MultiCategoryMasker
        logger.info("Using PyTorch/Ultralytics backend for masking")
        return MultiCategoryMasker(
            model_size='small' if model_path is None else None,
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu
        )


# Lazy import for backward compatibility
def __getattr__(name):
    if name == 'MultiCategoryMasker':
        if _TORCH_AVAILABLE:
            from .multi_category_masker import MultiCategoryMasker
            return MultiCategoryMasker
        else:
            # Fallback to ONNX if PyTorch not available
            from .onnx_masker import ONNXMasker
            logger.warning("PyTorch not available, returning ONNXMasker instead of MultiCategoryMasker")
            return ONNXMasker
    
    if name == 'ONNXMasker':
        from .onnx_masker import ONNXMasker
        return ONNXMasker
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['MultiCategoryMasker', 'ONNXMasker', 'get_masker']
