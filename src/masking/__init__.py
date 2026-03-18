"""
Masking Module
Multi-category object detection and masking for photogrammetry.

This module provides three backends:
1. ONNXMasker - Lightweight YOLO, uses ONNX Runtime (recommended for distribution)
2. MultiCategoryMasker - Full-featured YOLO, uses PyTorch/Ultralytics (for development)
3. SAMMasker - SAM ViT-B for superior segmentation quality (best results)

The module automatically selects the appropriate backend:
- If PyTorch is available: Uses MultiCategoryMasker or SAMMasker
- If only ONNX Runtime is available: Uses ONNXMasker (smaller binary)
"""

import logging
import importlib.util

from src.utils.runtime_backends import has_usable_torch_runtime

logger = logging.getLogger(__name__)

# Check which backends are available
_TORCH_AVAILABLE = False
_ONNX_AVAILABLE = False
_SAM_AVAILABLE = False

_TORCH_AVAILABLE = has_usable_torch_runtime()
_ONNX_AVAILABLE = importlib.util.find_spec('onnxruntime') is not None
_SAM_AVAILABLE = importlib.util.find_spec('segment_anything') is not None
_YOLO_AVAILABLE = importlib.util.find_spec('ultralytics') is not None

# Hybrid masker availability (requires both SAM and YOLO)
_HYBRID_AVAILABLE = _SAM_AVAILABLE and _YOLO_AVAILABLE


def get_masker(model_path=None, confidence_threshold=0.5, use_gpu=True, prefer_onnx=None, use_sam=False, use_hybrid=False):
    """
    Factory function to get the appropriate masker based on available backends.
    
    Args:
        model_path: Path to model file (.pt for PyTorch, .onnx for ONNX, .pth for SAM)
        confidence_threshold: Detection confidence threshold (0.0-1.0)
        use_gpu: Enable GPU acceleration if available
        prefer_onnx: If True, prefer ONNX even if PyTorch is available.
                     If None, auto-detect based on model_path extension.
        use_sam: If True, use SAM ViT-B instead of YOLO (requires segment-anything)
        use_hybrid: If True, use YOLO+SAM hybrid (requires both)
    
    Returns:
        Masker instance (HybridYOLOSAMMasker, SAMMasker, ONNXMasker, or MultiCategoryMasker)
    """
    # Check for hybrid first if requested
    if use_hybrid:
        if _HYBRID_AVAILABLE:
            from .hybrid_yolo_sam_masker import HybridYOLOSAMMasker
            logger.info("Using YOLO+SAM hybrid backend for masking")
            return HybridYOLOSAMMasker(
                yolo_model=model_path or 'yolov8m.pt',
                sam_checkpoint='sam_vit_b_01ec64.pth',
                use_gpu=use_gpu,
                yolo_confidence=confidence_threshold
            )
        else:
            logger.warning("Hybrid masker not available (requires both SAM and YOLO). Falling back to SAM or YOLO.")
            use_sam = _SAM_AVAILABLE  # Try SAM if available
    
    # Check for SAM if requested
    if use_sam:
        if _SAM_AVAILABLE:
            from .sam_masker import SAMMasker
            logger.info("Using SAM ViT-B backend for masking")
            return SAMMasker(
                model_checkpoint=model_path or 'sam_vit_b_01ec64.pth',
                use_gpu=use_gpu
            )
        else:
            logger.warning("SAM not available (pip install segment-anything). Falling back to YOLO.")
    
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
            try:
                from .multi_category_masker import MultiCategoryMasker
                return MultiCategoryMasker
            except Exception as exc:
                logger.warning("PyTorch masker import failed (%s). Falling back to ONNXMasker.", exc)
        # Fallback to ONNX if PyTorch is unavailable or import failed
        from .onnx_masker import ONNXMasker
        return ONNXMasker
    
    if name == 'ONNXMasker':
        from .onnx_masker import ONNXMasker
        return ONNXMasker
    
    if name == 'SAMMasker':
        if _SAM_AVAILABLE:
            try:
                from .sam_masker import SAMMasker
                return SAMMasker
            except Exception as exc:
                raise ImportError(f"SAM backend failed to load: {exc}") from exc
        raise ImportError("SAM not available. Install with: pip install segment-anything")
    
    if name == 'HybridYOLOSAMMasker':
        if _HYBRID_AVAILABLE:
            from .hybrid_yolo_sam_masker import HybridYOLOSAMMasker
            return HybridYOLOSAMMasker
        else:
            raise ImportError("Hybrid masker not available. Install with: pip install segment-anything ultralytics")
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['MultiCategoryMasker', 'ONNXMasker', 'SAMMasker', 'HybridYOLOSAMMasker', 'get_masker']
