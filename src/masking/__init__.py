"""
Masking Module
Multi-category object detection and masking for photogrammetry.
"""

# Lazy import to avoid loading torch during PyInstaller analysis
def __getattr__(name):
    if name == 'MultiCategoryMasker':
        from .multi_category_masker import MultiCategoryMasker
        return MultiCategoryMasker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['MultiCategoryMasker']
