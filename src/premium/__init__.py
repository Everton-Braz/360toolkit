"""
Premium features package.
WARNING: This directory should NOT be committed to GitHub!
Add to .gitignore: src/premium/
"""

# Lazy imports to avoid DLL crashes from pycolmap at import time
__all__ = ['RigColmapIntegrator']


def __getattr__(name):
    if name == 'RigColmapIntegrator':
        try:
            from .rig_colmap_integration import RigColmapIntegrator
            return RigColmapIntegrator
        except (ImportError, OSError) as e:
            raise ImportError(
                f"RigColmapIntegrator unavailable (pycolmap DLL issue): {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
