"""
Feature Flags for 360FrameTools.
Controls premium vs free features.
"""

import os
import logging

logger = logging.getLogger(__name__)


class FeatureFlags:
    """Feature flags manager for premium/free edition control."""
    
    # Check if premium build
    _is_premium = os.environ.get('TOOLKIT_PREMIUM_BUILD', 'false').lower() == 'true'
    
    # Class attribute for direct access (compatibility)
    IS_PREMIUM = _is_premium
    
    @classmethod
    def is_premium(cls) -> bool:
        """Check if this is a premium build."""
        return cls._is_premium
    
    @classmethod
    def enable_premium(cls):
        """Enable premium features (for testing)."""
        cls._is_premium = True
        os.environ['TOOLKIT_PREMIUM_BUILD'] = 'true'
    
    @classmethod
    def disable_premium(cls):
        """Disable premium features."""
        cls._is_premium = False
        os.environ['TOOLKIT_PREMIUM_BUILD'] = 'false'
    
    # Feature checks
    @classmethod
    def has_colmap_alignment(cls) -> bool:
        """Check if COLMAP alignment is available."""
        return cls._is_premium
    
    @classmethod
    def has_glomap(cls) -> bool:
        """Check if GloMAP (fast global SfM) is available."""
        return cls._is_premium
    
    @classmethod
    def has_realityscan_export(cls) -> bool:
        """Check if RealityScan/RealityCapture export is available."""
        return cls._is_premium
    
    @classmethod
    def has_lichtfeld_export(cls) -> bool:
        """Check if Lichtfeld Studio 3DGS export is available."""
        return cls._is_premium
    
    @classmethod
    def has_advanced_masking(cls) -> bool:
        """Check if advanced masking (animals, objects) is available."""
        return cls._is_premium
    
    @classmethod
    def has_8k_output(cls) -> bool:
        """Check if 8K output resolution is available."""
        return cls._is_premium
    
    @classmethod
    def has_batch_processing(cls) -> bool:
        """Batch processing is always available."""
        return True
    
    @classmethod
    def has_gpu_acceleration(cls) -> bool:
        """GPU acceleration is always available."""
        return True
    
    @classmethod
    def get_enabled_features(cls) -> list:
        """Get list of enabled features."""
        features = [
            ('Batch Processing', cls.has_batch_processing()),
            ('GPU Acceleration', cls.has_gpu_acceleration()),
            ('COLMAP Alignment', cls.has_colmap_alignment()),
            ('GloMAP (Fast SfM)', cls.has_glomap()),
            ('RealityScan Export', cls.has_realityscan_export()),
            ('Lichtfeld Export', cls.has_lichtfeld_export()),
            ('Advanced Masking', cls.has_advanced_masking()),
            ('8K Output', cls.has_8k_output()),
        ]
        return [(name, enabled) for name, enabled in features]


# Log feature status on import
if FeatureFlags.is_premium():
    logger.info("360FrameTools Premium Edition - All features enabled")
else:
    logger.info("360FrameTools Free Edition - Some features limited")
