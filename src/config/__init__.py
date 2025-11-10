"""
360FrameTools Configuration Module
Provides access to default settings and camera presets.
"""

from .defaults import *

__all__ = [
    # Export all defaults
    'DEFAULT_FPS', 'FPS_MIN', 'FPS_MAX',
    'SDK_QUALITY_OPTIONS', 'DEFAULT_SDK_QUALITY',
    'EXTRACTION_METHODS', 'DEFAULT_EXTRACTION_METHOD',
    'DEFAULT_SPLIT_COUNT', 'DEFAULT_H_FOV',
    'DEFAULT_YAW', 'DEFAULT_PITCH', 'DEFAULT_ROLL',
    'TRANSFORM_TYPES', 'DEFAULT_TRANSFORM_TYPE',
    'MASKING_CATEGORIES', 'YOLOV8_MODELS',
    'DEFAULT_MODEL_SIZE', 'DEFAULT_CONFIDENCE_THRESHOLD',
    'APP_NAME', 'APP_VERSION', 'APP_DESCRIPTION'
]
