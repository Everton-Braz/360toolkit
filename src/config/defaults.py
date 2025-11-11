"""
360FrameTools - Default Configuration Parameters
Central location for all default settings across the pipeline.
"""

# ============================================================================
# STAGE 1: FRAME EXTRACTION DEFAULTS
# ============================================================================

# Extraction parameters
DEFAULT_FPS = 2.0  # Frames per second extraction rate (CHANGED: 2 FPS)
FPS_MIN = 0.1
FPS_MAX = 30.0

# SDK quality settings
SDK_QUALITY_OPTIONS = ['draft', 'good', 'best']
DEFAULT_SDK_QUALITY = 'good'

# Extraction methods (SDK is PRIMARY - Best Quality)
EXTRACTION_METHODS = {
    'sdk_stitching': 'SDK Stitching (Best Quality - RECOMMENDED)',
    'ffmpeg_stitched': 'FFmpeg Stitched (Good for pre-stitched MP4)',
    'ffmpeg_dual_lens': 'FFmpeg Dual-Lens (Both Lenses Separately)',
    'ffmpeg_lens1': 'FFmpeg Lens 1 Only (Front Fisheye)',
    'ffmpeg_lens2': 'FFmpeg Lens 2 Only (Back Fisheye)',
    'opencv_dual_lens': 'OpenCV Dual-Lens (Both Lenses Separately)',
    'opencv_lens1': 'OpenCV Lens 1 Only (Front Fisheye)',
    'opencv_lens2': 'OpenCV Lens 2 Only (Back Fisheye)'
}
# SDK is PRIMARY method (use FFmpeg only for already-stitched MP4 files)
DEFAULT_EXTRACTION_METHOD = 'sdk_stitching'

# Lens extraction options
LENS_OPTIONS = {
    'both': 'Extract Both Lenses',
    'lens1': 'Extract Lens 1 Only (Front)',
    'lens2': 'Extract Lens 2 Only (Back)'
}

# Output formats
SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'tiff', 'tif']
DEFAULT_OUTPUT_FORMAT = 'png'

# File handling
DELETE_INTERMEDIATE_FILES = False

# ============================================================================
# STAGE 2: PERSPECTIVE SPLITTING DEFAULTS
# ============================================================================

# Camera configuration
DEFAULT_SPLIT_COUNT = 8  # Number of cameras in horizontal ring
SPLIT_COUNT_MIN = 1
SPLIT_COUNT_MAX = 12

# Field of view
DEFAULT_H_FOV = 110  # Horizontal field of view in degrees (CHANGED: 110°)
H_FOV_MIN = 30
H_FOV_MAX = 150

# Default camera orientation (degrees)
DEFAULT_YAW = 0  # -180 to 180 (0° = front)
DEFAULT_PITCH = 0  # -90 to +90 (0° = horizon)
DEFAULT_ROLL = 0  # -180 to 180 (typically 0°)

# Transform types
TRANSFORM_TYPES = {
    'perspective': 'Perspective (E2P)',
    'cubemap_6face': 'Cubemap (6-face)',
    'cubemap_8tile': 'Cubemap (8-tile)'
}
DEFAULT_TRANSFORM_TYPE = 'cubemap_8tile'  # CHANGED: Default is now 8-tile cubemap

# Output dimensions (SQUARE for photogrammetry)
DEFAULT_OUTPUT_WIDTH = 1440
DEFAULT_OUTPUT_HEIGHT = 1440

# Cubemap settings
DEFAULT_CUBEMAP_FACE_SIZE = 1024
DEFAULT_CUBEMAP_OVERLAP = 10  # Overlap percentage (0-50)

# Camera ring presets
DEFAULT_COMPASS_RINGS = [
    {
        'name': 'main',
        'pitch': 0,
        'camera_count': 8,
        'fov': 110
    }
]

# Camera states (for UI)
CAMERA_STATE_EXPORT = 0  # Blue - will be exported
CAMERA_STATE_PREVIEW = 1  # Yellow - preview only
CAMERA_STATE_DISABLED = 2  # Red - disabled/skip
CAMERA_STATE_MASK = 3  # Green - export with person masking

CAMERA_STATE_NAMES = {
    CAMERA_STATE_EXPORT: 'Export',
    CAMERA_STATE_PREVIEW: 'Preview',
    CAMERA_STATE_DISABLED: 'Disabled',
    CAMERA_STATE_MASK: 'Export with Mask'
}

# ============================================================================
# STAGE 3: MASKING DEFAULTS
# ============================================================================

# COCO dataset class IDs
COCO_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    'traffic_light': 9,
    'fire_hydrant': 10,
    'stop_sign': 11,
    'parking_meter': 12,
    'bench': 13,
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'backpack': 24,
    'umbrella': 25,
    'handbag': 26,
    'tie': 27,
    'suitcase': 28,
    'cell_phone': 67,
    'laptop': 63,
    'keyboard': 66,
    'mouse': 64
}

# Masking categories configuration
MASKING_CATEGORIES = {
    'persons': {
        'name': 'Persons',
        'classes': [0],  # person
        'enabled': True  # CHANGED: Persons enabled by default
    },
    'personal_objects': {
        'name': 'Personal Objects',
        'classes': [24, 26, 28, 67],  # backpack, handbag, suitcase, cell phone
        'enabled': True  # CHANGED: Personal objects enabled by default
    },
    'animals': {
        'name': 'Animals',
        'classes': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # bird through giraffe
        'enabled': False  # CHANGED: Animals disabled by default
    }
}

# YOLOv8 model settings
YOLOV8_MODELS = {
    'nano': {
        'filename': 'yolov8n-seg.pt',
        'size_mb': 7,
        'speed_seconds': 0.2,
        'accuracy': 85
    },
    'small': {
        'filename': 'yolov8s-seg.pt',
        'size_mb': 23,
        'speed_seconds': 0.5,
        'accuracy': 90
    },
    'medium': {
        'filename': 'yolov8m-seg.pt',
        'size_mb': 52,
        'speed_seconds': 1.0,
        'accuracy': 92
    },
    'large': {
        'filename': 'yolov8l-seg.pt',
        'size_mb': 83,
        'speed_seconds': 1.5,
        'accuracy': 94
    },
    'xlarge': {
        'filename': 'yolov8x-seg.pt',
        'size_mb': 136,
        'speed_seconds': 2.5,
        'accuracy': 95
    }
}

DEFAULT_MODEL_SIZE = 'medium'  # CHANGED: Default model is medium
DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # CHANGED: Confidence threshold is 0.60
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# GPU/Device settings
DEFAULT_USE_GPU = True  # Auto-detect and use if available
DEFAULT_BATCH_SIZE = 4  # Images per GPU batch
BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 16

# Mask output settings
MASK_SUFFIX = '_mask'
MASK_FORMAT = 'png'
MASK_VALUE_REMOVE = 0  # Black (0) = mask/remove region
MASK_VALUE_KEEP = 255  # White (255) = keep/valid region

# ============================================================================
# PIPELINE & PERFORMANCE
# ============================================================================

# Cache settings
DEFAULT_CACHE_SIZE_MB = 2048
CACHE_SIZE_MIN = 512
CACHE_SIZE_MAX = 8192

# Threading
DEFAULT_THREAD_COUNT = 'auto'  # Auto-detect optimal thread count

# Temp folder
import os
DEFAULT_TEMP_FOLDER = os.path.join(os.environ.get('TEMP', 'C:\\Temp'), '360FrameTools')

# Logging
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
DEFAULT_LOG_LEVEL = 'INFO'

# ============================================================================
# UI SETTINGS
# ============================================================================

# Theme
THEMES = ['dark', 'light', 'system']
DEFAULT_THEME = 'dark'

# Window dimensions
WINDOW_MIN_WIDTH = 1280
WINDOW_MIN_HEIGHT = 800
WINDOW_DEFAULT_WIDTH = 1400
WINDOW_DEFAULT_HEIGHT = 900

# Compass widget
COMPASS_WIDGET_SIZE = 400  # 400x400 px

# File dialog filters
FILE_DIALOG_FILTERS = {
    'video': 'Video Files (*.insv *.mp4 *.mov)',
    'image': 'Image Files (*.jpg *.jpeg *.png *.tiff *.tif)',
    'all': 'All Files (*.*)'
}

# ============================================================================
# METADATA SETTINGS
# ============================================================================

# Metadata preservation (NO GPS/GYRO as per spec)
PRESERVE_CAMERA_METADATA = True
PRESERVE_GPS_METADATA = False  # Explicitly disabled
PRESERVE_GYRO_METADATA = False  # Explicitly disabled

# Camera metadata fields to preserve
CAMERA_METADATA_FIELDS = [
    'Make',
    'Model',
    'LensModel',
    'FocalLength',
    'FocalLengthIn35mmFilm',
    'FNumber',
    'ExposureTime',
    'ISO',
    'WhiteBalance',
    'DateTimeOriginal'
]

# Custom EXIF fields for camera orientation (Stage 2)
EXIF_CAMERA_YAW_TAG = 'UserComment'  # Embed yaw in UserComment
EXIF_CAMERA_PITCH_TAG = 'ImageDescription'  # Embed pitch
EXIF_CAMERA_ROLL_TAG = 'Software'  # Embed roll

# ============================================================================
# PROJECT FILE SETTINGS
# ============================================================================

PROJECT_FILE_EXTENSION = '.360ft'
PROJECT_FILE_VERSION = '1.0'

# ============================================================================
# VERSION INFO
# ============================================================================

APP_NAME = '360FrameTools'
APP_VERSION = '1.0.0'
APP_AUTHOR = '360FrameTools Development Team'
APP_DESCRIPTION = 'Unified photogrammetry preprocessing pipeline: Extract → Split → Mask'
