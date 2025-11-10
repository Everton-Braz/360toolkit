"""
Resource path helper for PyInstaller bundled applications
"""
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    
    When running as .exe, PyInstaller extracts resources to a temp folder
    and stores the path in sys._MEIPASS. This function handles both cases.
    
    Args:
        relative_path: Relative path to resource
            Examples:
            - 'sdk/bin/MediaSDKTest.exe'
            - 'ffmpeg/ffmpeg.exe'
            - 'models/yolov8n-seg.pt'
    
    Returns:
        Absolute Path object to the resource
    
    Example:
        >>> sdk_exe = get_resource_path('sdk/bin/MediaSDKTest.exe')
        >>> if sdk_exe.exists():
        ...     print(f"Found SDK at: {sdk_exe}")
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # This is where all bundled files are extracted at runtime
        base_path = Path(sys._MEIPASS)
        logger.debug(f"Running as bundled .exe, base path: {base_path}")
    except AttributeError:
        # Running in development mode (not bundled)
        # Use project root directory
        base_path = Path(__file__).parent.parent.parent
        logger.debug(f"Running in development mode, base path: {base_path}")
    
    resource_path = base_path / relative_path
    logger.debug(f"Resource path for '{relative_path}': {resource_path}")
    
    return resource_path


def is_bundled() -> bool:
    """
    Check if application is running as PyInstaller bundle.
    
    Returns:
        True if running as .exe, False if running as Python script
    """
    return hasattr(sys, '_MEIPASS')


def get_base_path() -> Path:
    """
    Get the base directory path.
    
    Returns:
        Path to temp extraction folder (if bundled) or project root (if dev)
    """
    try:
        return Path(sys._MEIPASS)
    except AttributeError:
        return Path(__file__).parent.parent.parent
