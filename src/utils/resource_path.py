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
    if getattr(sys, 'frozen', False):
        # Running as bundled .exe
        if hasattr(sys, '_MEIPASS'):
            base_path = Path(sys._MEIPASS)
        else:
            # Fallback for one-dir mode if _MEIPASS is not set
            # Try _internal folder next to executable (PyInstaller 6+)
            base_path = Path(sys.executable).parent / '_internal'
            if not base_path.exists():
                base_path = Path(sys.executable).parent
                
        logger.debug(f"Running as bundled .exe, base path: {base_path}")
    else:
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
    return getattr(sys, 'frozen', False)


def get_base_path() -> Path:
    """
    Get the base directory path.
    
    Returns:
        Path to temp extraction folder (if bundled) or project root (if dev)
    """
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS)
        else:
            base_path = Path(sys.executable).parent / '_internal'
            if base_path.exists():
                return base_path
            return Path(sys.executable).parent
    else:
        return Path(__file__).parent.parent.parent
