"""
360FrameTools - Settings Manager
Handles user preferences, path detection, and configuration persistence.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Manages application settings including SDK and FFmpeg paths.
    
    Features:
    - Automatic path detection for SDK and FFmpeg
    - User-configurable path overrides
    - Settings persistence (JSON file)
    - Validation of paths and executables
    """
    
    def __init__(self, settings_file: Optional[Path] = None):
        """
        Initialize settings manager.
        
        Args:
            settings_file: Path to settings JSON file (default: config/user_settings.json)
        """
        if settings_file is None:
            settings_file = Path(__file__).parent / "user_settings.json"
        
        self.settings_file = Path(settings_file)
        self.settings = self._load_settings()
        
        # Auto-detect paths if not configured
        if not self.settings.get('sdk_path') or not self.is_sdk_valid(self.settings.get('sdk_path')):
            logger.info("SDK path not configured or invalid, running auto-detection...")
            detected_sdk = self.auto_detect_sdk()
            if detected_sdk:
                self.settings['sdk_path'] = str(detected_sdk)
                self.settings['sdk_auto_detected'] = True
        
        if not self.settings.get('ffmpeg_path') or not self.is_ffmpeg_valid(self.settings.get('ffmpeg_path')):
            logger.info("FFmpeg path not configured or invalid, running auto-detection...")
            detected_ffmpeg = self.auto_detect_ffmpeg()
            if detected_ffmpeg:
                self.settings['ffmpeg_path'] = str(detected_ffmpeg)
                self.settings['ffmpeg_auto_detected'] = True
    
    def _load_settings(self) -> Dict:
        """Load settings from JSON file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
        
        # Return defaults
        return {
            'sdk_path': None,
            'ffmpeg_path': None,
            'yolo_model_path': None,
            'auto_detect_on_startup': True,
            'output_format': 'PNG',
            'default_fps': 1.0,
            'default_h_fov': 110,
            'default_split_count': 8,
            'yolo_model_size': 'medium'
        }
    
    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            logger.info(f"Settings saved to {self.settings_file}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    # === SDK PATH MANAGEMENT ===
    
    def auto_detect_sdk(self) -> Optional[Path]:
        """
        Auto-detect Insta360 MediaSDK installation.
        
        Search order:
        1. C:/Users/<User>/Documents/Windows_CameraSDK*/MediaSDK*/
        2. C:/Program Files/Insta360/MediaSDK/
        3. ../sdk/ relative to app
        
        Returns:
            Path to MediaSDK folder if found, None otherwise
        """
        logger.info("Searching for Insta360 MediaSDK...")
        
        # 1. User Documents folder
        docs = Path.home() / "Documents"
        for sdk_dir in docs.glob("Windows_CameraSDK*"):
            for media_sdk in sdk_dir.glob("MediaSDK*"):
                if self.is_sdk_valid(media_sdk):
                    logger.info(f"[OK] SDK found at {media_sdk}")
                    return media_sdk
        
        # 2. Program Files
        program_files = [
            Path("C:/Program Files/Insta360/MediaSDK"),
            Path("C:/Program Files (x86)/Insta360/MediaSDK")
        ]
        for pf in program_files:
            if self.is_sdk_valid(pf):
                logger.info(f"[OK] SDK found at {pf}")
                return pf
        
        # 3. Relative to app
        app_dir = Path(__file__).parent.parent.parent
        relative_sdk = app_dir / "sdk"
        if self.is_sdk_valid(relative_sdk):
            logger.info(f"[OK] SDK found at {relative_sdk}")
            return relative_sdk
        
        logger.warning("[X] SDK not found in any standard location")
        return None
    
    def is_sdk_valid(self, sdk_path: Optional[Path]) -> bool:
        """
        Validate SDK path by checking for required DLLs.
        
        Required files:
        - CameraSDK.dll or MediaSDK.dll
        - bin/ folder with DLLs
        """
        if sdk_path is None:
            return False
        
        sdk_path = Path(sdk_path)
        if not sdk_path.exists() or not sdk_path.is_dir():
            return False
        
        # Check for key DLL files
        dll_files = [
            sdk_path / "CameraSDK.dll",
            sdk_path / "MediaSDK.dll",
            sdk_path / "bin" / "CameraSDK.dll",
            sdk_path / "bin" / "MediaSDK.dll"
        ]
        
        return any(dll.exists() for dll in dll_files)
    
    def set_sdk_path(self, path: Optional[Path], auto_detected: bool = False) -> bool:
        """Set SDK path and save. Returns True if valid, False otherwise."""
        if path is None:
            self.settings['sdk_path'] = None
            self.settings['sdk_auto_detected'] = False
            self.save_settings()
            logger.info("SDK path cleared")
            return True
        
        if not self.is_sdk_valid(path):
            logger.warning(f"Invalid SDK path: {path}")
            return False
        
        self.settings['sdk_path'] = str(path)
        self.settings['sdk_auto_detected'] = auto_detected
        self.save_settings()
        logger.info(f"SDK path set to {path}")
        return True
    
    def get_sdk_path(self) -> Optional[Path]:
        """Get current SDK path"""
        sdk_str = self.settings.get('sdk_path')
        if sdk_str:
            return Path(sdk_str)
        return None
    
    # === FFMPEG PATH MANAGEMENT ===
    
    def auto_detect_ffmpeg(self) -> Optional[Path]:
        """
        Auto-detect FFmpeg installation.
        
        Search order:
        1. System PATH (shutil.which)
        2. C:/ffmpeg/bin/ffmpeg.exe
        3. WinGet installed location (C:/Users/<User>/AppData/Local/Microsoft/WinGet/Packages/...)
        4. ../ffmpeg/ relative to app
        
        Returns:
            Path to ffmpeg.exe if found, None otherwise
        """
        logger.info("Searching for FFmpeg...")
        
        # 1. System PATH
        ffmpeg_in_path = shutil.which('ffmpeg')
        if ffmpeg_in_path:
            ffmpeg_path = Path(ffmpeg_in_path)
            if self.is_ffmpeg_valid(ffmpeg_path):
                logger.info(f"[OK] FFmpeg found in PATH: {ffmpeg_path}")
                return ffmpeg_path
        
        # 2. Standard installation
        standard_paths = [
            Path("C:/ffmpeg/bin/ffmpeg.exe"),
            Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe")
        ]
        for path in standard_paths:
            if self.is_ffmpeg_valid(path):
                logger.info(f"[OK] FFmpeg found at {path}")
                return path
        
        # 3. WinGet packages location
        winget_base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if winget_base.exists():
            for pkg_dir in winget_base.glob("*FFmpeg*"):
                for ffmpeg_exe in pkg_dir.rglob("ffmpeg.exe"):
                    if self.is_ffmpeg_valid(ffmpeg_exe):
                        logger.info(f"[OK] FFmpeg found at {ffmpeg_exe}")
                        return ffmpeg_exe
        
        # 4. Relative to app
        app_dir = Path(__file__).parent.parent.parent
        relative_ffmpeg = app_dir / "ffmpeg" / "bin" / "ffmpeg.exe"
        if self.is_ffmpeg_valid(relative_ffmpeg):
            logger.info(f"[OK] FFmpeg found at {relative_ffmpeg}")
            return relative_ffmpeg
        
        logger.warning("[X] FFmpeg not found in any standard location")
        return None
    
    def is_ffmpeg_valid(self, ffmpeg_path: Optional[Path]) -> bool:
        """Validate FFmpeg path by checking executable exists"""
        if ffmpeg_path is None:
            return False
        
        ffmpeg_path = Path(ffmpeg_path)
        if ffmpeg_path.is_dir():
            # If directory provided, look for ffmpeg.exe inside
            ffmpeg_path = ffmpeg_path / "ffmpeg.exe"
        
        return ffmpeg_path.exists() and ffmpeg_path.is_file()
    
    def set_ffmpeg_path(self, path: Optional[Path], auto_detected: bool = False) -> bool:
        """Set FFmpeg path and save. Returns True if valid, False otherwise."""
        if path is None:
            self.settings['ffmpeg_path'] = None
            self.settings['ffmpeg_auto_detected'] = False
            self.save_settings()
            logger.info("FFmpeg path cleared")
            return True
        
        if not self.is_ffmpeg_valid(path):
            logger.warning(f"Invalid FFmpeg path: {path}")
            return False
        
        self.settings['ffmpeg_path'] = str(path)
        self.settings['ffmpeg_auto_detected'] = auto_detected
        self.save_settings()
        logger.info(f"FFmpeg path set to {path}")
        return True
    
    def get_ffmpeg_path(self) -> Optional[Path]:
        """Get current FFmpeg path"""
        ffmpeg_str = self.settings.get('ffmpeg_path')
        if ffmpeg_str:
            return Path(ffmpeg_str)
        return None
    
    # === YOLO MODEL PATH MANAGEMENT ===
    
    def auto_detect_yolo_model(self) -> Optional[Path]:
        """
        Auto-detect YOLOv8 model file.
        
        Search order:
        1. App directory (yolov8*.pt or yolov8*.onnx)
        2. User's .cache/ultralytics/models/
        3. Standard Ultralytics cache location
        
        Returns:
            Path to YOLO model if found, None otherwise
        """
        logger.info("Searching for YOLO model...")
        
        # 1. App directory
        app_dir = Path(__file__).parent.parent.parent
        for pattern in ['yolov8*.pt', 'yolov8*.onnx']:
            for model_file in app_dir.glob(pattern):
                if self.is_yolo_model_valid(model_file):
                    logger.info(f"[OK] YOLO model found at {model_file}")
                    return model_file
        
        # 2. User cache directory
        cache_dir = Path.home() / ".cache" / "ultralytics" / "models"
        if cache_dir.exists():
            for pattern in ['yolov8*.pt', 'yolov8*.onnx']:
                for model_file in cache_dir.glob(pattern):
                    if self.is_yolo_model_valid(model_file):
                        logger.info(f"[OK] YOLO model found at {model_file}")
                        return model_file
        
        logger.warning("[X] YOLO model not found in any standard location")
        return None
    
    def is_yolo_model_valid(self, model_path: Optional[Path]) -> bool:
        """Validate YOLO model path by checking file exists"""
        if model_path is None:
            return False
        
        model_path = Path(model_path)
        if not model_path.exists() or not model_path.is_file():
            return False
        
        # Check file extension
        valid_extensions = ['.pt', '.onnx', '.engine']
        return model_path.suffix.lower() in valid_extensions
    
    def set_yolo_model_path(self, path: Optional[Path], auto_detected: bool = False) -> bool:
        """Set YOLO model path and save. Returns True if valid, False otherwise."""
        if path is None:
            self.settings['yolo_model_path'] = None
            self.settings['yolo_model_auto_detected'] = False
            self.save_settings()
            logger.info("YOLO model path cleared")
            return True
        
        if not self.is_yolo_model_valid(path):
            logger.warning(f"Invalid YOLO model path: {path}")
            return False
        
        self.settings['yolo_model_path'] = str(path)
        self.settings['yolo_model_auto_detected'] = auto_detected
        self.save_settings()
        logger.info(f"YOLO model path set to {path}")
        return True
    
    def get_yolo_model_path(self) -> Optional[Path]:
        """Get current YOLO model path"""
        model_str = self.settings.get('yolo_model_path')
        if model_str:
            return Path(model_str)
        return None
    
    def get_yolo_model_size(self) -> str:
        """Get YOLO model size setting"""
        return self.settings.get('yolo_model_size', 'medium')
    
    def set_yolo_model_size(self, size: str):
        """Set YOLO model size (nano/small/medium/large/xlarge)"""
        valid_sizes = ['nano', 'small', 'medium', 'large', 'xlarge']
        if size not in valid_sizes:
            logger.warning(f"Invalid model size: {size}. Using 'medium'.")
            size = 'medium'
        
        self.settings['yolo_model_size'] = size
        self.save_settings()
    
    # === GENERAL SETTINGS ===
    
    def get(self, key: str, default=None):
        """Get a setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value):
        """Set a setting value and save"""
        self.settings[key] = value
        self.save_settings()
    
    def get_all(self) -> Dict:
        """Get all settings"""
        return self.settings.copy()
    
    def get_auto_detect_on_startup(self) -> bool:
        """Get auto-detect on startup setting"""
        return self.settings.get('auto_detect_on_startup', True)
    
    def set_auto_detect_on_startup(self, enabled: bool):
        """Set auto-detect on startup setting"""
        self.settings['auto_detect_on_startup'] = enabled
        self.save_settings()
    
    def get_last_input_directory(self) -> Optional[Path]:
        """Get last used input directory"""
        path_str = self.settings.get('last_input_directory')
        if path_str:
            return Path(path_str)
        return None
    
    def set_last_input_directory(self, path: Path):
        """Set last used input directory"""
        self.settings['last_input_directory'] = str(path)
        self.save_settings()
    
    def get_last_output_directory(self) -> Optional[Path]:
        """Get last used output directory"""
        path_str = self.settings.get('last_output_directory')
        if path_str:
            return Path(path_str)
        return None
    
    def set_last_output_directory(self, path: Path):
        """Set last used output directory"""
        self.settings['last_output_directory'] = str(path)
        self.save_settings()
    
    def get_recent_files(self) -> List[str]:
        """Get list of recent files"""
        return self.settings.get('recent_files', [])
    
    def add_recent_file(self, file_path: Path):
        """Add file to recent files list"""
        recent = self.get_recent_files()
        file_str = str(file_path)
        
        # Remove if already exists
        if file_str in recent:
            recent.remove(file_str)
        
        # Add to front
        recent.insert(0, file_str)
        
        # Keep only 10 most recent
        recent = recent[:10]
        
        self.settings['recent_files'] = recent
        self.save_settings()
    
    def get_sdk_info(self, sdk_path: Optional[Path]) -> Dict:
        """
        Get detailed information about SDK installation.
        
        Returns:
            Dictionary with SDK info (version, files, etc.)
        """
        if sdk_path is None or not self.is_sdk_valid(sdk_path):
            return {'valid': False, 'error': 'Invalid or missing SDK path'}
        
        sdk_path = Path(sdk_path)
        info = {
            'valid': True,
            'path': str(sdk_path),
            'dll_files': []
        }
        
        # Find DLL files
        for dll in sdk_path.rglob("*.dll"):
            info['dll_files'].append(dll.name)
        
        return info
    
    def get_ffmpeg_info(self, ffmpeg_path: Optional[Path]) -> Dict:
        """
        Get detailed information about FFmpeg installation.
        
        Returns:
            Dictionary with FFmpeg info (version, etc.)
        """
        if ffmpeg_path is None or not self.is_ffmpeg_valid(ffmpeg_path):
            return {'valid': False, 'error': 'Invalid or missing FFmpeg path'}
        
        ffmpeg_path = Path(ffmpeg_path)
        info = {
            'valid': True,
            'path': str(ffmpeg_path),
            'executable': ffmpeg_path.name
        }
        
        # Try to get version
        try:
            import subprocess
            result = subprocess.run(
                [str(ffmpeg_path), '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0]
                info['version'] = first_line
        except Exception as e:
            info['version'] = f'Could not determine version: {e}'
        
        return info
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.settings = {
            'sdk_path': None,
            'ffmpeg_path': None,
            'yolo_model_path': None,
            'auto_detect_on_startup': True,
            'output_format': 'PNG',
            'default_fps': 1.0,
            'default_h_fov': 110,
            'default_split_count': 8,
            'yolo_model_size': 'medium'
        }
        self.save_settings()
        logger.info("Settings reset to defaults")
    
    # === DIAGNOSTICS ===
    
    def get_diagnostics(self) -> List[Tuple[str, bool, str]]:
        """
        Get diagnostic information about SDK and FFmpeg.
        
        Returns:
            List of (name, is_valid, path_or_message) tuples
        """
        diagnostics = []
        
        # SDK
        sdk_path = self.get_sdk_path()
        if sdk_path:
            sdk_valid = self.is_sdk_valid(sdk_path)
            sdk_status = f"{sdk_path}" if sdk_valid else f"{sdk_path} (INVALID)"
            diagnostics.append(("SDK", sdk_valid, sdk_status))
        else:
            diagnostics.append(("SDK", False, "Not configured"))
        
        # FFmpeg
        ffmpeg_path = self.get_ffmpeg_path()
        if ffmpeg_path:
            ffmpeg_valid = self.is_ffmpeg_valid(ffmpeg_path)
            ffmpeg_status = f"{ffmpeg_path}" if ffmpeg_valid else f"{ffmpeg_path} (INVALID)"
            diagnostics.append(("FFmpeg", ffmpeg_valid, ffmpeg_status))
        else:
            diagnostics.append(("FFmpeg", False, "Not configured"))
        
        # YOLO Model
        yolo_path = self.get_yolo_model_path()
        if yolo_path:
            yolo_valid = self.is_yolo_model_valid(yolo_path)
            yolo_status = f"{yolo_path}" if yolo_valid else f"{yolo_path} (INVALID)"
            diagnostics.append(("YOLO Model", yolo_valid, yolo_status))
        else:
            diagnostics.append(("YOLO Model", False, "Not configured (will use Ultralytics default)"))
        
        return diagnostics


# === SINGLETON INSTANCE ===

_settings_instance = None

def get_settings() -> SettingsManager:
    """Get the global settings instance"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SettingsManager()
    return _settings_instance
