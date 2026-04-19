"""
360FrameTools - Settings Manager
Handles user preferences, path detection, and configuration persistence.
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# colmap_paths removed in simple-version — stub functions to avoid NameErrors
def build_colmap_cli_context(*args, **kwargs): return None
def normalize_colmap_executable(*args, **kwargs): return None
def preferred_colmap_candidates(*args, **kwargs): return []
from src.utils.dependency_provisioning import get_downloaded_colmap_candidates, get_downloaded_spheresfm_candidates
from src.utils.app_paths import get_settings_file_path
from src.utils.resource_path import get_base_path

logger = logging.getLogger(__name__)

_APP_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SAM3_ROOT = _APP_ROOT / 'downloads' / 'sam3cpp'
_DEFAULT_SAM3_SEGMENTER = _DEFAULT_SAM3_ROOT / 'build' / 'examples' / 'Release' / 'segment_persons.exe'
_DEFAULT_SAM3_GUI = _DEFAULT_SAM3_ROOT / 'build' / 'examples' / 'Release' / 'sam3_image.exe'
_DEFAULT_SAM3_MODEL = _DEFAULT_SAM3_ROOT / 'models' / 'sam3-q4_0.ggml'


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
            settings_file: Path to settings JSON file (default: user-writable app data path)
        """
        if settings_file is None:
            settings_file = get_settings_file_path()
        
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

    def _refresh_bundled_reconstruction_paths_REMOVED(self) -> None:  # removed in simple-version
        """Prefer executables shipped with the current frozen build over stale saved paths."""
        if not getattr(sys, 'frozen', False):
            return

        changed = False

        bundled_spheresfm = self.auto_detect_spheresfm()
        if bundled_spheresfm and self.settings.get('spheresfm_path') != str(bundled_spheresfm):
            self.settings['spheresfm_path'] = str(bundled_spheresfm)
            self.settings['spheresfm_auto_detected'] = True
            changed = True

        bundled_colmap = self.auto_detect_colmap()
        if bundled_colmap:
            if self.settings.get('colmap_gpu_path') != str(bundled_colmap):
                self.settings['colmap_gpu_path'] = str(bundled_colmap)
                changed = True
            if self.settings.get('colmap_path') != str(bundled_colmap):
                self.settings['colmap_path'] = str(bundled_colmap)
                changed = True
            self.settings['colmap_auto_detected'] = True

        if changed:
            self.save_settings()
        
    
    def _load_settings(self) -> Dict:
        """Load settings from JSON file"""
        defaults = {
            'sdk_path': None,
            'ffmpeg_path': None,
            'spheresfm_path': None,
            'colmap_gpu_path': None,
            'colmap_path': None,
            'yolo_model_path': None,
            'auto_detect_on_startup': True,
            'theme': 'dark',
            'output_format': 'PNG',
            'default_fps': 1.0,
            'default_h_fov': 110,
            'default_split_count': 8,
            'yolo_model_size': 'medium',
            'sam3_segmenter_path': str(_DEFAULT_SAM3_SEGMENTER),
            'sam3_model_path': str(_DEFAULT_SAM3_MODEL),
            'sam3_image_exe_path': str(_DEFAULT_SAM3_GUI),
        }
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        defaults.update(loaded)
                    return defaults
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
        
        # Return defaults
        return defaults
    
    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            logger.info(f"Settings saved to {self.settings_file}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def _bundled_sdk_candidates(self) -> List[Path]:
        """Return bundled/local SDK candidates in preferred search order."""
        app_dir = get_base_path()
        return [
            app_dir / "sdk",
            app_dir / "sdk" / "MediaSDK",
            app_dir / "sdk" / "MediaSDK-3.1.0.0-20250904-win64" / "MediaSDK",
            app_dir / "sdk" / "MediaSDK-3.0.5-20250619-win64" / "MediaSDK",
            app_dir / "bin" / "sdk",
            app_dir / "bin" / "sdk" / "MediaSDK",
        ]

    def _bundled_ffmpeg_candidates(self) -> List[Path]:
        """Return bundled/local FFmpeg candidates in preferred search order."""
        app_dir = get_base_path()
        return [
            app_dir / "ffmpeg" / "ffmpeg.exe",
            app_dir / "ffmpeg.exe",
            app_dir / "ffmpeg" / "bin" / "ffmpeg.exe",
            app_dir.parent / "ffmpeg" / "ffmpeg.exe",
            app_dir.parent / "ffmpeg" / "bin" / "ffmpeg.exe",
        ]

    def _detect_bundled_sdk(self) -> Optional[Path]:
        """Return the first valid bundled/local SDK path."""
        for candidate in self._bundled_sdk_candidates():
            if self.is_sdk_valid(candidate):
                logger.info(f"[OK] SDK found in bundled/local path: {candidate}")
                return candidate
        return None

    def _detect_bundled_ffmpeg(self) -> Optional[Path]:
        """Return the first valid bundled/local FFmpeg path."""
        for candidate in self._bundled_ffmpeg_candidates():
            if self.is_ffmpeg_valid(candidate):
                logger.info(f"[OK] FFmpeg found in bundled/local path: {candidate}")
                return candidate
        return None
    
    # === SDK PATH MANAGEMENT ===
    
    def auto_detect_sdk(self) -> Optional[Path]:
        """
        Auto-detect Insta360 MediaSDK installation.
        
        Search order:
        1. Bundled/local app locations (_internal/sdk, sdk/, bin/sdk)
        2. C:/Users/<User>/Documents/Windows_CameraSDK*/MediaSDK*/
        3. C:/Program Files/Insta360/MediaSDK/
        
        Returns:
            Path to MediaSDK folder if found, None otherwise
        """
        logger.info("Searching for Insta360 MediaSDK...")

        bundled_sdk = self._detect_bundled_sdk()
        if bundled_sdk:
            return bundled_sdk
        
        # 2. User Documents folder
        docs = Path.home() / "Documents"
        for sdk_dir in docs.glob("Windows_CameraSDK*"):
            for media_sdk in sdk_dir.glob("MediaSDK*"):
                if self.is_sdk_valid(media_sdk):
                    logger.info(f"[OK] SDK found at {media_sdk}")
                    return media_sdk
                nested_media_sdk = media_sdk / "MediaSDK"
                if self.is_sdk_valid(nested_media_sdk):
                    logger.info(f"[OK] SDK found at {nested_media_sdk}")
                    return nested_media_sdk
            # Fallback recursive search for nested MediaSDK folders
            for nested in sdk_dir.rglob("MediaSDK"):
                if self.is_sdk_valid(nested):
                    logger.info(f"[OK] SDK found at {nested}")
                    return nested
        
        # 3. Program Files
        program_files = [
            Path("C:/Program Files/Insta360/MediaSDK"),
            Path("C:/Program Files (x86)/Insta360/MediaSDK")
        ]
        for pf in program_files:
            if self.is_sdk_valid(pf):
                logger.info(f"[OK] SDK found at {pf}")
                return pf
        
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
            sdk_path / "bin" / "MediaSDK.dll",
            sdk_path / "MediaSDK" / "bin" / "MediaSDK.dll"
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
        bundled_sdk = self._detect_bundled_sdk()
        if bundled_sdk:
            return bundled_sdk

        sdk_str = self.settings.get('sdk_path')
        if sdk_str and self.is_sdk_valid(sdk_str):
            return Path(sdk_str)
        return None
    
    # === FFMPEG PATH MANAGEMENT ===
    
    def auto_detect_ffmpeg(self) -> Optional[Path]:
        """
        Auto-detect FFmpeg installation.
        
        Search order:
        1. Bundled/local app locations (_internal/ffmpeg, ffmpeg/)
        2. System PATH (shutil.which)
        3. C:/ffmpeg/bin/ffmpeg.exe
        4. WinGet installed location (C:/Users/<User>/AppData/Local/Microsoft/WinGet/Packages/...)
        
        Returns:
            Path to ffmpeg.exe if found, None otherwise
        """
        logger.info("Searching for FFmpeg...")

        bundled_ffmpeg = self._detect_bundled_ffmpeg()
        if bundled_ffmpeg:
            return bundled_ffmpeg
        
        # 2. System PATH
        ffmpeg_in_path = shutil.which('ffmpeg')
        if ffmpeg_in_path:
            ffmpeg_path = Path(ffmpeg_in_path)
            if self.is_ffmpeg_valid(ffmpeg_path):
                logger.info(f"[OK] FFmpeg found in PATH: {ffmpeg_path}")
                return ffmpeg_path
        
        # 3. Standard installation
        standard_paths = [
            Path("C:/ffmpeg/bin/ffmpeg.exe"),
            Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe")
        ]
        for path in standard_paths:
            if self.is_ffmpeg_valid(path):
                logger.info(f"[OK] FFmpeg found at {path}")
                return path
        
        # 4. WinGet packages location
        winget_base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if winget_base.exists():
            for pkg_dir in winget_base.glob("*FFmpeg*"):
                for ffmpeg_exe in pkg_dir.rglob("ffmpeg.exe"):
                    if self.is_ffmpeg_valid(ffmpeg_exe):
                        logger.info(f"[OK] FFmpeg found at {ffmpeg_exe}")
                        return ffmpeg_exe
        
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
        bundled_ffmpeg = self._detect_bundled_ffmpeg()
        if bundled_ffmpeg:
            return bundled_ffmpeg

        ffmpeg_str = self.settings.get('ffmpeg_path')
        if ffmpeg_str and self.is_ffmpeg_valid(ffmpeg_str):
            return Path(ffmpeg_str)
        return None

    # === RECONSTRUCTION BINARY PATH MANAGEMENT ===

    def auto_detect_spheresfm(self) -> Optional[Path]:
        """
        Auto-detect SphereSfM-compatible colmap binary.
        """
        logger.info("Searching for SphereSfM...")

        app_dir = get_base_path()
        candidates = [
            app_dir / "bin" / "SphereSfM-2024-12-14" / "colmap.exe",
            app_dir / "bin" / "SphereSfM-2024-12-14" / "colmap.bat",
            app_dir / "bin" / "SphereSfM-2024-12-14" / "colmap.cmd",
            app_dir / "bin" / "SphereSfM" / "colmap.exe",
            app_dir / "bin" / "SphereSfM" / "colmap.bat",
            app_dir / "bin" / "SphereSfM" / "colmap.cmd",
            *get_downloaded_spheresfm_candidates(),
        ]

        docs = Path.home() / "Documents"
        candidates.extend([
            docs / "APLICATIVOS" / "360toolkit" / "bin" / "SphereSfM-2024-12-14" / "colmap.exe",
            docs / "APLICATIVOS" / "360ToolKit" / "bin" / "SphereSfM-2024-12-14" / "colmap.exe",
            docs / "APLICATIVOS" / "360toolkit" / "bin" / "SphereSfM" / "colmap.exe",
            docs / "APLICATIVOS" / "360ToolKit" / "bin" / "SphereSfM" / "colmap.exe",
        ])

        for candidate in candidates:
            if self.is_colmap_valid(candidate):
                logger.info(f"[OK] SphereSfM found at {candidate}")
                return candidate

        logger.warning("[X] SphereSfM not found in standard locations")
        return None

    def auto_detect_colmap(self) -> Optional[Path]:
        """
        Auto-detect COLMAP GPU binary.

        Search order:
        1. Workspace bundled SphereSfM/COLMAP binaries
        2. Common local installs
        3. System PATH (colmap)

        Returns:
            Path to colmap executable if found, None otherwise
        """
        logger.info("Searching for COLMAP (GPU build)...")

        app_dir = get_base_path()
        bundled_candidates = preferred_colmap_candidates(app_dir)
        for candidate in bundled_candidates:
            if self.is_colmap_valid(candidate):
                logger.info(f"[OK] COLMAP found at {candidate}")
                return candidate

        if getattr(sys, "frozen", False):
            logger.warning("[X] Bundled COLMAP not found in frozen application layout")
            return None

        docs = Path.home() / "Documents"
        common_candidates = [
            docs / "APLICATIVOS" / "360toolkit" / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI" / "bin" / "colmap.exe",
            docs / "APLICATIVOS" / "360toolkit" / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI" / "COLMAP.bat",
            docs / "APLICATIVOS" / "360ToolKit" / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI" / "bin" / "colmap.exe",
            docs / "APLICATIVOS" / "360ToolKit" / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI" / "COLMAP.bat",
            docs / "APLICATIVOS" / "360toolkit" / "bin" / "colmap" / "colmap.exe",
            docs / "APLICATIVOS" / "360ToolKit" / "bin" / "colmap" / "colmap.exe",
            docs / "colmap-x64-windows-cuda" / "colmap.exe",
            docs / "colmap-x64-windows-cuda" / "colmap.bat",
            Path("C:/colmap/colmap.exe"),
        ]
        for candidate in common_candidates:
            if self.is_colmap_valid(candidate):
                logger.info(f"[OK] COLMAP found at {candidate}")
                return candidate

        for command in ("colmap", "colmap.exe", "colmap.bat", "colmap.cmd"):
            in_path = shutil.which(command)
            if in_path and self.is_colmap_valid(Path(in_path)):
                logger.info(f"[OK] COLMAP found in PATH: {in_path}")
                return Path(in_path)

        logger.warning("[X] COLMAP GPU build not found in any standard location")
        return None

    def is_colmap_valid(self, colmap_path: Optional[Path]) -> bool:
        """Validate COLMAP path by checking executable exists."""
        if colmap_path is None:
            return False

        colmap_path = normalize_colmap_executable(colmap_path)
        if colmap_path is None:
            return False
        if colmap_path.is_dir():
            for name in ("colmap.exe", "colmap.bat", "colmap.cmd"):
                candidate = colmap_path / name
                if candidate.exists() and candidate.is_file():
                    return True
            return False

        return colmap_path.exists() and colmap_path.is_file()

    def set_colmap_path(self, path: Optional[Path], auto_detected: bool = False) -> bool:
        """Backward-compatible alias for COLMAP GPU path setter."""
        return self.set_colmap_gpu_path(path, auto_detected=auto_detected)

    def set_colmap_gpu_path(self, path: Optional[Path], auto_detected: bool = False) -> bool:
        """Set COLMAP GPU path and save. Returns True if valid, False otherwise."""
        if path is None:
            self.settings['colmap_gpu_path'] = None
            self.settings['colmap_path'] = None
            self.settings['colmap_auto_detected'] = False
            self.save_settings()
            logger.info("COLMAP GPU path cleared")
            return True

        if not self.is_colmap_valid(path):
            logger.warning(f"Invalid COLMAP path: {path}")
            return False

        path_obj = normalize_colmap_executable(path)
        if path_obj is None:
            logger.warning(f"Invalid COLMAP path: {path}")
            return False
        if not path_obj.is_absolute():
            logger.warning(f"COLMAP path must be absolute (got: {path_obj})")
            return False

        path_obj = path_obj.resolve()

        self.settings['colmap_gpu_path'] = str(path_obj)
        self.settings['colmap_path'] = str(path_obj)
        self.settings['colmap_auto_detected'] = auto_detected
        self.save_settings()
        logger.info(f"COLMAP GPU path set to {path_obj}")
        return True

    def get_colmap_path(self) -> Optional[Path]:
        """Backward-compatible alias for COLMAP GPU path getter."""
        return self.get_colmap_gpu_path()

    def get_colmap_gpu_path(self) -> Optional[Path]:
        """Get current COLMAP GPU path."""
        colmap_str = self.settings.get('colmap_gpu_path') or self.settings.get('colmap_path')
        if colmap_str:
            normalized = normalize_colmap_executable(colmap_str)
            if normalized and self.settings.get('colmap_gpu_path') != str(normalized):
                self.settings['colmap_gpu_path'] = str(normalized)
                self.settings['colmap_path'] = str(normalized)
                self.save_settings()
            return normalized
        return None

    def set_spheresfm_path(self, path: Optional[Path], auto_detected: bool = False) -> bool:
        """Set SphereSfM path and save. Returns True if valid, False otherwise."""
        if path is None:
            self.settings['spheresfm_path'] = None
            self.settings['spheresfm_auto_detected'] = False
            self.save_settings()
            logger.info("SphereSfM path cleared")
            return True

        if not self.is_colmap_valid(path):
            logger.warning(f"Invalid SphereSfM path: {path}")
            return False

        path_obj = Path(path)
        if path_obj.is_dir():
            for name in ("colmap.exe", "colmap.bat", "colmap.cmd"):
                candidate = path_obj / name
                if candidate.exists() and candidate.is_file():
                    path_obj = candidate
                    break

        self.settings['spheresfm_path'] = str(path_obj)
        self.settings['spheresfm_auto_detected'] = auto_detected
        self.save_settings()
        logger.info(f"SphereSfM path set to {path_obj}")
        return True

    def get_spheresfm_path(self) -> Optional[Path]:
        """Get current SphereSfM path."""
        spheresfm_str = self.settings.get('spheresfm_path')
        if spheresfm_str:
            return Path(spheresfm_str)
        return None

    def get_colmap_info(self, colmap_path: Optional[Path]) -> Dict:
        """
        Get detailed information about COLMAP installation.

        Returns:
            Dictionary with COLMAP info (version, path)
        """
        if colmap_path is None or not self.is_colmap_valid(colmap_path):
            return {'valid': False, 'error': 'Invalid or missing COLMAP path'}

        colmap_path = normalize_colmap_executable(colmap_path)
        if colmap_path is None:
            return {'valid': False, 'error': 'Invalid or missing COLMAP path'}
        if colmap_path.is_dir():
            for name in ("colmap.exe", "colmap.bat", "colmap.cmd"):
                candidate = colmap_path / name
                if candidate.exists() and candidate.is_file():
                    colmap_path = candidate
                    break

        info = {
            'valid': True,
            'path': str(colmap_path),
            'executable': colmap_path.name,
        }

        try:
            import subprocess
            run_cwd, run_env = build_colmap_cli_context(colmap_path)
            version_cmds = [
                [str(colmap_path), 'help'],
                [str(colmap_path), '--version'],
                [str(colmap_path), 'version'],
            ]
            for cmd in version_cmds:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=run_cwd,
                    env=run_env,
                )
                output = (result.stdout or result.stderr).strip()
                if result.returncode == 0 and output:
                    info['version'] = output.splitlines()[0]
                    break
            if 'version' not in info:
                info['version'] = 'Could not determine version'
        except Exception as e:
            info['version'] = f'Could not determine version: {e}'

        return info

    def _normalize_colmap_setting(self) -> Optional[Path]:
        stored_value = self.settings.get('colmap_gpu_path') or self.settings.get('colmap_path')
        if not stored_value:
            return None

        normalized = normalize_colmap_executable(stored_value)
        if normalized and not getattr(sys, 'frozen', False):
            try:
                normalized_str = str(normalized.resolve()).lower()
            except Exception:
                normalized_str = str(normalized).lower()

            normalized_str = normalized_str.replace('/', '\\')
            is_packaged_dist_path = '\\dist\\360toolkitgs' in normalized_str and '\\_internal\\' in normalized_str

            if is_packaged_dist_path:
                detected = self.auto_detect_colmap()
                if detected and detected.exists() and detected.is_file():
                    logger.info(
                        "Replacing source-mode COLMAP path from packaged dist runtime with workspace binary: %s",
                        detected,
                    )
                    return detected
                logger.info(
                    "Ignoring source-mode COLMAP path from packaged dist runtime: %s",
                    normalized,
                )
                return None

        if normalized and normalized.exists() and normalized.is_file():
            return normalized
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

    # === SAM3 PATH MANAGEMENT ===

    def get_sam3_segmenter_path(self) -> Optional[Path]:
        value = self.settings.get('sam3_segmenter_path')
        return Path(value) if value else None

    def set_sam3_segmenter_path(self, path: Optional[Path | str]):
        self.settings['sam3_segmenter_path'] = str(path) if path else ''
        self.save_settings()

    def get_sam3_model_path(self) -> Optional[Path]:
        value = self.settings.get('sam3_model_path')
        return Path(value) if value else None

    def set_sam3_model_path(self, path: Optional[Path | str]):
        self.settings['sam3_model_path'] = str(path) if path else ''
        self.save_settings()

    def get_sam3_image_exe_path(self) -> Optional[Path]:
        value = self.settings.get('sam3_image_exe_path')
        return Path(value) if value else None

    def set_sam3_image_exe_path(self, path: Optional[Path | str]):
        self.settings['sam3_image_exe_path'] = str(path) if path else ''
        self.save_settings()
    
    # === GLOMAP PATH MANAGEMENT ===
    
    def auto_detect_glomap(self) -> Optional[Path]:
        """
        Backward-compatible resolver for legacy GloMAP path.

        Modern COLMAP builds include `global_mapper`, so we now resolve
        this value to the COLMAP executable path.
        """
        logger.info("Resolving legacy GloMAP path via COLMAP global_mapper support...")
        return self.get_colmap_gpu_path() or self.auto_detect_colmap()
    
    def is_glomap_valid(self, glomap_path: Optional[Path]) -> bool:
        """Legacy validator: accepts any valid COLMAP executable path."""
        return self.is_colmap_valid(glomap_path)
    
    def set_glomap_path(self, path: Optional[Path], auto_detected: bool = False) -> bool:
        """Legacy setter: persists path, expecting COLMAP executable."""
        if path is None:
            self.settings['glomap_path'] = None
            self.settings['glomap_auto_detected'] = False
            self.save_settings()
            logger.info("Legacy GloMAP path cleared")
            return True

        if not self.is_colmap_valid(path):
            logger.warning(f"Invalid legacy GloMAP/COLMAP path: {path}")
            return False

        self.settings['glomap_path'] = str(Path(path))
        self.settings['glomap_auto_detected'] = auto_detected
        self.save_settings()
        logger.info(f"Legacy GloMAP path set to COLMAP executable {path}")
        return True
    
    def get_glomap_path(self) -> Optional[Path]:
        """Legacy getter: returns dedicated path if set, otherwise COLMAP path."""
        glomap_str = self.settings.get('glomap_path')
        if glomap_str:
            return Path(glomap_str)
        return self.get_colmap_gpu_path()
    
    def get_glomap_info(self, glomap_path: Optional[Path]) -> Dict:
        """
        Backward-compatible info helper for legacy GloMAP slot.

        Since global mapper is integrated in COLMAP, this proxies COLMAP info.

        Returns:
            Dictionary with GloMAP info (version, CUDA support, etc.)
        """
        if glomap_path is None:
            glomap_path = self.get_glomap_path()

        if glomap_path is None or not self.is_colmap_valid(glomap_path):
            return {'valid': False, 'error': 'Invalid or missing COLMAP path for global mapper'}

        info = self.get_colmap_info(glomap_path)
        if info.get('valid'):
            info['version'] = f"{info.get('version', 'COLMAP')} (global_mapper integrated)"
            info['cuda'] = 'cuda' in info.get('version', '').lower() or 'gpu' in info.get('version', '').lower()
        return info
    
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
    
    def get_theme(self) -> str:
        """Get UI theme: dark, light, or system."""
        return self.settings.get('theme', 'dark')

    def set_theme(self, theme: str):
        """Set UI theme if valid and save."""
        valid = {'dark', 'light', 'system'}
        value = (theme or 'dark').lower()
        if value not in valid:
            value = 'dark'
        self.settings['theme'] = value
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
            'spheresfm_path': None,
            'colmap_gpu_path': None,
            'colmap_path': None,
            'yolo_model_path': None,
            'auto_detect_on_startup': True,
            'theme': 'dark',
            'output_format': 'PNG',
            'default_fps': 1.0,
            'default_h_fov': 110,
            'default_split_count': 8,
            'yolo_model_size': 'medium',
            'sam3_segmenter_path': str(_DEFAULT_SAM3_SEGMENTER),
            'sam3_model_path': str(_DEFAULT_SAM3_MODEL),
            'sam3_image_exe_path': str(_DEFAULT_SAM3_GUI),
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

        # COLMAP
        spheresfm_path = self.get_spheresfm_path()
        if spheresfm_path:
            spheresfm_valid = self.is_colmap_valid(spheresfm_path)
            spheresfm_info = self.get_colmap_info(spheresfm_path)
            spheresfm_version = spheresfm_info.get('version', 'Unknown') if spheresfm_valid else 'Unknown'
            spheresfm_status = f"{spheresfm_path} [{spheresfm_version}]" if spheresfm_valid else f"{spheresfm_path} (INVALID)"
            diagnostics.append(("SphereSfM", spheresfm_valid, spheresfm_status))
        else:
            diagnostics.append(("SphereSfM", False, "Not configured"))

        colmap_path = self.get_colmap_gpu_path()
        if colmap_path:
            colmap_valid = self.is_colmap_valid(colmap_path)
            colmap_info = self.get_colmap_info(colmap_path)
            colmap_version = colmap_info.get('version', 'Unknown') if colmap_valid else 'Unknown'
            colmap_status = f"{colmap_path} [{colmap_version}]" if colmap_valid else f"{colmap_path} (INVALID)"
            diagnostics.append(("COLMAP GPU", colmap_valid, colmap_status))
        else:
            diagnostics.append(("COLMAP GPU", False, "Not configured"))
        
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
