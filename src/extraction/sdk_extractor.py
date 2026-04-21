r"""
Insta360 MediaSDK 3.1.x Integration Module

PRIMARY EXTRACTION METHOD - Uses official Insta360 MediaSDK for highest quality stitching.

SDK Documentation: https://github.com/Insta360Develop/Desktop-MediaSDK-Cpp
SDK Location: C:\Users\Everton-PC\Documents\Windows_CameraSDK-2.1.1_MediaSDK-3.1.0\MediaSDK-3.1.0.0-20250904-win64\MediaSDK

Hardware Requirements:
- GPU with CUDA or Vulkan support (REQUIRED for v3.x)
- Windows 7+ (x64 only) or Ubuntu 22.04
- 8GB+ VRAM recommended for 8K output

TESTED STITCH TYPES (2024-12-04):
- dynamicstitch: PERFECT results, fast (~1.4s per frame) - RECOMMENDED DEFAULT
- aistitch + v2 model: PERFECT results, slower (~2.2s per frame) - HIGHEST QUALITY
- optflow: Good but can have noise in sky areas (~1.4s per frame)
- template: Fast but low quality - use for previews only

Key MediaSDK APIs:
- SetImageSequenceInfo(output_dir, IMAGE_TYPE): Export video frames as image sequence
- SetExportFrameSequence(frame_indices): Extract specific frames by index
- SetStitchType(STITCH_TYPE): dynamicstitch (recommended), aistitch, optflow, template
- EnableStitchFusion(True): Chromatic calibration for seamless blending (CRITICAL)
- EnableFlowState(True): FlowState stabilization (REQUIRED for DirectionLock)
- EnableDirectionLock(True): Locks horizon/direction like Insta360 app (requires FlowState)
- EnableColorPlus(True, strength): AI color enhancement (strength 0.0-1.0, default 0.3)
- SetAiStitchModelFile(model_path): ai_stitcher_model_v1.ins or v2.ins (v2 is better)
- SetOutputSize(width, height): Output resolution (must be 2:1 ratio)
- StartStitch(): Begin stitching process

Color Correction APIs (all map to CLI flags, range noted):
- SetExposure(int):    -exposure   [-100, 100]
- SetHighlights(int): -highlights [-100, 100]
- SetShadows(int):    -shadows    [-100, 100]
- SetContrast(int):   -contrast   [-100, 100]
- SetBrightness(int): -brightness [-100, 100]
- SetBlackpoint(int): -blackpoint [-100, 100]
- SetSaturation(int): -saturation [-100, 100]
- SetVibrance(int):   -vibrance   [-100, 100]
- SetWarmth(int):     -warmth     [-100, 100]
- SetTint(int):       -tint       [-100, 100]
- SetDefinition(int): -definition [0, 100]

Frame Extraction Workflow:
1. SetInputPath([video_file_1, video_file_2])  # Dual-track or single-track
2. SetImageSequenceInfo(output_dir, IMAGE_TYPE.JPEG)  # Or PNG
3. SetExportFrameSequence([0, 10, 20, 30, ...])  # Frame indices based on FPS
4. SetStitchType(STITCH_TYPE.DYNAMICSTITCH)  # PERFECT results (tested!)
5. SetAiStitchModelFile(ai_model_v2_path)  # Use v2 model for best quality
6. EnableStitchFusion(True)  # CRITICAL for seamless blending
7. SetOutputSize(7680, 3840)  # 8K output
8. StartStitch()  # Execute

Output Naming:
- With SetExportFrameSequence: {frame_index}.jpg (e.g., 10.jpg = frame #10)
- Without: {timestamp_ms}.jpg (e.g., 100.jpg = frame at 100ms)

Fallback Strategy:
If SDK executable not found or GPU unavailable -> FFmpeg Method (proven filter chain)
"""

import subprocess
import logging
import json
import cv2
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple

from src.config.settings import get_settings
from src.utils.resource_path import get_base_path

logger = logging.getLogger(__name__)


def _subprocess_no_window_kwargs() -> dict:
    kwargs = {}
    if os.name == 'nt':
        creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, 'STARTF_USESHOWWINDOW', 0)
        kwargs['creationflags'] = creationflags
        kwargs['startupinfo'] = startupinfo
    return kwargs


class IncompleteSDKExtractionError(RuntimeError):
    """Raised when MediaSDK exits or stalls before producing the requested frame set."""

    def __init__(
        self,
        message: str,
        *,
        expected_count: int,
        actual_count: int,
        missing_indices: Optional[List[int]] = None,
    ):
        super().__init__(message)
        self.expected_count = expected_count
        self.actual_count = actual_count
        self.missing_indices = list(missing_indices or [])


# Auto-detect SDK path (supports frozen PyInstaller apps)
def _get_default_sdk_path():
    """Get SDK path from environment or bundled locations only."""
    # 1. Check environment variable (set by runtime hook)
    if 'INSTA360_SDK_PATH' in os.environ:
        logger.info(f"Using SDK from environment: {os.environ['INSTA360_SDK_PATH']}")
        return os.environ['INSTA360_SDK_PATH']
    
    # 2. Check if running as PyInstaller frozen app
    if hasattr(sys, '_MEIPASS'):
        # Try bundled SDK locations
        base_path = Path(sys._MEIPASS)
        bundled_locations = [
            base_path / 'sdk' / 'MediaSDK-3.0.5-20250619-win64',
            base_path / '_internal' / 'sdk' / 'MediaSDK-3.0.5-20250619-win64',
            base_path / 'sdk',
            base_path / '_internal' / 'sdk',
        ]
        for loc in bundled_locations:
            if loc.exists():
                logger.info(f"Found bundled SDK at: {loc}")
                return str(loc)
        logger.warning("Running as frozen app but SDK not found in bundle!")

    base_path = get_base_path()
    dev_candidates = [
        base_path / 'sdk',
        base_path / 'bin' / 'sdk',
    ]
    for loc in dev_candidates:
        if loc.exists():
            logger.info(f"Found SDK candidate at: {loc}")
            return str(loc)

    logger.warning("SDK path not auto-detected; user configuration is required")
    return None

# SDK configuration
DEFAULT_SDK_PATH = _get_default_sdk_path()


# Stitch type presets (MediaSDK 3.0.5)
# NOTE: These values must match exactly what MediaSDKTest.exe expects!
# Run "MediaSDKTest.exe --help" to see valid stitch_type values.
# TESTED RESULTS:
#   - dynamicstitch: PERFECT results, fast (1.42s) - RECOMMENDED
#   - aistitch + v2 model: PERFECT results, slower (2.21s) - BEST QUALITY
#   - optflow: Good but noisy sky (1.42s)
#   - template: Fast but low quality
STITCH_TYPES = {
    'dynamic': 'dynamicstitch', # Dynamic Stitching - PERFECT results, fast (RECOMMENDED)
    'aistitch': 'aistitch',     # AI Stitching with v2 model - PERFECT results, slower
    'optflow': 'optflow',       # Optical Flow - good but can have noise
    'template': 'template'      # Template (FAST, lower quality)
}


# Quality presets
# TESTED: 'dynamicstitch' and 'aistitch' with v2 model produce PERFECT results
# The SDK CLI accepts: template, optflow, dynamicstitch, aistitch
# Default color correction values (neutral — no correction applied unless overridden)
_COLOR_DEFAULTS = {
    'exposure':    0,   # [-100, 100]
    'highlights':  0,   # [-100, 100]
    'shadows':     0,   # [-100, 100]
    'contrast':    0,   # [-100, 100]
    'brightness':  0,   # [-100, 100]
    'blackpoint':  0,   # [-100, 100]
    'saturation':  0,   # [-100, 100]
    'vibrance':    0,   # [-100, 100]
    'warmth':      0,   # [-100, 100]
    'tint':        0,   # [-100, 100]
    'definition':  0,   # [0, 100]
}

_NATIVE_COLOR_KEYS = (
    'exposure', 'highlights', 'shadows', 'contrast', 'brightness',
    'blackpoint', 'saturation', 'vibrance', 'warmth', 'tint', 'definition',
)

QUALITY_PRESETS = {
    'best': {
        'stitch_type': 'dynamicstitch', # Dynamic Stitching - PERFECT results, recommended!
        'use_ai_model_v2': True,        # Use v2 model for best AI processing
        'enable_stitchfusion': True,    # Chromatic calibration (CRITICAL for seamless blending)
        'enable_flowstate': True,       # Stabilization (also required by direction_lock)
        'enable_direction_lock': False, # Lock horizon/direction (requires flowstate=True)
        'enable_colorplus': True,       # AI color enhancement
        'colorplus_strength': 0.3,      # Color Plus strength (0.0-1.0, SDK default 0.3)
        'enable_denoise': True,         # AI denoising (reduces noise in sky/flat areas)
        'enable_defringe': True,        # Purple fringe removal
        **_COLOR_DEFAULTS,              # Color correction (all neutral by default)
    },
    'good': {
        'stitch_type': 'dynamicstitch', # Dynamic Stitching - PERFECT results
        'use_ai_model_v2': False,       # Use v1 model (faster)
        'enable_stitchfusion': True,    # Keep chromatic calibration
        'enable_flowstate': True,       # Stabilization
        'enable_direction_lock': False, # Lock horizon/direction (requires flowstate=True)
        'enable_colorplus': False,
        'colorplus_strength': 0.3,
        'enable_denoise': False,
        'enable_defringe': False,
        **_COLOR_DEFAULTS,
    },
    'balanced': {
        'stitch_type': 'optflow',       # Optical Flow - good quality, moderate speed
        'use_ai_model_v2': False,
        'enable_stitchfusion': True,    # Chromatic calibration
        'enable_flowstate': True,       # Stabilization
        'enable_direction_lock': False,
        'enable_colorplus': False,
        'colorplus_strength': 0.3,
        'enable_denoise': False,
        'enable_defringe': False,
        **_COLOR_DEFAULTS,
    },
    'draft': {
        'stitch_type': 'template',      # Template - fastest, lowest quality
        'use_ai_model_v2': False,
        'enable_stitchfusion': False,
        'enable_flowstate': False,
        'enable_direction_lock': False, # NOTE: direction_lock requires flowstate; disabled in draft
        'enable_colorplus': False,
        'colorplus_strength': 0.3,
        'enable_denoise': False,
        'enable_defringe': False,
        **_COLOR_DEFAULTS,
    }
}


class SDKExtractor:
    """
    Insta360 MediaSDK wrapper for frame extraction with stitching.
    
    Uses subprocess to call MediaSDK-Demo.exe with appropriate parameters.
    Implements SetImageSequenceInfo + SetExportFrameSequence workflow.
    """
    
    def __init__(self, sdk_path: Optional[str] = None):
        """
        Initialize SDK extractor.
        
        Args:
            sdk_path: Path to MediaSDK installation (auto-detects if None)
        """
        # Use settings manager for SDK path
        settings = get_settings()
        
        if sdk_path:
            self.sdk_path = Path(sdk_path)
        else:
            configured_sdk = settings.get_sdk_path()
            if configured_sdk:
                self.sdk_path = configured_sdk
                logger.info(f"Using SDK from settings: {configured_sdk}")
            else:
                # Fall back to default detection
                if DEFAULT_SDK_PATH:
                    self.sdk_path = Path(DEFAULT_SDK_PATH)
                    logger.info(f"No SDK configured in settings, using detected path: {self.sdk_path}")
                else:
                    self.sdk_path = get_base_path() / "sdk"
                    logger.info("No SDK configured in settings and no bundled SDK found")
        
        self.is_cancelled = False
        
        # Detect SDK executable (multiple possible locations)
        self.demo_exe = self._find_sdk_executable()
        
        # Find model files (search in multiple possible locations)
        self._locate_model_files()
        
        # Check SDK availability
        self.available = self._check_sdk_available()
        
        # Process handle for termination support
        self._current_process = None
        
        if self.available:
            logger.info(f"[OK] Insta360 MediaSDK detected: {self.demo_exe}")
            logger.info(f"[OK] AI Model V1: {self.ai_model_v1.exists()}")
            logger.info(f"[OK] AI Model V2: {self.ai_model_v2.exists()}")
        else:
            logger.warning("WARNING: Insta360 MediaSDK not found - will fallback to FFmpeg")
    
    def _find_sdk_executable(self) -> Optional[Path]:
        """Find MediaSDK executable in multiple possible locations."""
        possible_paths = [
            # Bundled structure - PREFER MediaSDKTest.exe as it supports CLI args
            self.sdk_path / "bin" / "MediaSDKTest.exe",
            self.sdk_path / "bin" / "RealTimeStitcherSDKTest.exe",
            
            # MediaSDK 3.0.5 structure
            self.sdk_path / "MediaSDK-3.0.5-20250619-win64" / "MediaSDK" / "bin" / "MediaSDKTest.exe",
            self.sdk_path / "MediaSDK-3.0.5-20250619-win64" / "MediaSDK" / "bin" / "RealTimeStitcherSDKTest.exe",
            self.sdk_path / "MediaSDK" / "bin" / "MediaSDKTest.exe",
            self.sdk_path / "MediaSDK" / "bin" / "RealTimeStitcherSDKTest.exe",
            
            # Demo executable (common in older SDK versions)
            self.sdk_path / "Demo" / "Windows" / "Media SDK" / "MediaSDK-Demo.exe",
            self.sdk_path / "Demo" / "Windows" / "Media SDK" / "Release" / "MediaSDK-Demo.exe",
            
            # Alternate locations
            self.sdk_path / "bin" / "MediaSDK-Demo.exe",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found SDK executable: {path}")
                # Store SDK base directory for model file search
                # Path is like: .../MediaSDK/bin/MediaSDKTest.exe
                # Go up 2 levels: MediaSDK/bin -> MediaSDK
                self.sdk_base = path.parent.parent
                
                # CRITICAL FIX: SDK 3.1.x looks for models relative to CWD (bin/)
                # Create junction: bin/models -> ../models if it doesn't exist
                bin_models = path.parent / "models"
                sdk_models = self.sdk_base / "models"
                if sdk_models.exists() and not bin_models.exists():
                    try:
                        import subprocess
                        # Use mklink /J (junction) which doesn't require admin rights
                        subprocess.run(
                            ["cmd", "/c", "mklink", "/J", str(bin_models), str(sdk_models)],
                            check=True,
                            capture_output=True,
                            **_subprocess_no_window_kwargs(),
                        )
                        logger.info(f"[OK] Created models junction: {bin_models} -> {sdk_models}")
                    except Exception as e:
                        logger.warning(f"Failed to create models junction: {e}")
                        logger.warning("SDK may fail to find model files. Try manually creating junction.")
                
                return path
        
        logger.warning(f"MediaSDK executable not found in {len(possible_paths)} checked locations")
        return None
    
    def _locate_model_files(self):
        """Find model files in SDK installation (multiple possible locations)."""
        if not self.demo_exe or not hasattr(self, 'sdk_base'):
            # SDK not found, use default paths
            self.ai_model_v1 = self.sdk_path / "models" / "ai_stitch_model_v1.ins"
            self.ai_model_v2 = self.sdk_path / "models" / "ai_stitch_model_v2.ins"
            self.colorplus_model = self.sdk_path / "models" / "colorplus_model.ins"
            self.denoise_model = self.sdk_path / "models" / "jpg_denoise_9d006262.ins"
            self.defringe_model = self.sdk_path / "models" / "defringe_hr_dynamic_7b56e80f.ins"
            self.deflicker_model = self.sdk_path / "models" / "deflicker_86ccba0d.ins"
            return
        
        # Search in SDK-relative locations
        possible_model_dirs = [
            self.sdk_base / "models",     # MediaSDK 3.1.0 location (FIRST PRIORITY)
            self.sdk_base / "modelfile",  # MediaSDK 3.0.5 location
            self.sdk_base / "data",       # Older SDK versions
            self.sdk_path / "models",     # Root SDK path
            self.sdk_path / "data",       # Root SDK path alternate
            self.sdk_path / "modelfile",  # Root SDK path alternate
        ]
        
        # Find AI stitch model v1 (try different names)
        self.ai_model_v1 = self._find_model_file(
            possible_model_dirs,
            ["ai_stitcher_v1.ins", "ai_stitcher_model_v1.ins", "ai_stitch_model_v1.ins"]
        )
        
        # Find AI stitch model v2
        self.ai_model_v2 = self._find_model_file(
            possible_model_dirs,
            ["ai_stitcher_v2.ins", "ai_stitcher_model_v2.ins", "ai_stitch_model_v2.ins"]
        )
        
        # Find Color Plus model
        self.colorplus_model = self._find_model_file(
            possible_model_dirs,
            ["colorplus_model.ins"]
        )
        
        # Find denoise model
        self.denoise_model = self._find_model_file(
            possible_model_dirs,
            ["jpg_denoise_9d006262.ins", "denoise_model.ins"]
        )
        
        # Find defringe model
        self.defringe_model = self._find_model_file(
            possible_model_dirs,
            ["defringe_hr_dynamic_7b56e80f.ins", "defringe_model.ins"]
        )
        
        # Find deflicker model
        self.deflicker_model = self._find_model_file(
            possible_model_dirs,
            ["deflicker_86ccba0d.ins", "deflicker_model.ins"]
        )
    
    def _find_model_file(self, search_dirs: List[Path], filenames: List[str]) -> Path:
        """Search for a model file in multiple directories with multiple possible names."""
        for directory in search_dirs:
            if not directory.exists():
                continue
            for filename in filenames:
                model_path = directory / filename
                if model_path.exists():
                    logger.info(f"Found model: {model_path}")
                    return model_path
        
        # Not found - return first possible path as fallback
        return search_dirs[0] / filenames[0] if search_dirs and filenames else Path("not_found.ins")
    
    def _check_sdk_available(self) -> bool:
        """Check if SDK is properly installed and GPU is available."""
        if not self.demo_exe or not self.demo_exe.exists():
            logger.error("SDK executable not found")
            return False
        
        # Check AI model availability (optional - can use other stitch types)
        if not self.ai_model_v1.exists():
            logger.warning("AI stitch model v1 not found - will use Optical Flow stitching")
            # SDK is still available, just without AI stitching
        
        # Check GPU availability (CUDA required for best/good presets)
        self._gpu_available = self._check_gpu_available()
        
        return True
    
    def _check_gpu_available(self) -> bool:
        """Check if NVIDIA GPU is available for SDK stitching."""
        # Check for nvcuda.dll in System32
        sys32 = Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32'
        nvcuda = sys32 / 'nvcuda.dll'
        if nvcuda.exists():
            logger.info(f"[OK] GPU available: Found nvcuda.dll at {nvcuda}")
            return True
        
        # Check via CUDA environment variable
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and Path(cuda_path).exists():
            logger.info(f"[OK] GPU available: Found CUDA at {cuda_path}")
            return True
        
        logger.warning("[WARNING] GPU not available - nvcuda.dll not found. SDK will use CPU fallback (template stitching).")
        return False
    
    def is_available(self) -> bool:
        """Check if SDK is available for use."""
        return self.available
    
    def cancel(self):
        """Cancel ongoing SDK extraction operation"""
        self.is_cancelled = True
        logger.info("SDKExtractor cancellation requested")
    
    def extract_frames(
        self,
        input_path: str,
        output_dir: str,
        fps: float = 1.0,
        quality: str = 'best',
        resolution: Optional[Tuple[int, int]] = None,
        output_format: str = 'jpg',
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        sdk_options: Optional[Dict] = None
    ) -> List[str]:
        """
        Extract frames using MediaSDK with stitching.

        Args:
            input_path: Path to .insv video file
            output_dir: Output directory for frames
            fps: Extraction rate (frames per second)
            quality: 'best', 'good', 'balanced', or 'draft'
            resolution: (width, height) for output (None = original)
            output_format: 'jpg' or 'png'
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (None = full video)
            progress_callback: Callback(progress_percent)
            sdk_options: Optional dict to override individual preset values.
                Supported keys:
                  enable_direction_lock (bool) - lock horizon/direction
                      (requires enable_flowstate=True; enabled automatically)
                  enable_colorplus (bool)       - AI color enhancement
                  colorplus_strength (float)    - Color Plus AI strength [0.0-1.0]
                  exposure (int)      [-100, 100]
                  highlights (int)    [-100, 100]
                  shadows (int)       [-100, 100]
                  contrast (int)      [-100, 100]
                  brightness (int)    [-100, 100]
                  blackpoint (int)    [-100, 100]
                  saturation (int)    [-100, 100]
                  vibrance (int)      [-100, 100]
                  warmth (int)        [-100, 100]
                  tint (int)          [-100, 100]
                  definition (int)    [0, 100]
                  output_rotation (int) - post-rotate output images by 0/90/180/270
                  auto_rotate_output (bool) - auto-fix known camera orientations
                  disable_cuda (bool) - disable CUDA when True
                  enable_soft_encode (bool) - use software encoder when True
                  enable_soft_decode (bool) - use software decoder when True
                  image_processing_accel (str) - auto/cuda/vulkan/cpu depending on SDK support
                  enable_dense_overlap_recovery (bool) - use overlapping SDK windows for dense all-frame extraction
                  dense_overlap_step (int) - frame step between overlap windows
                  dense_overlap_window (int) - number of frames requested per overlap window
                  enable_sparse_retry_recovery (bool) - retry each requested sparse frame through SDK-only windows

        Returns:
            List of extracted frame paths
        """
        if not self.available:
            raise RuntimeError("MediaSDK not available - use FFmpeg fallback")

        output_format = 'png' if str(output_format).strip().lower() == 'png' else 'jpg'
        
        # Check GPU availability and adjust quality preset if needed
        if not self._gpu_available and quality in ['best', 'good', 'balanced']:
            logger.warning(f"[GPU FALLBACK] '{quality}' preset requires GPU. Falling back to 'draft' (template stitching).")
            logger.warning("[GPU FALLBACK] For best quality, run on a PC with NVIDIA GPU.")
            quality = 'draft'
        
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info for frame calculation
        video_info = self._get_video_info(input_path)
        total_frames = video_info.get('total_frames', 0)
        video_fps = video_info.get('fps', 24.0)
        duration = video_info.get('duration', 0)
        
        # Apply time range constraints
        if end_time is None:
            end_time = duration
        
        # Convert time range to frame range
        start_frame = int(start_time * video_fps)
        end_frame = min(int(end_time * video_fps), total_frames)

        # Detect dual-track vs single-track and whether this source needs A1 recovery handling.
        source_input_files = self._detect_input_files(input_path)
        source_requires_recovery = self._source_requires_sdk_recovery(source_input_files)

        input_files_override = None
        if sdk_options:
            input_files_override = sdk_options.get('_input_files_override')
        input_files = list(input_files_override) if input_files_override else list(source_input_files)
        
        # Calculate frame indices to extract based on desired FPS within time range
        frame_interval = max(1, int(video_fps / fps))
        frame_indices = list(range(start_frame, end_frame, frame_interval))

        if self._should_use_dense_overlap_strategy(source_requires_recovery, frame_indices, frame_interval, sdk_options):
            return self._extract_dense_sequence_with_overlap(
                input_path=str(input_path),
                output_dir=str(output_dir),
                fps=fps,
                quality=quality,
                resolution=resolution,
                output_format=output_format,
                start_time=start_time,
                end_time=end_time,
                progress_callback=progress_callback,
                sdk_options=sdk_options,
                video_fps=video_fps,
                frame_indices=frame_indices,
            )

        if self._should_use_sparse_retry_strategy(source_requires_recovery, frame_indices, frame_interval, sdk_options):
            return self._extract_sparse_indices_with_retry(
                input_path=str(input_path),
                output_dir=str(output_dir),
                fps=fps,
                quality=quality,
                resolution=resolution,
                output_format=output_format,
                progress_callback=progress_callback,
                sdk_options=sdk_options,
                video_fps=video_fps,
                frame_indices=frame_indices,
                total_frames=total_frames,
            )
        
        logger.info(f"Time range: {start_time}s - {end_time}s (frames {start_frame} - {end_frame})")
        logger.info(f"Extracting {len(frame_indices)} frames from {total_frames} total")
        logger.info(f"Frame interval: {frame_interval} (video FPS: {video_fps}, target FPS: {fps})")

        output_rotation = self._determine_output_rotation(source_input_files, sdk_options)

        # Patch unsupported camtype values (e.g., Antigravity A1 uses types 112/155
        # which are not in the SDK dispatch table → replace with nearest valid types)
        _patched_temps = [] if input_files_override else self._patch_insv_camtype_if_needed(input_files)
        
        # Build MediaSDK command
        cmd = self._build_extraction_command(
            input_files=input_files,
            output_dir=output_dir,
            frame_indices=frame_indices,
            quality=quality,
            resolution=resolution,
            output_format=output_format,
            sdk_options=sdk_options
        )
        
        logger.info(f"Running MediaSDK extraction...")
        logger.info(f"Command: {' '.join(str(c) for c in cmd)}")
        
        # Execute SDK with Popen for termination support
        try:
            # Set CWD to SDK bin directory to ensure DLLs are found
            sdk_cwd = self.demo_exe.parent if self.demo_exe else None
            
            # Prepare environment - CRITICAL: build a clean PATH for MediaSDK subprocess
            # MediaSDK uses CUDA 10.x DLLs; PyTorch's CUDA 12.x DLLs in _internal can conflict
            env = os.environ.copy()
            
            if getattr(sys, 'frozen', False):
                exe_dir = Path(sys.executable).parent
                internal_dir = exe_dir / '_internal'
                
                # Build clean PATH: SDK bin + System dirs + original PATH (excluding _internal/torch)
                # This prevents CUDA DLL version conflicts between MediaSDK (CUDA 10.x) and PyTorch (CUDA 12.x)
                clean_path_parts = []
                
                # 1. SDK bin directory FIRST (its own DLLs take priority)
                if sdk_cwd:
                    clean_path_parts.append(str(sdk_cwd))
                
                # 2. System directories (nvcuda.dll, kernel32.dll, etc.)
                sys_root = os.environ.get('SystemRoot', 'C:\\Windows')
                clean_path_parts.append(os.path.join(sys_root, 'System32'))
                clean_path_parts.append(sys_root)
                clean_path_parts.append(os.path.join(sys_root, 'System32', 'Wbem'))
                
                # 3. Original PATH entries, EXCLUDING _internal and torch paths
                internal_str = str(internal_dir).lower()
                for p in env.get('PATH', '').split(os.pathsep):
                    p_lower = p.strip().lower()
                    if not p_lower:
                        continue
                    # Skip _internal dir and torch subdirs to avoid CUDA DLL conflicts
                    if internal_str in p_lower or 'torch\\lib' in p_lower or 'torch/lib' in p_lower:
                        continue
                    if p.strip() not in clean_path_parts:
                        clean_path_parts.append(p.strip())
                
                env['PATH'] = os.pathsep.join(clean_path_parts)
                logger.info(f"Built clean SDK PATH (excluded _internal to avoid CUDA conflicts)")
                
                # Copy essential VC++ runtime DLLs to SDK bin (these are safe to copy)
                if sdk_cwd and internal_dir.exists():
                    try:
                        import shutil
                        deps_to_copy = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
                                        'concrt140.dll', 'zlib.dll', 'libiomp5md.dll']
                        for dep in deps_to_copy:
                            src_dll = internal_dir / dep
                            dst_dll = sdk_cwd / dep
                            if src_dll.exists() and not dst_dll.exists():
                                logger.info(f"Copying missing dependency to SDK bin: {dep}")
                                shutil.copy2(src_dll, dst_dll)
                    except Exception as e:
                        logger.warning(f"Failed to copy dependencies: {e}")
            else:
                # Not frozen: just prepend SDK bin
                if sdk_cwd:
                    env['PATH'] = str(sdk_cwd) + os.pathsep + env.get('PATH', '')

            # DIAGNOSTIC: Log PATH and CWD
            logger.info(f"SDK CWD: {sdk_cwd}")
            # Log full PATH to debug missing system paths
            logger.info(f"SDK PATH: {env['PATH']}") 
            
            # Check for critical DLLs in SDK folder
            if sdk_cwd:
                found_dlls = [f.name for f in sdk_cwd.glob('*.dll')]
                logger.info(f"DLLs in SDK bin: {found_dlls}")
                
                # Check for nvcuda.dll in System32
                sys32 = Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32'
                nvcuda = sys32 / 'nvcuda.dll'
                if nvcuda.exists():
                    logger.info(f"Found nvcuda.dll at {nvcuda}")
                else:
                    logger.warning(f"nvcuda.dll NOT FOUND in {sys32} (GPU stitching may fail)")

            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=sdk_cwd,  # CRITICAL: Run from SDK bin folder to find DLLs
                env=env,      # CRITICAL: Include _internal in PATH
                **_subprocess_no_window_kwargs(),
            )
            
            # Calculate watchdog thresholds.
            # max_runtime is a hard upper bound only; successful completion still requires
            # the exact requested frame set. stall_timeout detects SDK runs that stop
            # producing new requested frames while the process stays alive.
            frame_count = len(frame_indices)
            base_time = frame_count * 4.0
            overhead = 300
            max_runtime = max(1800, min(14400, base_time + overhead))
            timeout_slice = 300
            stall_timeout = max(120, min(600, frame_count * 4))
            logger.info(
                "[INFO] SDK watchdog configured: max_runtime=%ss, stall_timeout=%ss for %s frames",
                int(max_runtime),
                int(stall_timeout),
                frame_count,
            )
            
            # Monitor progress in real-time with smart completion detection
            import time
            import threading
            
            last_count = 0
            completion_detected = False
            no_change_duration = 0
            
            def monitor_progress():
                """Monitor extraction progress by counting output files and detect completion"""
                nonlocal extracted_frames, last_count, completion_detected, no_change_duration
                consecutive_no_change = 0
                
                while self._current_process and self._current_process.poll() is None:
                    try:
                        current_count = len(self._collect_existing_frame_indices(output_dir, frame_indices, output_format))
                        
                        if current_count > last_count:
                            # Progress detected
                            last_count = current_count
                            consecutive_no_change = 0
                            no_change_duration = 0
                            progress_percent = int((last_count / frame_count * 100)) if frame_count > 0 else 0
                            if progress_callback:
                                progress_callback(progress_percent)
                            logger.debug(f"Progress: {last_count}/{frame_count} frames ({progress_percent}%)")
                        else:
                            # No new files - check for completion
                            consecutive_no_change += 1
                            no_change_duration += 2  # 2 seconds per check
                            
                            # If we have the exact requested frame set and no changes for 10+ seconds,
                            # the SDK has finished writing images even if the process is still alive.
                            if current_count >= frame_count and consecutive_no_change >= 5:
                                logger.info(f"[DETECTION] All {current_count} frames extracted, no changes for {no_change_duration}s")
                                completion_detected = True
                                # Terminate the waiting process
                                if self._current_process and self._current_process.poll() is None:
                                    logger.info("[DETECTION] SDK appears complete - terminating wait")
                                    try:
                                        self._current_process.terminate()
                                    except:
                                        pass
                                break
                    except Exception as e:
                        logger.debug(f"Error monitoring progress: {e}")
                    
                    time.sleep(2)  # Check every 2 seconds
            
            # Start progress monitor thread
            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()
            
            # Wait for completion with watchdog extensions while progress continues.
            stdout = ""
            stderr = ""
            returncode = 0
            
            try:
                wait_started = time.monotonic()
                while True:
                    remaining_runtime = max_runtime - (time.monotonic() - wait_started)
                    if remaining_runtime <= 0:
                        raise subprocess.TimeoutExpired(cmd, max_runtime)

                    try:
                        stdout, stderr = self._current_process.communicate(timeout=min(timeout_slice, remaining_runtime))
                        returncode = self._current_process.returncode if self._current_process is not None else returncode
                        self._current_process = None
                        break
                    except subprocess.TimeoutExpired:
                        current_count = len(self._collect_existing_frame_indices(output_dir, frame_indices, output_format))
                        if current_count >= frame_count:
                            logger.info("[OK] All requested frames are present - stopping SDK wait")
                            completion_detected = True
                            process = self._current_process
                            try:
                                if process is not None:
                                    process.terminate()
                                    stdout, stderr = process.communicate(timeout=10)
                            except Exception:
                                if process is not None:
                                    process.kill()
                                    stdout, stderr = process.communicate()
                            finally:
                                if process is not None:
                                    returncode = process.returncode if process.returncode is not None else returncode
                                self._current_process = None
                            break

                        if no_change_duration >= stall_timeout:
                            raise subprocess.TimeoutExpired(cmd, no_change_duration)

                        logger.info(
                            "[INFO] SDK still progressing (%s/%s frames, %ss since last new frame) - extending wait",
                            current_count,
                            frame_count,
                            no_change_duration,
                        )
                
                # Log SDK output immediately for diagnostics
                if stdout:
                    stdout_text = stdout.decode('utf-8', errors='replace') if isinstance(stdout, bytes) else str(stdout)
                    logger.info(f"[SDK STDOUT]\n{stdout_text}")
                if stderr:
                    stderr_text = stderr.decode('utf-8', errors='replace') if isinstance(stderr, bytes) else str(stderr)
                    logger.warning(f"[SDK STDERR]\n{stderr_text}")
                
                current_count = len(self._collect_existing_frame_indices(output_dir, frame_indices, output_format))
                if current_count >= frame_count:
                    completion_detected = True

                if returncode != 0 and not completion_detected:
                    # Handle specific error codes
                    if returncode == 3221225781 or returncode == -1073741515: # 0xC0000135
                        error_msg = (
                            "MediaSDK failed to start (Missing DLL Dependency).\n"
                            "This usually means:\n"
                            "1. NVIDIA Drivers are missing or outdated (nvcuda.dll required)\n"
                            "2. Visual C++ Redistributable 2015-2022 is missing\n"
                            "3. System DLLs are corrupted\n"
                            "Please install latest NVIDIA Drivers and VC++ Redistributable."
                        )
                        logger.error(f"[CRITICAL] {error_msg}")
                        raise RuntimeError(error_msg)
                    elif returncode == 3221225477 or returncode == -1073741819: # 0xC0000005
                        error_msg = "MediaSDK crashed (Access Violation). Likely GPU driver incompatibility."
                        logger.error(f"[CRITICAL] {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)
                
                logger.info("[OK] MediaSDK extraction completed successfully!")
                if stdout:
                    logger.debug(f"SDK output: {stdout}")
                    
            except subprocess.TimeoutExpired:
                logger.warning("[WARNING] SDK watchdog triggered before extraction completed")
                current_count = len(self._collect_existing_frame_indices(output_dir, frame_indices, output_format))

                if current_count >= frame_count:
                    logger.info(f"[OK] All {current_count} frames extracted - SDK completed despite timeout")
                    completion_detected = True
                    # Kill the waiting process
                    try:
                        self._current_process.terminate()
                        self._current_process.wait(timeout=5)
                    except:
                        self._current_process.kill()
                        self._current_process.wait()
                    finally:
                        self._current_process = None
                else:
                    missing_indices = self._get_missing_frame_indices(output_dir, frame_indices, output_format)
                    logger.error(f"[ERROR] Only {current_count}/{frame_count} frames extracted before watchdog timeout")
                    try:
                        self._current_process.kill()
                        self._current_process.wait()
                    except:
                        pass
                    finally:
                        self._current_process = None
                    raise IncompleteSDKExtractionError(
                        f"SDK stalled before completing extraction: {current_count}/{frame_count} frames",
                        expected_count=frame_count,
                        actual_count=current_count,
                        missing_indices=missing_indices,
                    )
                
                logger.debug(f"SDK output: {stdout}")
            
            extracted_frames = self._validate_requested_frame_set(
                output_dir,
                frame_indices,
                output_format,
                operation_label="SDK extraction",
            )

            if progress_callback:
                progress_callback(100)

            if output_rotation:
                self._rotate_output_frames(extracted_frames, output_rotation)
            
            # Verify image file sizes (detect black/empty images)
            if extracted_frames:
                sample_path = Path(extracted_frames[0])
                if sample_path.exists():
                    file_size = sample_path.stat().st_size
                    logger.info(f"[VERIFY] Sample image: {sample_path.name} ({file_size:,} bytes)")
                    if file_size < 10000:  # Less than 10KB is suspiciously small
                        logger.error(f"[ERROR] Extracted images are suspiciously small ({file_size} bytes) - likely black/empty!")
                        logger.error("[ERROR] This usually means SDK failed to stitch properly. Check SDK output above.")
            
            return extracted_frames
            
        except subprocess.CalledProcessError as e:
            # Check for known SDK issues
            error_msg = e.stderr if e.stderr else str(e)
            
            # Exit code 3221225786 (0xC000013A) = STATUS_CONTROL_C_EXIT (process terminated)
            # Exit code -1073741819 (0xC0000005) = STATUS_ACCESS_VIOLATION (crash)
            if 'FrameTypeTimelapseQuat failed' in error_msg:
                logger.warning("[WARNING] SDK metadata read error (timelapse data missing)")
                logger.warning("INFO: This is a known SDK issue with some .insv files")
            elif 'no device found' in error_msg:
                logger.warning("[WARNING] SDK failed: No GPU device found (expected on CPU-only systems)")
                raise RuntimeError("MediaSDK requires GPU (no device found)")
            elif e.returncode == 3221225786:
                logger.warning("[WARNING] SDK process was terminated (user cancel or timeout)")
            else:
                logger.error(f"[ERROR] MediaSDK extraction failed (exit code {e.returncode})")

            if e.stderr:
                logger.error("[SDK STDERR]\n%s", e.stderr.rstrip())
            else:
                logger.error(f"Error details: {error_msg}")
            raise RuntimeError(f"MediaSDK extraction failed: {error_msg[:200]}")
        finally:
            # Always clean up temporary patched INSV files
            for tmp_path in _patched_temps:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    
    def stop(self):
        """Stop currently running SDK process."""
        if self._current_process and self._current_process.poll() is None:
            logger.warning("[WARNING] Terminating SDK process...")
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=5)
                logger.info("[OK] SDK process terminated")
            except:
                logger.warning("[WARNING] Force killing SDK process...")
                self._current_process.kill()
                self._current_process.wait()
            finally:
                self._current_process = None
    
    def _get_video_info(self, video_path: Path) -> Dict:
        """Get video metadata using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Could not open video with OpenCV: {video_path}")
            return {'total_frames': 0, 'fps': 24.0, 'duration': 0}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'width': width,
            'height': height
        }
    
    def _detect_input_files(self, input_path: Path) -> List[str]:
        """
        Detect if input is dual-track or single-track.
        
        Insta360 5.7K+ videos use dual-track:
        - VID_XXX_00_XXX.insv (main track)
        - VID_XXX_10_XXX.insv (second track)
        
        X4 cameras use single-track with embedded dual video streams.
        """
        input_files = [str(input_path)]
        
        # Check for dual-track pattern: _00_ → _10_
        if "_00_" in input_path.name:
            second_track = input_path.parent / input_path.name.replace("_00_", "_10_")
            if second_track.exists():
                input_files.append(str(second_track))
                logger.info(f"Detected dual-track video: {input_path.name} + {second_track.name}")
        
        return input_files

    # Camtype tokens that are NOT in the SDK's dispatch table and their nearest
    # working replacements (determined by exhaustive sweep 0-200, May 2025).
    # Both replacements are in the top quality tier (~5710 KB/frame output).
    _UNSUPPORTED_CAMTYPE_PATCHES = {
        b'_112_': b'_113_',  # A1 offset field: type 112 unsupported → 113 (nearest valid)
        b'_155_': b'_156_',  # A1 original_offset field: type 155 unsupported → 156
    }
    _CAMTYPE_TAIL_SCAN = 300 * 1024  # bytes to read from EOF (covers the INSV trailer)

    def _source_requires_sdk_recovery(self, input_files: List[str]) -> bool:
        """Return True only for A1-style sources that need the SDK retry workaround."""
        for path in input_files:
            if self._tail_contains_any_token(Path(path), self._UNSUPPORTED_CAMTYPE_PATCHES):
                return True
        return False

    def _should_use_dense_overlap_strategy(
        self,
        source_requires_recovery: bool,
        frame_indices: List[int],
        frame_interval: int,
        sdk_options: Optional[Dict] = None,
    ) -> bool:
        """Return True when a dense all-frame export should use overlap recovery."""
        if not source_requires_recovery:
            return False

        if not frame_indices or frame_interval != 1:
            return False

        if sdk_options and sdk_options.get('_dense_overlap_internal'):
            return False

        enabled = True if sdk_options is None else sdk_options.get('enable_dense_overlap_recovery', True)
        return bool(enabled) and len(frame_indices) > 12

    def _should_use_sparse_retry_strategy(
        self,
        source_requires_recovery: bool,
        frame_indices: List[int],
        frame_interval: int,
        sdk_options: Optional[Dict] = None,
    ) -> bool:
        """Return True when sparse frame extraction should retry each target frame."""
        if not source_requires_recovery:
            return False

        if not frame_indices or frame_interval <= 1:
            return False

        if sdk_options and sdk_options.get('_sparse_retry_internal'):
            return False

        enabled = True if sdk_options is None else sdk_options.get('enable_sparse_retry_recovery', True)
        return bool(enabled) and len(frame_indices) <= 120

    def _get_missing_frame_indices(
        self,
        output_dir: Path,
        frame_indices: List[int],
        output_format: str,
    ) -> List[int]:
        existing = self._collect_existing_frame_indices(output_dir, frame_indices, output_format)
        return [frame_index for frame_index in frame_indices if frame_index not in existing]

    def _validate_requested_frame_set(
        self,
        output_dir: Path,
        frame_indices: List[int],
        output_format: str,
        *,
        operation_label: str,
    ) -> List[str]:
        extracted_frames = self._collect_frame_paths(output_dir, frame_indices, output_format)
        missing_indices = self._get_missing_frame_indices(output_dir, frame_indices, output_format)
        expected_count = len(frame_indices)
        actual_count = len(extracted_frames)

        if actual_count == 0:
            raise RuntimeError(f"{operation_label} produced no output frames in {output_dir}")

        if missing_indices:
            preview = ", ".join(str(index) for index in missing_indices[:10])
            if len(missing_indices) > 10:
                preview += ", ..."
            raise IncompleteSDKExtractionError(
                f"{operation_label} incomplete: {actual_count}/{expected_count} frames (missing: {preview})",
                expected_count=expected_count,
                actual_count=actual_count,
                missing_indices=missing_indices,
            )

        logger.info(f"[OK] Extracted {actual_count}/{expected_count} frames (100.0%)")
        return extracted_frames

    def _extract_dense_sequence_with_overlap(
        self,
        input_path: str,
        output_dir: str,
        fps: float,
        quality: str,
        resolution: Optional[Tuple[int, int]],
        output_format: str,
        start_time: float,
        end_time: Optional[float],
        progress_callback: Optional[Callable[[int], None]],
        sdk_options: Optional[Dict],
        video_fps: float,
        frame_indices: List[int],
    ) -> List[str]:
        """Recover dense all-frame exports with overlapping SDK windows."""
        logger.info("[SDK] Using dense overlap recovery for all-frame extraction")

        overlap_step = 3
        overlap_window = 6
        if sdk_options:
            overlap_step = max(1, int(sdk_options.get('dense_overlap_step', overlap_step)))
            overlap_window = max(overlap_step, int(sdk_options.get('dense_overlap_window', overlap_window)))

        nested_options = dict(sdk_options or {})
        nested_options['_dense_overlap_internal'] = True
        nested_options['enable_dense_overlap_recovery'] = False

        output_path = Path(output_dir)
        expected_count = len(frame_indices)
        first_index = frame_indices[0]
        last_index_exclusive = frame_indices[-1] + 1

        for window_start in range(first_index, last_index_exclusive, overlap_step):
            window_end = min(window_start + overlap_window, last_index_exclusive)
            window_start_time = window_start / video_fps
            window_end_time = window_end / video_fps

            logger.info(
                "[SDK] Dense overlap window %s-%s (%0.3fs-%0.3fs)",
                window_start,
                window_end - 1,
                window_start_time,
                window_end_time,
            )

            try:
                self.extract_frames(
                    input_path=input_path,
                    output_dir=output_dir,
                    fps=fps,
                    quality=quality,
                    resolution=resolution,
                    output_format=output_format,
                    start_time=window_start_time,
                    end_time=window_end_time,
                    progress_callback=None,
                    sdk_options=nested_options,
                )
            except Exception as exc:
                logger.warning(
                    "[SDK] Dense overlap window %s-%s failed: %s",
                    window_start,
                    window_end - 1,
                    exc,
                )

            if progress_callback:
                recovered = len(self._collect_frame_paths(output_path, frame_indices, output_format))
                progress_callback(int(recovered / expected_count * 100))

        if progress_callback:
            progress_callback(100)

        return self._validate_requested_frame_set(
            output_path,
            frame_indices,
            output_format,
            operation_label="SDK dense overlap recovery",
        )

    def _extract_sparse_indices_with_retry(
        self,
        input_path: str,
        output_dir: str,
        fps: float,
        quality: str,
        resolution: Optional[Tuple[int, int]],
        output_format: str,
        progress_callback: Optional[Callable[[int], None]],
        sdk_options: Optional[Dict],
        video_fps: float,
        frame_indices: List[int],
        total_frames: int,
    ) -> List[str]:
        """Extract sparse frame sets by retrying only the frames missing after a preflight batch pass."""
        logger.info("[SDK] Using sparse retry recovery for %d requested frames", len(frame_indices))

        output_path = Path(output_dir)
        frame_interval = max(1, int(video_fps / fps))
        nested_options = dict(sdk_options or {})
        nested_options['_sparse_retry_internal'] = True
        nested_options['enable_sparse_retry_recovery'] = False
        nested_options['enable_dense_overlap_recovery'] = False

        retry_input_files = self._detect_input_files(Path(input_path))
        retry_temp_files = self._patch_insv_camtype_if_needed(retry_input_files)
        nested_options['_input_files_override'] = retry_input_files

        recovered_paths: List[str] = []
        try:
            # Try one batched sparse extraction first and only retry the frames that are still missing.
            batch_start_time = frame_indices[0] / video_fps
            batch_end_frame = min(total_frames, frame_indices[-1] + frame_interval)
            batch_end_time = batch_end_frame / video_fps
            logger.info(
                "[SDK] Sparse recovery preflight batch %s-%s (%0.3fs-%0.3fs)",
                frame_indices[0],
                max(frame_indices[-1], batch_end_frame - 1),
                batch_start_time,
                batch_end_time,
            )

            try:
                self.extract_frames(
                    input_path=input_path,
                    output_dir=output_dir,
                    fps=fps,
                    quality=quality,
                    resolution=resolution,
                    output_format=output_format,
                    start_time=batch_start_time,
                    end_time=batch_end_time,
                    progress_callback=None,
                    sdk_options=nested_options,
                )
            except Exception as exc:
                logger.warning("[SDK] Sparse recovery preflight batch failed: %s", exc)

            recovered_paths = self._collect_frame_paths(output_path, frame_indices, output_format)
            recovered_indices = self._collect_existing_frame_indices(output_path, frame_indices, output_format)
            logger.info(
                "[SDK] Sparse recovery preflight satisfied %d/%d requested frames",
                len(recovered_indices),
                len(frame_indices),
            )

            missing_indices = [frame_index for frame_index in frame_indices if frame_index not in recovered_indices]
            for missing_position, frame_index in enumerate(missing_indices, start=1):
                best_candidate = self._recover_single_frame_with_retry(
                    input_path=input_path,
                    output_dir=output_path,
                    frame_index=frame_index,
                    fps=fps,
                    video_fps=video_fps,
                    quality=quality,
                    resolution=resolution,
                    output_format=output_format,
                    sdk_options=nested_options,
                    total_frames=total_frames,
                )
                if best_candidate:
                    recovered_paths.append(best_candidate)

                if progress_callback:
                    completed = len(recovered_indices) + missing_position
                    progress_callback(int(completed / len(frame_indices) * 100))

            if progress_callback:
                progress_callback(100)

            return self._validate_requested_frame_set(
                output_path,
                frame_indices,
                output_format,
                operation_label="SDK sparse retry recovery",
            )
        finally:
            for tmp_path in retry_temp_files:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _collect_existing_frame_indices(
        self,
        output_dir: Path,
        frame_indices: List[int],
        output_format: str,
    ) -> set[int]:
        """Return the requested frame indices that already exist in the output directory."""
        ext = f".{output_format.lower()}"
        existing = set()
        for frame_index in frame_indices:
            if (output_dir / f"{frame_index}{ext}").exists():
                existing.add(frame_index)
        return existing

    def _recover_single_frame_with_retry(
        self,
        input_path: str,
        output_dir: Path,
        frame_index: int,
        fps: float,
        video_fps: float,
        quality: str,
        resolution: Optional[Tuple[int, int]],
        output_format: str,
        sdk_options: Dict,
        total_frames: int,
    ) -> Optional[str]:
        """Retry a single target frame using multiple SDK-only extraction windows."""
        ext = f".{output_format.lower()}"
        final_path = output_dir / f"{frame_index}{ext}"
        attempt_windows = [
            (frame_index, frame_index + 1),
            (max(0, frame_index - 3), min(total_frames, frame_index + 3)),
            (max(0, frame_index - 6), min(total_frames, frame_index)),
            (frame_index, min(total_frames, frame_index + 6)),
        ]

        best_temp_path: Optional[Path] = None
        best_score = float('-inf')

        for attempt_number, (window_start, window_end) in enumerate(attempt_windows, start=1):
            if window_end <= window_start:
                continue

            temp_dir = Path(tempfile.mkdtemp(prefix=f"sdk_retry_{frame_index:04d}_{attempt_number}_", dir=str(output_dir.parent)))
            try:
                self.extract_frames(
                    input_path=input_path,
                    output_dir=str(temp_dir),
                    fps=fps,
                    quality=quality,
                    resolution=resolution,
                    output_format=output_format,
                    start_time=window_start / video_fps,
                    end_time=window_end / video_fps,
                    progress_callback=None,
                    sdk_options=sdk_options,
                )
            except Exception as exc:
                logger.warning("[SDK] Retry attempt %d failed for frame %d: %s", attempt_number, frame_index, exc)

            candidate_path = temp_dir / f"{frame_index}{ext}"
            if not candidate_path.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

            candidate_score = self._score_extracted_frame(candidate_path)
            logger.info("[SDK] Frame %d attempt %d score: %.2f", frame_index, attempt_number, candidate_score)
            if candidate_score > best_score:
                if best_temp_path and best_temp_path.parent.exists():
                    shutil.rmtree(best_temp_path.parent, ignore_errors=True)
                best_score = candidate_score
                best_temp_path = candidate_path
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if not best_temp_path or not best_temp_path.exists():
            logger.warning("[SDK] Could not recover frame %d", frame_index)
            return None

        shutil.copy2(best_temp_path, final_path)
        shutil.rmtree(best_temp_path.parent, ignore_errors=True)
        return str(final_path)

    def _score_extracted_frame(self, image_path: Path) -> float:
        """Score an extracted frame so gray/corrupted results rank lower than valid images."""
        image = cv2.imread(str(image_path))
        if image is None:
            return float('-inf')

        height = image.shape[0]
        lower_half = image[height // 2 :, :, :]
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)

        detail = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        saturation = float(hsv[:, :, 1].mean())
        return detail * 2.0 + contrast * 8.0 + saturation

    def _determine_output_rotation(self, input_files: List[str], sdk_options: Optional[Dict] = None) -> int:
        """Determine whether extracted SDK outputs need a post-rotation."""
        rotation_override = None
        auto_rotate = True

        if sdk_options:
            rotation_override = sdk_options.get('output_rotation')
            auto_rotate = sdk_options.get('auto_rotate_output', True)

        if rotation_override is not None:
            rotation = int(rotation_override) % 360
            if rotation in (0, 90, 180, 270):
                logger.info(f"[Orientation] Using explicit output rotation override: {rotation}°")
                return rotation
            logger.warning(f"[Orientation] Ignoring unsupported output rotation override: {rotation_override}")

        if not auto_rotate:
            return 0

        for path in input_files:
            if self._tail_contains_any_token(Path(path), self._UNSUPPORTED_CAMTYPE_PATCHES):
                logger.info("[Orientation] Detected A1-style camtype trailer; rotating SDK outputs 180°")
                return 180

        return 0

    def _tail_contains_any_token(self, path: Path, token_map: Dict[bytes, bytes]) -> bool:
        """Check whether the file trailer contains any token from token_map."""
        if not path.exists() or path.suffix.lower() not in ('.insv', '.mp4'):
            return False

        try:
            file_size = path.stat().st_size
            tail_read = min(self._CAMTYPE_TAIL_SCAN, file_size)
            with open(path, 'rb') as handle:
                handle.seek(-tail_read, 2)
                tail = handle.read()
            return any(token in tail for token in token_map)
        except OSError as exc:
            logger.warning(f"[Orientation] Could not inspect trailer for {path.name}: {exc}")
            return False

    def _rotate_output_frames(self, frame_paths: List[str], rotation: int) -> None:
        """Rotate extracted SDK output frames in place."""
        cv_codes = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        code = cv_codes.get(rotation % 360)
        if code is None:
            return

        rotated = 0
        for frame_path in frame_paths:
            try:
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                image = cv2.rotate(image, code)
                cv2.imwrite(str(frame_path), image)
                rotated += 1
            except Exception as exc:
                logger.warning(f"[Orientation] Failed to rotate {frame_path}: {exc}")

        logger.info(f"[Orientation] Rotated {rotated}/{len(frame_paths)} SDK output frames by {rotation}°")

    def _patch_insv_camtype_if_needed(self, input_files: List[str]) -> List[str]:
        """
        Detect unsupported camtype tokens in INSV calibration strings and
        rewrite the file to a temp copy with the nearest supported type.

        Cameras like the Antigravity A1 embed lens calibrations that end in
        ``_112_`` (offset) and ``_155_`` (original_offset).  Neither value is
        in the SDK's dispatch table, producing:
            CameraLensType:155  /  no implemention!

        Strategy (safe, minimal mutation):
          1. Read only the last 300 KB (the INSV trailer region).
          2. If a known-bad token is present, replace it in-memory.
          3. Write head (unchanged) + patched tail to a sibling temp file.
          4. Return a list of temp paths that the caller must delete.

        Returns:
            List of temporary file paths to delete after SDK call.
        """
        import tempfile as _tempfile
        temps: List[str] = []

        for i, path in enumerate(input_files):
            p = Path(path)
            if p.suffix.lower() not in ('.insv', '.mp4'):
                continue

            file_size = p.stat().st_size
            tail_read = min(self._CAMTYPE_TAIL_SCAN, file_size)

            with open(p, 'rb') as f:
                f.seek(-tail_read, 2)
                tail = f.read()

            if not any(tok in tail for tok in self._UNSUPPORTED_CAMTYPE_PATCHES):
                continue  # No patching needed

            logger.info(
                "[CAMTYPE PATCH] Unsupported lens type detected in '%s' — "
                "creating patched temp copy...", p.name
            )

            patched_tail = tail
            for old, new in self._UNSUPPORTED_CAMTYPE_PATCHES.items():
                count = patched_tail.count(old)
                if count:
                    patched_tail = patched_tail.replace(old, new)
                    logger.info(
                        "[CAMTYPE PATCH]   %s → %s  (%d occurrence(s))",
                        old.decode(), new.decode(), count
                    )

            # Write temp file: verbatim head + patched tail
            tmp = _tempfile.NamedTemporaryFile(
                suffix='.insv', dir=p.parent, delete=False,
                prefix='_campatched_'
            )
            try:
                head_size = file_size - tail_read
                with open(p, 'rb') as fsrc:
                    remaining = head_size
                    buf_size = 4 * 1024 * 1024
                    while remaining > 0:
                        chunk = fsrc.read(min(buf_size, remaining))
                        if not chunk:
                            break
                        tmp.write(chunk)
                        remaining -= len(chunk)
                tmp.write(patched_tail)
            finally:
                tmp.close()

            input_files[i] = tmp.name
            temps.append(tmp.name)
            logger.info("[CAMTYPE PATCH] Patched copy: %s", tmp.name)

        return temps

    def _build_extraction_command(
        self,
        input_files: List[str],
        output_dir: Path,
        frame_indices: List[int],
        quality: str,
        resolution: Optional[Tuple[int, int]],
        output_format: str,
        sdk_options: Optional[Dict] = None
    ) -> List[str]:
        """Build MediaSDK command line arguments."""
        cmd = [str(self.demo_exe)]
        
        # SDK 3.1.x: Set model root directory (replaces individual model paths)
        # CRITICAL: Must use ABSOLUTE path since SDK runs from bin/ directory
        models_dir = self.sdk_base / "models"
        if models_dir.exists():
            cmd.extend(["-model_dir", str(models_dir.resolve())])
            logger.info(f"[SDK 3.1.x] Model directory set: {models_dir.resolve()}")
        
        # Input files (SetInputPath)
        cmd.extend(["-inputs"] + input_files)
        
        # Output directory for image sequence (SetImageSequenceInfo)
        cmd.extend(["-image_sequence_dir", str(output_dir)])
        
        # Image format (IMAGE_TYPE enum)
        image_type = "png" if output_format.lower() == "png" else "jpg"
        cmd.extend(["-image_type", image_type])
        
        # Frame indices to extract (SetExportFrameSequence)
        # CRITICAL: Use dash-separated format (NOT comma-separated)
        # Correct: "0-24-48" | Wrong: "0,24,48"
        frame_seq = "-".join(str(i) for i in frame_indices)
        cmd.extend(["-export_frame_index", frame_seq])  # Singular, not plural!
        
        # Quality preset (copy so we don't mutate the global dict)
        preset = dict(QUALITY_PRESETS.get(quality, QUALITY_PRESETS['best']))

        # Apply any caller-supplied overrides (sdk_options)
        if sdk_options:
            preset.update(sdk_options)
            logger.info(f"[SDK] Applied sdk_options overrides: {list(sdk_options.keys())}")

        # Determine stitch type
        stitch_type_key = preset['stitch_type']
        
        # Select AI model based on preset preference
        # v2 model produces PERFECT results (tested), prefer it when available
        use_v2 = preset.get('use_ai_model_v2', False)
        if use_v2 and self.ai_model_v2.exists():
            ai_model = self.ai_model_v2
            logger.info(f"[OK] Using AI Model V2 (best quality): {ai_model.name}")
        elif self.ai_model_v1.exists():
            ai_model = self.ai_model_v1
            logger.info(f"[OK] Using AI Model V1: {ai_model.name}")
        else:
            ai_model = None
            logger.warning("[WARNING] No AI model found")
        
        # SDK 3.1.x: AI model auto-discovered from model_dir (SetModelFileRootDir)
        # NOTE: -ai_stitching_model is DEPRECATED in SDK 3.1.x and conflicts with -model_dir
        # The SDK automatically selects the correct AI model based on camera type
        
        # Set stitch type (SetStitchType)
        stitch_type_value = STITCH_TYPES.get(stitch_type_key, 'dynamicstitch')
        cmd.extend(["-stitch_type", stitch_type_value])
        logger.info(f"[SDK] Stitch Method: {stitch_type_value.upper()}")
        
        # CRITICAL: Enable chromatic calibration for seamless blending
        if preset.get('enable_stitchfusion', False):
            cmd.append("-enable_stitchfusion")
        
        # ---- Stabilization (EnableFlowState) ----
        # Direction Lock REQUIRES FlowState to be active; force it on if needed.
        want_direction_lock = preset.get('enable_direction_lock', False)
        enable_flowstate = preset.get('enable_flowstate', False) or want_direction_lock
        if enable_flowstate:
            cmd.append("-enable_flowstate")
            logger.info("[SDK] FlowState stabilization ENABLED")

        # Direction Lock (EnableDirectionLock) - locks horizon like Insta360 app
        # SDK requirement: flowstate must be enabled first (enforced above)
        if want_direction_lock:
            cmd.append("-enable_directionlock")
            logger.info("[SDK] Direction Lock ENABLED (horizon stabilized)")

        # Color Plus (EnableColorPlus) - SDK 3.1.x auto-finds model
        if preset.get('enable_colorplus', False):
            cmd.append("-enable_colorplus")
            logger.info(f"[SDK] Color Plus ENABLED (strength={preset.get('colorplus_strength', 0.3):.2f})")
        
        # Denoise (EnableSequenceDenoise) - SDK 3.1.x auto-finds model
        if preset.get('enable_denoise', False):
            cmd.append("-enable_denoise")
        
        # Defringe (EnableDefringe) - SDK 3.1.x auto-finds model
        if preset.get('enable_defringe', False):
            cmd.append("-enable_defringe")

        # ---- Color Correction (all values passed only when non-zero) ----
        # Bulk extraction uses the SDK's native colour pipeline by default.
        # This avoids a second full-image OpenCV rewrite pass over every 8K frame,
        # which can stall large jobs before Stage 3 starts.
        use_native_color_corrections = bool(preset.get('_use_sdk_native_color_corrections', True))
        _active = {k: preset.get(k, 0) for k in _NATIVE_COLOR_KEYS if preset.get(k, 0) != 0}
        if _active:
            if use_native_color_corrections:
                for key, value in _active.items():
                    cmd.extend([f"-{key}", str(int(value))])
                logger.info(f"[SDK] Applying native color corrections during extraction: {_active}")
            else:
                logger.info(f"[SDK] Color values will be applied via OpenCV post-process: {_active}")
        
        # Output resolution (SetOutputSize - must be 2:1 ratio)
        if resolution:
            width, height = resolution
            cmd.extend(["-output_size", f"{width}x{height}"])
        else:
            # Default to 8K for best quality
            cmd.extend(["-output_size", "7680x3840"])
        
        # ===== GPU ACCELERATION FLAGS (CRITICAL for performance) =====
        # Reference: https://github.com/Insta360Develop/Desktop-MediaSDK-Cpp
        
        disable_cuda = bool(preset.get('disable_cuda', False))
        cmd.extend(["-disable_cuda", "true" if disable_cuda else "false"])
        logger.info(f"[SDK GPU] CUDA acceleration {'DISABLED' if disable_cuda else 'ENABLED'} (disable_cuda={'true' if disable_cuda else 'false'})")
        
        # 2. USE HARDWARE ENCODER (not software)
        enable_soft_encode = bool(preset.get('enable_soft_encode', False))
        cmd.extend(["-enable_soft_encode", "true" if enable_soft_encode else "false"])
        logger.info(f"[SDK GPU] {'Software' if enable_soft_encode else 'Hardware'} encoder enabled")
        
        # 3. USE HARDWARE DECODER (not software)
        enable_soft_decode = bool(preset.get('enable_soft_decode', False))
        cmd.extend(["-enable_soft_decode", "true" if enable_soft_decode else "false"])
        logger.info(f"[SDK GPU] {'Software' if enable_soft_decode else 'Hardware'} decoder enabled")
        
        # 4. IMAGE PROCESSING ACCELERATION (auto = GPU if available)
        image_processing_accel = str(preset.get('image_processing_accel', 'auto')).strip().lower() or 'auto'
        cmd.extend(["-image_processing_accel", image_processing_accel])
        logger.info(f"[SDK GPU] Image processing acceleration: {image_processing_accel.upper()}")
        
        # 5. OPTIONAL: Enable H.265 encoder for better compression (GPU-accelerated)
        #    Only if output is video (not needed for image sequences, but doesn't hurt)
        # cmd.append("-enable_h265_encoder")  # Uncomment if you want H.265
        
        logger.info("[SDK GPU] All GPU acceleration features enabled")
        
        return cmd
    
    def _collect_frame_paths(
        self,
        output_dir: Path,
        frame_indices: List[int],
        output_format: str
    ) -> List[str]:
        """
        Collect paths to extracted frames.
        
        MediaSDK names files by frame index: 0.jpg, 10.jpg, 20.jpg, etc.
        """
        ext = f".{output_format.lower()}"
        extracted = []
        missing = []

        # First try exact index-based filenames (0.jpg, 12.jpg, ...)
        for idx in frame_indices:
            frame_path = output_dir / f"{idx}{ext}"
            if frame_path.exists():
                extracted.append(str(frame_path))
            else:
                missing.append(idx)

        # If no files were collected by index, try a more permissive discovery
        if len(extracted) == 0:
            logger.debug("No index-named frames found - trying permissive image discovery")
            image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            permissive = sorted([str(p) for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
            if permissive:
                logger.info(f"Found {len(permissive)} image files via permissive discovery")
                return permissive

        # Log summary instead of per-frame warnings
        if missing:
            if len(missing) <= 10:
                logger.debug(f"Missing frames: {missing}")
            else:
                logger.debug(f"Missing {len(missing)} frames (first 10: {missing[:10]}...)")

        return extracted


# Convenience functions for batch_orchestrator.py integration

def extract_frames_sdk(
    input_path: str,
    output_dir: str,
    fps: float = 1.0,
    quality: str = 'best',
    resolution: Optional[Tuple[int, int]] = None,
    output_format: str = 'jpg',
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    progress_callback: Optional[Callable[[int], None]] = None
) -> List[str]:
    """
    Extract frames using MediaSDK (convenience function).
    
    Args:
        input_path: Path to .insv video
        output_dir: Output directory
        fps: Extraction rate
        quality: 'best', 'good', 'draft'
        resolution: Output resolution (None = 8K default)
        output_format: 'jpg' or 'png'
        start_time: Start time in seconds
        end_time: End time in seconds (None = video end)
        progress_callback: Progress callback
    
    Returns:
        List of extracted frame paths
    
    Raises:
        RuntimeError: If SDK not available or extraction fails
    """
    extractor = SDKExtractor()
    
    if not extractor.is_available():
        raise RuntimeError(
            "MediaSDK not available. Please install SDK or use FFmpeg fallback.\n"
            f"Expected SDK path: {DEFAULT_SDK_PATH}"
        )
    
    return extractor.extract_frames(
        input_path=input_path,
        output_dir=output_dir,
        fps=fps,
        quality=quality,
        resolution=resolution,
        output_format=output_format,
        start_time=start_time,
        end_time=end_time,
        progress_callback=progress_callback
    )


def is_sdk_available() -> bool:
    """Check if MediaSDK is available."""
    extractor = SDKExtractor()
    return extractor.is_available()
