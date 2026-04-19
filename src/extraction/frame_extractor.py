"""
Frame Extraction Module for 360FrameTools
Simplified extractor supporting FFmpeg method.
OpenCV fallback methods removed for size optimization - use FFmpeg or SDK.

Based on Extraction Module but adapted for unified pipeline.
"""

import cv2
import json
import subprocess
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Callable, List
import shutil

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


INSV_TRAILER_SCAN_BYTES = 1024 * 1024
INSV_TRAILER_CAMERA_STRINGS = [
    (b'antigravity a1', 'Antigravity A1'),
    (b'insta360 x5', 'Insta360 X5'),
    (b'insta360 x4', 'Insta360 X4'),
    (b'insta360 x3', 'Insta360 X3'),
    (b'insta360 one x2', 'Insta360 ONE X2'),
    (b'insta360 one x', 'Insta360 ONE X'),
]
INSV_A1_CAMTYPE_TOKENS = (b'_112_', b'_155_')


def _find_bundled_ffmpeg() -> Optional[str]:
    """Find FFmpeg in bundled location (for PyInstaller builds)."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = Path(sys._MEIPASS)
        
        # Check in ffmpeg subfolder
        bundled_ffmpeg = base_path / 'ffmpeg' / 'ffmpeg.exe'
        if bundled_ffmpeg.exists():
            return str(bundled_ffmpeg)
        
        # Check in root of _internal
        internal_ffmpeg = base_path / 'ffmpeg.exe'
        if internal_ffmpeg.exists():
            return str(internal_ffmpeg)
    else:
        # Running from source - check local ffmpeg folder
        source_path = Path(__file__).parent.parent.parent
        local_ffmpeg = source_path / 'ffmpeg' / 'ffmpeg.exe'
        if local_ffmpeg.exists():
            return str(local_ffmpeg)
    
    return None


def _find_bundled_ffprobe() -> Optional[str]:
    """Find FFprobe in bundled location (for PyInstaller builds)."""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
        
        bundled_ffprobe = base_path / 'ffmpeg' / 'ffprobe.exe'
        if bundled_ffprobe.exists():
            return str(bundled_ffprobe)
        
        internal_ffprobe = base_path / 'ffprobe.exe'
        if internal_ffprobe.exists():
            return str(internal_ffprobe)
    else:
        source_path = Path(__file__).parent.parent.parent
        local_ffprobe = source_path / 'ffmpeg' / 'ffprobe.exe'
        if local_ffprobe.exists():
            return str(local_ffprobe)
    
    return None


class FrameExtractor:
    """
    Extract frames from video files (.INSV, .mp4, etc.)
    Requires FFmpeg for frame extraction.
    OpenCV only used for video metadata reading.
    """
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize FrameExtractor.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (auto-detect if None)
        """
        # Use settings manager for FFmpeg path
        settings = get_settings()
        
        if ffmpeg_path:
            self.ffmpeg_path = ffmpeg_path
        else:
            configured_ffmpeg = settings.get_ffmpeg_path()
            self.ffmpeg_path = str(configured_ffmpeg) if configured_ffmpeg else self._find_ffmpeg()
        
        self.ffprobe_path = self._find_ffprobe()
        self.has_ffmpeg = self.ffmpeg_path is not None
        self.is_cancelled = False
        
        if self.has_ffmpeg:
            logger.info(f"FFmpeg found at: {self.ffmpeg_path}")
        else:
            logger.error("FFmpeg not found - FFmpeg is REQUIRED for frame extraction")
        
        if self.ffprobe_path:
            logger.info(f"FFprobe found at: {self.ffprobe_path}")
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Auto-detect FFmpeg installation - check bundled first, then system PATH."""
        # Check bundled location first
        bundled = _find_bundled_ffmpeg()
        if bundled:
            logger.info(f"Found bundled FFmpeg at: {bundled}")
            return bundled
        
        # Fallback to system PATH
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            logger.info(f"Found system FFmpeg at: {ffmpeg_path}")
        return ffmpeg_path
    
    def _find_ffprobe(self) -> Optional[str]:
        """Auto-detect FFprobe installation - check bundled first, then system PATH."""
        # Check bundled location first
        bundled = _find_bundled_ffprobe()
        if bundled:
            logger.info(f"Found bundled FFprobe at: {bundled}")
            return bundled
        
        # Fallback to system PATH
        ffprobe_path = shutil.which('ffprobe')
        if ffprobe_path:
            logger.info(f"Found system FFprobe at: {ffprobe_path}")
        return ffprobe_path
    
    def cancel(self):
        """Cancel ongoing extraction operation"""
        self.is_cancelled = True
        logger.info("FrameExtractor cancellation requested")
    
    def extract_frames(self, input_file: str, output_dir: str, 
                      fps: float = 1.0, 
                      method: str = 'ffmpeg_stitched',
                      start_time: float = 0.0,
                      end_time: Optional[float] = None,
                      lens_mode: str = 'both',
                      progress_callback: Optional[Callable] = None) -> Dict:
        """
        Extract frames from video file with dual-lens support and time range selection.
        
        Args:
            input_file: Path to input video (.insv, .mp4, etc.)
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract
            method: Extraction method (see EXTRACTION_METHODS in config/defaults.py)
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (None = extract until end)
            lens_mode: 'both', 'lens1', or 'lens2' for dual-stream files
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Dictionary with results: {success, frame_count, output_files, error, lens_outputs}
        """
        
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            return {
                'success': False,
                'error': f"Input file not found: {input_file}",
                'frame_count': 0,
                'output_files': []
            }
        
        # Log extraction parameters
        time_range_msg = ""
        if start_time > 0 or end_time is not None:
            end_msg = f"{end_time}s" if end_time else "end"
            time_range_msg = f" (time range: {start_time}s - {end_msg})"
        
        logger.info(f"Extracting frames from {input_path.name} at {fps} FPS{time_range_msg} (method: {method}, lens: {lens_mode})")
        
        try:
            # Check FFmpeg availability
            if not self.has_ffmpeg:
                return {
                    'success': False,
                    'error': 'FFmpeg is required but not found. Please install FFmpeg.',
                    'frame_count': 0,
                    'output_files': [],
                    'lens_outputs': {}
                }

            if method in {'ffmpeg', 'ffmpeg_stitched'} and input_path.suffix.lower() == '.insv':
                return {
                    'success': False,
                    'error': (
                        'FFmpeg stitched extraction is not supported for .insv input. '
                        'Use SDK Stitching for stitched frames, or FFmpeg dual-lens/lens-specific methods only for raw fisheye export.'
                    ),
                    'frame_count': 0,
                    'output_files': [],
                    'lens_outputs': {}
                }
            
            # Route to appropriate FFmpeg extraction method
            if method.startswith('ffmpeg_'):
                if method == 'ffmpeg_stitched':
                    return self._extract_with_ffmpeg_stitched(input_path, output_path, fps, start_time, end_time, progress_callback)
                elif method == 'ffmpeg_v360_dual':
                    return self._extract_with_ffmpeg_v360_dual(input_path, output_path, fps, start_time, end_time, progress_callback)
                elif method == 'ffmpeg_dual_lens':
                    return self._extract_dual_lens_ffmpeg(input_path, output_path, fps, start_time, end_time, 'both', progress_callback)
                elif method == 'ffmpeg_lens1':
                    return self._extract_dual_lens_ffmpeg(input_path, output_path, fps, start_time, end_time, 'lens1', progress_callback)
                elif method == 'ffmpeg_lens2':
                    return self._extract_dual_lens_ffmpeg(input_path, output_path, fps, start_time, end_time, 'lens2', progress_callback)
            elif method.startswith('opencv_'):
                # OpenCV methods removed for size optimization
                logger.error(f"OpenCV extraction methods have been removed. Use FFmpeg or SDK instead.")
                return {
                    'success': False,
                    'error': 'OpenCV extraction methods removed. Use FFmpeg or SDK methods.',
                    'frame_count': 0,
                    'output_files': [],
                    'lens_outputs': {}
                }
            
            # Fallback to stitched FFmpeg
            logger.warning(f"Unknown method '{method}', falling back to FFmpeg stitched")
            if input_path.suffix.lower() == '.insv':
                return {
                    'success': False,
                    'error': (
                        'Unknown stitched extraction request for .insv input. '
                        'FFmpeg stitched extraction is disabled for .insv because it produces incorrect results.'
                    ),
                    'frame_count': 0,
                    'output_files': [],
                    'lens_outputs': {}
                }
            return self._extract_with_ffmpeg_stitched(input_path, output_path, fps, start_time, end_time, progress_callback)
        
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'frame_count': 0,
                'output_files': [],
                'lens_outputs': {}
            }
    
    def _extract_with_ffmpeg_stitched(self, input_path: Path, output_path: Path, 
                            fps: float, start_time: float, end_time: Optional[float],
                            progress_callback: Optional[Callable]) -> Dict:
        """
        Extract frames using FFmpeg from ALREADY STITCHED equirectangular video.
        
        IMPORTANT: This method is for PRE-STITCHED MP4 files (already equirectangular).
        For .insv dual-stream files, use the dual-lens methods instead!
        """
        
        logger.info(f"Using FFmpeg stitched extraction method at {fps} FPS (for pre-stitched video)")
        
        # Output pattern
        output_pattern = str(output_path / "frame_%05d.png")
        
        # Build FFmpeg command with time range options
        cmd = [self.ffmpeg_path]
        
        # Add start time BEFORE input (fast seek)
        if start_time > 0:
            cmd.extend(['-ss', str(start_time)])
        
        cmd.extend(['-i', str(input_path)])
        
        # Add end time (duration) AFTER input
        if end_time is not None:
            duration = end_time - start_time
            if duration > 0:
                cmd.extend(['-t', str(duration)])
        
        # Simple extraction for already-stitched video
        # Just extract frames at specified FPS without any filtering
        cmd.extend([
            '-vf', f'fps={fps}',  # Frame rate control
            '-q:v', '2',  # High quality PNG
            '-y',  # Overwrite
            output_pattern
        ])
        
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor progress (simplified)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {stderr}")
            return {
                'success': False,
                'error': f"FFmpeg extraction failed: {stderr[:200]}",
                'frame_count': 0,
                'output_files': []
            }
        
        # Count extracted frames
        output_files = sorted(output_path.glob("frame_*.png"))
        frame_count = len(output_files)
        
        logger.info(f"Successfully extracted {frame_count} frames")
        
        return {
            'success': True,
            'frame_count': frame_count,
            'output_files': [str(f) for f in output_files],
            'error': None
        }
    
    def get_video_info(self, input_file: str) -> Dict:
        """
        Get comprehensive video metadata for display in UI.
        
        Args:
            input_file: Path to video file
            
        Returns:
            Dictionary with video info including:
            - file_type: File extension and format description
            - duration: Duration in seconds
            - duration_formatted: Formatted as "5m 34s"
            - fps: Frames per second
            - frame_count: Total number of frames
            - width: Video width in pixels
            - height: Video height in pixels
            - resolution: Formatted resolution string
            - camera_model: Detected camera model (if available)
        """
        try:
            input_path = Path(input_file)
            
            # Detect file type
            file_ext = input_path.suffix.lower()
            file_types = {
                '.insv': 'INSV dual-fisheye',
                '.mp4': 'MP4 video',
                '.mov': 'QuickTime video',
                '.avi': 'AVI video',
                '.mkv': 'Matroska video'
            }
            file_type_desc = file_types.get(file_ext, f'{file_ext.upper()[1:]} video')
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(input_file)
            
            if not cap.isOpened():
                return {'success': False, 'error': 'Failed to open video'}
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Format resolution
            # For .insv files, detect if vertically stacked or side-by-side
            if file_ext == '.insv':
                if height >= width:
                    # Vertically stacked: 3840×3840 = two 3840×1920 lenses
                    per_lens_w = width
                    per_lens_h = height // 2
                    resolution_str = f"{width}×{height} dual-fisheye ({per_lens_w}×{per_lens_h} per lens, vertically stacked)"
                else:
                    # Side-by-side: 3840×1920 = two 1920×1920 lenses
                    per_lens_w = width // 2
                    per_lens_h = height
                    resolution_str = f"{width}×{height} dual-fisheye ({per_lens_w}×{per_lens_h} per lens, side-by-side)"
            else:
                resolution_str = f"{width}×{height}"
            
            camera_model, camera_model_source = self._detect_camera_model(input_path, file_ext, width, height)
            video_streams = self.get_video_streams(input_file)
            primary_stream_index = self.get_primary_video_stream_index(input_file)
            
            # Format duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_formatted = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            
            # Get codec info (FOURCC)
            codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            logger.info(f"Video info: {resolution_str}, {fps:.2f} FPS, {duration_formatted}, "
                       f"{frame_count} frames")
            if camera_model:
                logger.info(
                    "Detected input device for %s: %s (%s)",
                    input_path.name,
                    camera_model,
                    camera_model_source,
                )
            elif file_ext == '.insv':
                logger.warning(
                    "Could not determine INSV source device for %s from metadata or trailer",
                    input_path.name,
                )
            
            return {
                'success': True,
                'file_type': file_ext.upper()[1:],
                'file_type_desc': file_type_desc,
                'fps': round(fps, 2),
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': resolution_str,
                'duration': duration,
                'duration_seconds': int(duration),
                'duration_formatted': duration_formatted,
                'camera_model': camera_model,
                'camera_model_source': camera_model_source,
                'video_stream_count': len(video_streams),
                'primary_video_stream_index': primary_stream_index,
                'codec': codec if codec.isprintable() else 'Unknown',
                'file_size_mb': round(input_path.stat().st_size / (1024 * 1024), 2)
            }
        
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {'success': False, 'error': str(e)}

    def _detect_camera_model(self, input_path: Path, file_ext: str, width: int, height: int) -> tuple[str, str]:
        """Detect camera model from ffprobe metadata first, then INSV trailer markers."""
        metadata = self._read_media_metadata(input_path)
        if metadata:
            metadata_strings = [entry.lower() for entry in self._flatten_metadata_entries(metadata)]
            known_models = [
                (("insta360", "x5"), "Insta360 X5"),
                (("insta360", "x4"), "Insta360 X4"),
                (("insta360", "x3"), "Insta360 X3"),
                (("insta360", "one x2"), "Insta360 ONE X2"),
                (("insta360", "one x"), "Insta360 ONE X"),
            ]
            for entry in metadata_strings:
                for tokens, label in known_models:
                    if all(token in entry for token in tokens):
                        return label, 'metadata'

        if file_ext == '.insv':
            trailer_model, trailer_source = self._detect_insv_trailer_camera_model(input_path)
            if trailer_model:
                return trailer_model, trailer_source

        return '', 'unavailable'

    def _detect_insv_trailer_camera_model(self, input_path: Path) -> tuple[str, str]:
        """Detect camera model from INSV trailer strings and known A1 camtype tokens."""
        trailer = self._read_file_tail(input_path, INSV_TRAILER_SCAN_BYTES)
        if not trailer:
            return '', 'unavailable'

        trailer_lower = trailer.lower()

        for needle, label in INSV_TRAILER_CAMERA_STRINGS:
            if needle in trailer_lower:
                return label, 'insv_trailer_string'

        if any(token in trailer for token in INSV_A1_CAMTYPE_TOKENS):
            return 'Antigravity A1', 'insv_trailer_camtype'

        return '', 'unavailable'

    def _read_file_tail(self, input_path: Path, max_bytes: int) -> bytes:
        """Read up to ``max_bytes`` from the end of a file for trailer inspection."""
        try:
            file_size = input_path.stat().st_size
            tail_size = min(file_size, max_bytes)
            with input_path.open('rb') as handle:
                if tail_size:
                    handle.seek(-tail_size, 2)
                return handle.read(tail_size)
        except OSError as exc:
            logger.debug(f"Failed to read trailer from {input_path.name}: {exc}")
            return b''

    def _read_media_metadata(self, input_path: Path) -> Dict:
        """Read media container metadata via ffprobe when available."""
        if not self.ffprobe_path:
            return {}

        cmd = [
            self.ffprobe_path,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(input_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=15,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return {}
            return json.loads(result.stdout)
        except Exception as exc:
            logger.debug(f"ffprobe metadata read failed for {input_path.name}: {exc}")
            return {}

    def get_video_streams(self, input_file: str) -> List[Dict]:
        """Return ffprobe video streams for the input file, preserving stream order."""
        metadata = self._read_media_metadata(Path(input_file))
        streams = metadata.get('streams', []) if isinstance(metadata, dict) else []
        return [stream for stream in streams if isinstance(stream, dict) and stream.get('codec_type') == 'video']

    def get_video_stream_count(self, input_file: str) -> int:
        """Return the number of video streams discovered by ffprobe."""
        return len(self.get_video_streams(input_file))

    def get_primary_video_stream_index(self, input_file: str) -> int:
        """Choose the primary video stream for preview/extraction from a multi-stream container.

        Preference order:
        1. stream with disposition.default = 1
        2. stream with the largest pixel area
        3. first video stream in container order
        """
        streams = self.get_video_streams(input_file)
        if not streams:
            return 0

        default_stream = next(
            (stream for stream in streams if isinstance(stream.get('disposition'), dict) and int(stream['disposition'].get('default', 0)) == 1),
            None,
        )
        if default_stream is not None:
            return int(default_stream.get('index', 0))

        def _stream_rank(stream: Dict) -> tuple[int, int, int]:
            width = int(stream.get('width') or 0)
            height = int(stream.get('height') or 0)
            return (width * height, width, height)

        best = max(streams, key=_stream_rank)
        return int(best.get('index', 0))

    def _flatten_metadata_entries(self, value, prefix: str = '') -> List[str]:
        """Flatten nested ffprobe metadata into searchable key=value strings."""
        entries: List[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                entries.extend(self._flatten_metadata_entries(child, next_prefix))
        elif isinstance(value, list):
            for child in value:
                entries.extend(self._flatten_metadata_entries(child, prefix))
        elif value not in (None, ''):
            rendered = str(value)
            entries.append(f"{prefix}={rendered}" if prefix else rendered)
        return entries
    
    def _extract_with_ffmpeg_v360_dual(self, input_path: Path, output_path: Path,
                                        fps: float, start_time: float, end_time: Optional[float],
                                        progress_callback: Optional[Callable]) -> Dict:
        """
        Stitch dual-fisheye .insv streams into equirectangular frames using FFmpeg v360.

        Designed for cameras whose dual-fisheye lenses are oriented zenith/nadir
        (e.g. Antigravity A1) and are NOT supported by the Insta360 MediaSDK.

        Confirmed working parameters for Antigravity A1 (systematic sweep 2026-03-28):
          - Streams: 0:v:0 (up lens) + 0:v:1 (down lens), both 3840×3840
          - Stream order: [0,1]  (no swap needed)
          - Filter: dfisheye → equirect, ih_fov=185, iv_fov=185, pitch=-90
          - pitch=-90 corrects for the A1's up/down lens orientation
          - Output: 7680×3840 JPEG (2:1 equirectangular)

        Sweep method: ord01_pre-none_fov185_p-90_y0 produced the best equirectangular
        with flat horizon, sky on top, ground on bottom, no visible stitch artefacts.
        """
        logger.info(f"Using FFmpeg v360 dual-fisheye stitch at {fps} FPS")

        output_pattern = str(output_path / "frame_%05d.jpg")

        # Build filter_complex: hstack both streams, then v360 dfisheye→equirect,
        # then fps filter to sample at the requested rate.
        # pitch=-90 is CRITICAL for the A1 up/down lens arrangement.
        filter_complex = (
            "[0:v:0][0:v:1]hstack=inputs=2[st];"
            "[st]v360=dfisheye:equirect:ih_fov=185:iv_fov=185:pitch=-90:yaw=0:roll=0,"
            f"fps={fps}"
        )

        cmd = [self.ffmpeg_path]

        # Fast seek before input
        if start_time > 0:
            cmd.extend(['-ss', str(start_time)])

        cmd.extend(['-i', str(input_path)])

        # Duration after input
        if end_time is not None:
            duration = end_time - start_time
            if duration > 0:
                cmd.extend(['-t', str(duration)])

        cmd.extend([
            '-filter_complex', filter_complex,
            '-q:v', '2',   # High quality JPEG
            '-y',
            output_pattern
        ])

        logger.debug(f"FFmpeg v360 command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            # Check if frames were actually produced despite non-zero exit
            # (FFmpeg returns rc=1 when unmapped streams exist, e.g. subtitle track)
            produced = sorted(output_path.glob("frame_*.jpg"))
            if not produced:
                logger.error(f"FFmpeg v360 dual stitch failed: {stderr}")
                return {
                    'success': False,
                    'error': f"FFmpeg v360 dual stitch failed: {stderr[:300]}",
                    'frame_count': 0,
                    'output_files': []
                }
            logger.warning(f"FFmpeg returned rc={process.returncode} but {len(produced)} frames produced — treating as success")

        output_files = sorted(output_path.glob("frame_*.jpg"))
        frame_count = len(output_files)
        logger.info(f"[OK] v360 dual stitch: {frame_count} equirectangular frames")

        return {
            'success': True,
            'frame_count': frame_count,
            'output_files': [str(f) for f in output_files],
            'error': None
        }

    def _extract_dual_lens_ffmpeg(self, input_path: Path, output_path: Path,
                                   fps: float, start_time: float, end_time: Optional[float],
                                   lens_mode: str, progress_callback: Optional[Callable]) -> Dict:
        """
        Extract frames from dual-lens video using FFmpeg (lossless stream extraction).
        
        CRITICAL: This method is ONLY for raw .insv files with multiple video streams!
        Do NOT use on pre-stitched equirectangular .mp4 files.
        
        Based on WORKING_WITH_DUAL_STREAM_360_CAMERAS.md best practices:
        - Uses -map 0:v:0 for lens 1 (front fisheye)
        - Uses -map 0:v:1 for lens 2 (back fisheye)
        - Creates organized folder structure: lens_1/ and lens_2/
        - Lossless extraction with -c copy for streams
        
        Args:
            lens_mode: 'both', 'lens1', or 'lens2'
        """
        logger.info(f"Extracting dual-lens frames with FFmpeg (mode: {lens_mode})")
        
        # CRITICAL VALIDATION: Check if this is actually a dual-stream file
        file_ext = input_path.suffix.lower()
        if file_ext == '.mp4':
            error_msg = ('ERROR: Dual-lens extraction attempted on .mp4 file.\n\n'
                        '.mp4 files typically contain already-stitched equirectangular video with a SINGLE video stream.\n'
                        'Dual-lens methods require raw .insv files with MULTIPLE video streams (0:v:0 and 0:v:1).\n\n'
                        'SOLUTION: Use "FFmpeg Stitched" or "SDK Stitching" method for .mp4 files.')
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'frame_count': 0,
                'output_files': [],
                'lens_outputs': {}
            }

        
        lens_outputs = {}
        total_frames = 0
        all_output_files = []
        
        # Determine which lenses to extract
        lenses_to_extract = []
        if lens_mode == 'both':
            lenses_to_extract = [('lens_1', 0), ('lens_2', 1)]
        elif lens_mode == 'lens1':
            lenses_to_extract = [('lens_1', 0)]
        elif lens_mode == 'lens2':
            lenses_to_extract = [('lens_2', 1)]
        
        for lens_name, stream_index in lenses_to_extract:
            # Use flat naming with lens suffix directly in output_path (e.g. frame_00001_lens1.png)
            # This avoids subdirectories that confuse downstream tools (SphereSfM, COLMAP)
            lens_suffix = lens_name.replace('_', '')  # 'lens_1' -> 'lens1', 'lens_2' -> 'lens2'
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting {lens_name} (stream {stream_index}) -> flat naming with _{lens_suffix} suffix...")
            
            # Output pattern: frame_00001_lens1.png / frame_00001_lens2.png
            output_pattern = str(output_path / f"frame_%05d_{lens_suffix}.png")
            
            # Build FFmpeg command for this lens
            cmd = [self.ffmpeg_path]
            
            # Add start time BEFORE input (fast seek)
            if start_time > 0:
                cmd.extend(['-ss', str(start_time)])
            
            cmd.extend(['-i', str(input_path)])
            
            # Add end time (duration) AFTER input
            if end_time is not None:
                duration = end_time - start_time
                if duration > 0:
                    cmd.extend(['-t', str(duration)])
            
            # Map specific video stream (lens)
            cmd.extend([
                '-map', f'0:v:{stream_index}',  # Select specific lens stream
                '-vf', f'fps={fps}',             # Frame rate control
                '-q:v', '2',                     # High quality PNG
                '-y',                            # Overwrite
                output_pattern
            ])
            
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg failed for {lens_name}: {stderr}")
                continue
            
            # Count extracted frames for this lens (flat naming in output_path)
            output_files = sorted(output_path.glob(f"frame_*_{lens_suffix}.png"))
            frame_count = len(output_files)
            
            logger.info(f"Successfully extracted {frame_count} frames from {lens_name} (flat, _{lens_suffix} suffix)")
            
            lens_outputs[lens_name] = {
                'frame_count': frame_count,
                'output_files': [str(f) for f in output_files],
                'output_dir': str(output_path)
            }
            
            total_frames += frame_count
            all_output_files.extend([str(f) for f in output_files])
        
        if not lens_outputs:
            return {
                'success': False,
                'error': 'Failed to extract any lens frames',
                'frame_count': 0,
                'output_files': [],
                'lens_outputs': {}
            }
        
        return {
            'success': True,
            'frame_count': total_frames,
            'output_files': all_output_files,
            'lens_outputs': lens_outputs,
            'error': None
        }
