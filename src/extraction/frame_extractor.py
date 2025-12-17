"""
Frame Extraction Module for 360FrameTools
Simplified extractor supporting FFmpeg method.
OpenCV fallback methods removed for size optimization - use FFmpeg or SDK.

Based on Extraction Module but adapted for unified pipeline.
"""

import cv2
import subprocess
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Callable, List
import shutil

logger = logging.getLogger(__name__)


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
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
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
            
            # Route to appropriate FFmpeg extraction method
            if method.startswith('ffmpeg_'):
                if method == 'ffmpeg_stitched':
                    return self._extract_with_ffmpeg_stitched(input_path, output_path, fps, start_time, end_time, progress_callback)
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
                '.insv': 'Insta360 dual-fisheye',
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
            
            # Detect camera model from filename or resolution
            camera_model = "Unknown"
            if file_ext == '.insv':
                max_dim = max(width, height)
                if max_dim >= 5760:
                    camera_model = "Insta360 X3/X4/X5"
                elif max_dim >= 3840:
                    camera_model = "Insta360 ONE X/X2"
                else:
                    camera_model = "Insta360 camera"
            
            # Format duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_formatted = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            
            # Get codec info (FOURCC)
            codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            logger.info(f"Video info: {resolution_str}, {fps:.2f} FPS, {duration_formatted}, "
                       f"{frame_count} frames")
            
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
                'codec': codec if codec.isprintable() else 'Unknown',
                'file_size_mb': round(input_path.stat().st_size / (1024 * 1024), 2)
            }
        
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {'success': False, 'error': str(e)}
    
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
            lens_dir = output_path / lens_name
            lens_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting {lens_name} (stream {stream_index})...")
            
            # Output pattern
            output_pattern = str(lens_dir / "frame_%05d.png")
            
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
            
            # Count extracted frames for this lens
            output_files = sorted(lens_dir.glob("frame_*.png"))
            frame_count = len(output_files)
            
            logger.info(f"Successfully extracted {frame_count} frames from {lens_name}")
            
            lens_outputs[lens_name] = {
                'frame_count': frame_count,
                'output_files': [str(f) for f in output_files],
                'output_dir': str(lens_dir)
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
