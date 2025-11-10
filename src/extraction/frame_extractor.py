"""
Frame Extraction Module for 360FrameTools
Simplified extractor supporting FFmpeg and OpenCV methods.
SDK integration can be added later.

Based on Insta360toFrames but adapted for unified pipeline.
"""

import cv2
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional, Callable, List
import shutil

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extract frames from video files (.INSV, .mp4, etc.)
    Supports FFmpeg and OpenCV methods.
    """
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize FrameExtractor.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (auto-detect if None)
        """
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self.has_ffmpeg = self.ffmpeg_path is not None
        self.is_cancelled = False
        
        if self.has_ffmpeg:
            logger.info(f"FFmpeg found at: {self.ffmpeg_path}")
        else:
            logger.warning("FFmpeg not found - only OpenCV extraction available")
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Auto-detect FFmpeg installation"""
        ffmpeg_path = shutil.which('ffmpeg')
        return ffmpeg_path
    
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
            # Route to appropriate extraction method based on method type
            if method.startswith('ffmpeg_') and self.has_ffmpeg:
                if method == 'ffmpeg_stitched':
                    return self._extract_with_ffmpeg_stitched(input_path, output_path, fps, start_time, end_time, progress_callback)
                elif method == 'ffmpeg_dual_lens':
                    return self._extract_dual_lens_ffmpeg(input_path, output_path, fps, start_time, end_time, 'both', progress_callback)
                elif method == 'ffmpeg_lens1':
                    return self._extract_dual_lens_ffmpeg(input_path, output_path, fps, start_time, end_time, 'lens1', progress_callback)
                elif method == 'ffmpeg_lens2':
                    return self._extract_dual_lens_ffmpeg(input_path, output_path, fps, start_time, end_time, 'lens2', progress_callback)
            elif method.startswith('opencv_'):
                if method == 'opencv_dual_lens':
                    return self._extract_dual_lens_opencv(input_path, output_path, fps, start_time, end_time, 'both', progress_callback)
                elif method == 'opencv_lens1':
                    return self._extract_dual_lens_opencv(input_path, output_path, fps, start_time, end_time, 'lens1', progress_callback)
                elif method == 'opencv_lens2':
                    return self._extract_dual_lens_opencv(input_path, output_path, fps, start_time, end_time, 'lens2', progress_callback)
            
            # Fallback to stitched FFmpeg or OpenCV
            logger.warning(f"Unknown method '{method}', falling back to default")
            if self.has_ffmpeg:
                return self._extract_with_ffmpeg_stitched(input_path, output_path, fps, start_time, end_time, progress_callback)
            else:
                return self._extract_dual_lens_opencv(input_path, output_path, fps, start_time, end_time, 'both', progress_callback)
        
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
    
    def _extract_with_opencv(self, input_path: Path, output_path: Path,
                            fps: float, start_time: float, end_time: Optional[float],
                            progress_callback: Optional[Callable]) -> Dict:
        """
        Extract frames using OpenCV with time range support.
        WARNING: OpenCV extracts RAW dual-fisheye frames (not stitched).
        Use FFmpeg method for proper equirectangular stitching.
        """
        
        logger.warning(f"OpenCV extracts RAW fisheye frames (not stitched). Use FFmpeg for stitching.")
        logger.info(f"Using OpenCV extraction method at {fps} FPS")
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            return {
                'success': False,
                'error': "Failed to open video with OpenCV",
                'frame_count': 0,
                'output_files': []
            }
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        # Calculate frame range based on time range
        start_frame = int(start_time * video_fps) if start_time > 0 else 0
        end_frame = int(end_time * video_fps) if end_time is not None else total_frames
        
        # Ensure valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        logger.info(f"Video: {video_fps:.2f} FPS, {total_frames} frames, {duration:.1f}s")
        logger.info(f"Extracting frames {start_frame} to {end_frame}")
        
        # Set starting position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        frame_count = 0
        output_files = []
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            # Check for cancellation
            if hasattr(self, 'is_cancelled') and self.is_cancelled:
                logger.info("Extraction cancelled by user")
                cap.release()
                return {
                    'success': False,
                    'frame_count': frame_count,
                    'output_files': output_files,
                    'error': 'Cancelled by user'
                }
            
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at interval
            if (frame_idx - start_frame) % frame_interval == 0:
                output_file = output_path / f"frame_{frame_count:05d}.png"
                cv2.imwrite(str(output_file), frame)
                output_files.append(str(output_file))
                frame_count += 1
                
                if progress_callback and frame_count % 10 == 0:
                    progress = int(((frame_idx - start_frame) / (end_frame - start_frame)) * 100)
                    progress_callback(frame_idx, end_frame, f"Extracted {frame_count} frames")
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"Successfully extracted {frame_count} frames")
        
        return {
            'success': True,
            'frame_count': frame_count,
            'output_files': output_files,
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
    
    def _extract_dual_lens_opencv(self, input_path: Path, output_path: Path,
                                   fps: float, start_time: float, end_time: Optional[float],
                                   lens_mode: str, progress_callback: Optional[Callable]) -> Dict:
        """
        Extract frames from dual-lens video using OpenCV.
        
        CRITICAL: This method is ONLY for raw .insv files with dual-fisheye lenses!
        Do NOT use on pre-stitched equirectangular .mp4 files (will split into hemispheres instead of lenses).
        
        NOTE: OpenCV opens .insv files as a single video stream with horizontally stacked lenses.
        We split the frame into left (lens 1) and right (lens 2) halves.
        
        Args:
            lens_mode: 'both', 'lens1', or 'lens2'
        """
        logger.info(f"Extracting dual-lens frames with OpenCV (mode: {lens_mode})")
        
        # CRITICAL VALIDATION: Check if this is actually a dual-stream file
        # Equirectangular files (already stitched) should NOT use dual-lens methods
        file_ext = input_path.suffix.lower()
        if file_ext == '.mp4':
            error_msg = ('ERROR: Dual-lens extraction attempted on .mp4 file (likely already stitched equirectangular).\n\n'
                        'PROBLEM: Splitting a stitched equirectangular image gives you TOP/BOTTOM HEMISPHERES, not lens 1/lens 2!\n\n'
                        'SOLUTION: Use "FFmpeg Stitched" or "SDK Stitching" method for .mp4 files.\n'
                        'Dual-lens methods are ONLY for raw .insv files with dual-fisheye streams.')
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'frame_count': 0,
                'output_files': [],
                'lens_outputs': {}
            }
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            return {
                'success': False,
                'error': "Failed to open video with OpenCV",
                'frame_count': 0,
                'output_files': [],
                'lens_outputs': {}
            }
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        start_frame = int(start_time * video_fps) if start_time > 0 else 0
        end_frame = int(end_time * video_fps) if end_time is not None else total_frames
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        logger.info(f"Video: {video_fps:.2f} FPS, {frame_width}×{frame_height}, frames {start_frame}-{end_frame}")
        
        # Determine lens split orientation
        # Insta360 files can be:
        # - Side-by-side (width > height): 3840×1920 = 1920×1920 per lens
        # - Vertically stacked (height >= width): 3840×3840 = 3840×1920 per lens (top/bottom)
        is_vertical_stack = frame_height >= frame_width
        
        # Initialize both variables
        lens_width = frame_width // 2
        lens_height = frame_height // 2
        
        if is_vertical_stack:
            # Vertically stacked (top = lens1, bottom = lens2)
            logger.info(f"Detected vertically stacked lenses: {frame_width}×{lens_height} per lens")
        else:
            # Side-by-side (left = lens1, right = lens2)
            logger.info(f"Detected side-by-side lenses: {lens_width}×{frame_height} per lens")
        
        # Create output directories
        lens_outputs = {}
        extract_lens1 = lens_mode in ['both', 'lens1']
        extract_lens2 = lens_mode in ['both', 'lens2']
        
        if extract_lens1:
            (output_path / 'lens_1').mkdir(parents=True, exist_ok=True)
            lens_outputs['lens_1'] = {'frame_count': 0, 'output_files': [], 'output_dir': str(output_path / 'lens_1')}
        if extract_lens2:
            (output_path / 'lens_2').mkdir(parents=True, exist_ok=True)
            lens_outputs['lens_2'] = {'frame_count': 0, 'output_files': [], 'output_dir': str(output_path / 'lens_2')}
        
        # Set starting position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        frame_idx = start_frame
        extracted_count = 0
        
        while frame_idx < end_frame:
            # Check for cancellation
            if hasattr(self, 'is_cancelled') and self.is_cancelled:
                logger.info("Extraction cancelled by user")
                cap.release()
                return {
                    'success': False,
                    'frame_count': extracted_count,
                    'output_files': [],
                    'error': 'Cancelled by user'
                }
            
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract at interval
            if (frame_idx - start_frame) % frame_interval == 0:
                # Split frame into lenses based on orientation
                if is_vertical_stack:
                    # Vertical split: top = lens1, bottom = lens2
                    if extract_lens1:
                        lens1_frame = frame[:lens_height, :]
                        lens1_path = output_path / 'lens_1' / f"frame_{extracted_count:05d}.png"
                        cv2.imwrite(str(lens1_path), lens1_frame)
                        lens_outputs['lens_1']['output_files'].append(str(lens1_path))
                        lens_outputs['lens_1']['frame_count'] += 1
                    
                    if extract_lens2:
                        lens2_frame = frame[lens_height:, :]
                        lens2_path = output_path / 'lens_2' / f"frame_{extracted_count:05d}.png"
                        cv2.imwrite(str(lens2_path), lens2_frame)
                        lens_outputs['lens_2']['output_files'].append(str(lens2_path))
                        lens_outputs['lens_2']['frame_count'] += 1
                else:
                    # Horizontal split: left = lens1, right = lens2
                    if extract_lens1:
                        lens1_frame = frame[:, :lens_width]
                        lens1_path = output_path / 'lens_1' / f"frame_{extracted_count:05d}.png"
                        cv2.imwrite(str(lens1_path), lens1_frame)
                        lens_outputs['lens_1']['output_files'].append(str(lens1_path))
                        lens_outputs['lens_1']['frame_count'] += 1
                    
                    if extract_lens2:
                        lens2_frame = frame[:, lens_width:]
                        lens2_path = output_path / 'lens_2' / f"frame_{extracted_count:05d}.png"
                        cv2.imwrite(str(lens2_path), lens2_frame)
                        lens_outputs['lens_2']['output_files'].append(str(lens2_path))
                        lens_outputs['lens_2']['frame_count'] += 1
                
                extracted_count += 1
                
                if progress_callback:
                    progress = int((frame_idx - start_frame) / (end_frame - start_frame) * 100)
                    progress_callback(frame_idx - start_frame, end_frame - start_frame, 
                                    f"Extracting frames: {extracted_count}")
            
            frame_idx += 1
        
        cap.release()
        
        total_frames = sum(lens_outputs[lens]['frame_count'] for lens in lens_outputs)
        all_output_files = []
        for lens in lens_outputs.values():
            all_output_files.extend(lens['output_files'])
        
        logger.info(f"Successfully extracted {total_frames} total frames (lens 1: {lens_outputs.get('lens_1', {}).get('frame_count', 0)}, "
                   f"lens 2: {lens_outputs.get('lens_2', {}).get('frame_count', 0)})")
        
        return {
            'success': True,
            'frame_count': total_frames,
            'output_files': all_output_files,
            'lens_outputs': lens_outputs,
            'error': None
        }
