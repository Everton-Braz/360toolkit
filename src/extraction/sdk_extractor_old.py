r"""
Insta360 SDK Frame Extractor
Integrates with official Insta360 SDK for high-quality stitching.

SDK Path: C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1
"""

import subprocess
import logging
import cv2
from pathlib import Path
from typing import Dict, Optional, Callable, List

logger = logging.getLogger(__name__)


# SDK configuration
DEFAULT_SDK_PATH = r"C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1"

# Quality presets
QUALITY_PRESETS = {
    'draft': {
        'stitch_type': 'template',
        'enable_seamless': False,
        'enable_stabilization': False,
        'enable_colorplus': False,
        'enable_denoise': False
    },
    'good': {
        'stitch_type': 'optflow',
        'enable_seamless': True,
        'enable_stabilization': False,
        'enable_colorplus': False,
        'enable_denoise': False
    },
    'best': {
        'stitch_type': 'aistitch',
        'enable_seamless': True,
        'enable_stabilization': True,
        'enable_colorplus': True,
        'enable_denoise': True
    }
}

# Resolution presets
RESOLUTION_PRESETS = {
    'original': None,  # Use video's original resolution
    '8k': (7680, 3840),
    '6k': (6144, 3072),
    '4k': (3840, 1920),
    '2k': (1920, 960)
}


class SDKExtractor:
    """
    Extract and stitch frames using Insta360 SDK.
    Provides highest quality stitching without Insta360 Studio.
    """
    
    def __init__(self, sdk_path: str = DEFAULT_SDK_PATH):
        """
        Initialize SDK extractor.
        
        Args:
            sdk_path: Path to Insta360 SDK installation
        """
        self.sdk_path = Path(sdk_path)
        
        # Try multiple possible locations for the demo executable
        possible_locations = [
            self.sdk_path / "Demo" / "Windows" / "Media SDK" / "MediaSDK-Demo.exe",
            self.sdk_path / "Demo" / "Windows" / "Media SDK" / "Release" / "MediaSDK-Demo.exe",
            self.sdk_path / "bin" / "MediaSDK-Demo.exe",
            self.sdk_path / "MediaSDK-Demo.exe"
        ]
        
        self.demo_exe = None
        for location in possible_locations:
            if location.exists():
                self.demo_exe = location
                logger.info(f"Found SDK demo at: {location}")
                break
        
        # Model files
        self.ai_model_v1 = self.sdk_path / "data" / "ai_stitch_model_v1.ins"
        self.ai_model_v2 = self.sdk_path / "data" / "ai_stitch_model_v2.ins"
        self.colorplus_model = self.sdk_path / "data" / "colorplus_model.ins"
        
        # Check SDK availability
        self.available = self._check_sdk_availability()
        
        if self.available:
            logger.info(f"Insta360 SDK initialized successfully")
            logger.info(f"  SDK Path: {sdk_path}")
            logger.info(f"  Demo EXE: {self.demo_exe}")
        else:
            logger.warning(f"Insta360 SDK not found at: {sdk_path}")
            logger.warning(f"  Searched locations:")
            for loc in possible_locations:
                logger.warning(f"    - {loc}")
    
    def _check_sdk_availability(self) -> bool:
        """Check if SDK demo executable exists"""
        return self.demo_exe is not None and self.demo_exe.exists()
    
    def extract_frames(self, input_file: str, output_dir: str,
                      fps: float = 1.0,
                      quality: str = 'good',
                      resolution: str = 'original',
                      output_format: str = 'png',
                      start_time: float = 0.0,
                      end_time: Optional[float] = None,
                      progress_callback: Optional[Callable] = None) -> Dict:
        """
        Extract stitched frames from .INSV video using SDK.
        
        Args:
            input_file: Path to .INSV file
            output_dir: Directory to save frames
            fps: Frames per second to extract
            quality: 'draft', 'good', or 'best'
            resolution: 'original', '8k', '6k', '4k', or '2k'
            output_format: 'png' or 'jpeg'
            start_time: Start time in seconds
            end_time: End time in seconds (None = full video)
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Dictionary with extraction results
        """
        
        if not self.available:
            return {
                'success': False,
                'error': 'SDK not available',
                'frame_count': 0,
                'output_files': []
            }
        
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting frames with Insta360 SDK: {input_path.name}")
        logger.info(f"Quality: {quality}, Resolution: {resolution}, FPS: {fps}")
        
        try:
            # Step 1: Stitch video with SDK (creates temporary stitched video)
            temp_video = output_path / "temp_stitched.mp4"
            
            stitch_success = self._stitch_video(
                input_path,
                temp_video,
                quality,
                resolution
            )
            
            if not stitch_success:
                return {
                    'success': False,
                    'error': 'SDK stitching failed',
                    'frame_count': 0,
                    'output_files': []
                }
            
            # Step 2: Extract frames from stitched video
            frame_results = self._extract_frames_from_video(
                temp_video,
                output_path,
                fps,
                start_time,
                end_time,
                output_format,
                progress_callback
            )
            
            # Step 3: Clean up temporary video
            if temp_video.exists():
                temp_video.unlink()
                logger.debug("Cleaned up temporary stitched video")
            
            return frame_results
            
        except Exception as e:
            logger.error(f"SDK extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'frame_count': 0,
                'output_files': []
            }
    
    def _stitch_video(self, input_file: Path, output_file: Path,
                     quality: str, resolution: str) -> bool:
        """
        Stitch .INSV video using SDK.
        
        Args:
            input_file: Input .INSV file
            output_file: Output stitched video
            quality: Quality preset
            resolution: Resolution preset
            
        Returns:
            True if successful
        """
        
        # Get quality settings
        quality_settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['good'])
        
        # Get resolution
        resolution_size = RESOLUTION_PRESETS.get(resolution)
        if resolution_size is None:
            # Use original resolution - need to detect from video
            resolution_size = (7680, 3840)  # Default to 8K
        
        # Build SDK command
        cmd = [str(self.demo_exe)]
        
        # Input file
        cmd.extend(["-inputs", str(input_file)])
        
        # Output file
        cmd.extend(["-output", str(output_file)])
        
        # Resolution
        cmd.extend(["-output_size", f"{resolution_size[0]}x{resolution_size[1]}"])
        
        # Stitch type
        cmd.extend(["-stitch_type", quality_settings['stitch_type']])
        
        # AI model (if using AI stitching)
        if quality_settings['stitch_type'] == 'aistitch':
            cmd.extend(["-ai_stitching_model", str(self.ai_model_v1)])
        
        # Seamless blending
        if quality_settings['enable_seamless']:
            cmd.append("-enable_stitchfusion")
        
        # Stabilization
        if quality_settings['enable_stabilization']:
            cmd.append("-enable_flowstate")
        
        # Color Plus
        if quality_settings['enable_colorplus']:
            cmd.append("-enable_colorplus")
            cmd.extend(["-colorplus_model", str(self.colorplus_model)])
        
        # Denoise
        if quality_settings['enable_denoise']:
            cmd.append("-enable_denoise")
        
        # H.265 for 4K+
        if resolution_size[0] >= 3840:
            cmd.extend(["-enable_h265_encoder", "h265"])
        
        logger.debug(f"SDK command: {' '.join(cmd)}")
        logger.info("Running Insta360 SDK stitching (this may take a few minutes)...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600  # 1 hour timeout
            )
            
            logger.info("SDK stitching completed successfully")
            logger.debug(f"SDK output: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"SDK stitching failed: {e}")
            logger.error(f"SDK error output: {e.stderr}")
            return False
        
        except subprocess.TimeoutExpired:
            logger.error("SDK stitching timed out (>1 hour)")
            return False
    
    def _extract_frames_from_video(self, video_file: Path, output_dir: Path,
                                   fps: float, start_time: float, end_time: Optional[float],
                                   output_format: str,
                                   progress_callback: Optional[Callable]) -> Dict:
        """
        Extract frames from stitched video.
        
        Args:
            video_file: Stitched video file
            output_dir: Output directory
            fps: Frames per second
            start_time: Start time in seconds
            end_time: End time in seconds
            output_format: 'png' or 'jpeg'
            progress_callback: Progress callback
            
        Returns:
            Extraction results
        """
        
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            return {
                'success': False,
                'error': 'Failed to open stitched video',
                'frame_count': 0,
                'output_files': []
            }
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time * video_fps) if start_time > 0 else 0
        end_frame = int(end_time * video_fps) if end_time is not None else total_frames
        
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        # Set starting position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        # Extract frames
        frame_count = 0
        output_files = []
        frame_idx = start_frame
        
        # File extension
        ext = '.png' if output_format.lower() == 'png' else '.jpg'
        
        logger.info(f"Extracting {fps} FPS from stitched video...")
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at interval
            if (frame_idx - start_frame) % frame_interval == 0:
                output_file = output_dir / f"frame_{frame_count:05d}{ext}"
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
