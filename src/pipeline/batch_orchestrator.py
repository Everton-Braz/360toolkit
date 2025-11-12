"""
Batch Pipeline Orchestrator for 360FrameTools
Coordinates the 3-stage pipeline: Extract → Split → Mask

Uses QThread for non-blocking UI execution with progress signals.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Callable
import cv2
import json
from datetime import datetime
from PIL import Image
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from ..extraction import FrameExtractor
from ..extraction.sdk_extractor import SDKExtractor
from ..transforms import E2PTransform, E2CTransform
from .metadata_handler import MetadataHandler
from ..config.defaults import (
    DEFAULT_FPS, DEFAULT_H_FOV, DEFAULT_SPLIT_COUNT,
    DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT
)

logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """
    Worker thread for pipeline execution.
    Emits signals for progress updates and completion.
    """
    
    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    stage_complete = pyqtSignal(int, dict)  # stage_number, results
    finished = pyqtSignal(dict)  # final_results
    error = pyqtSignal(str)  # error_message
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.is_cancelled = False
        self.is_paused = False
        
        # Initialize components
        self.frame_extractor = FrameExtractor()
        self.sdk_extractor = SDKExtractor()  # Insta360 SDK extractor
        self.e2p_transform = E2PTransform()
        self.e2c_transform = E2CTransform()
        self.metadata_handler = MetadataHandler()
        self.masker = None  # Initialize only if masking enabled
    
    def discover_stage_input_folder(self, stage: int, output_dir: str) -> Optional[Path]:
        """
        Smart folder discovery for individual stage processing.
        
        Stage 1: Looks for input file (returns None - user must select)
        Stage 2: Looks for stage1_frames folder with equirectangular images
        Stage 3: Looks for stage2_perspectives folder with perspective images
        
        Returns Path to folder if found, None if not found (user must select manually)
        """
        output_path = Path(output_dir)
        
        if stage == 1:
            # Stage 1 requires input file - user must select
            return None
        
        elif stage == 2:
            # Look for stage1_frames folder (match common image extensions, case-insensitive)
            stage1_folder = output_path / 'stage1_frames'
            image_patterns = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
            if stage1_folder.exists() and any(stage1_folder.glob(p) for p in image_patterns):
                logger.info(f"[OK] Auto-discovered Stage 1 output: {stage1_folder}")
                return stage1_folder
            return None
        
        elif stage == 3:
            # Look for stage2_perspectives folder (match common image extensions)
            stage2_folder = output_path / 'stage2_perspectives'
            image_patterns = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
            if stage2_folder.exists() and any(stage2_folder.glob(p) for p in image_patterns):
                logger.info(f"[OK] Auto-discovered Stage 2 output: {stage2_folder}")
                return stage2_folder
            return None
        
        return None
    
    def set_stage_input_folder(self, stage: int, folder_path: Path):
        """
        Manually set input folder for individual stage processing.
        Updates config to process only that stage with the provided input.
        """
        folder_path = Path(folder_path)
        
        if stage == 2:
            # Stage 2 input is equirectangular images from Stage 1
            self.config['stage2_input_dir'] = str(folder_path)
            # Disable Stage 1, enable Stage 2
            self.config['enable_stage1'] = False
            self.config['enable_stage2'] = True
            self.config['enable_stage3'] = False
            logger.info(f"[OK] Set Stage 2 input: {folder_path}")
        
        elif stage == 3:
            # Stage 3 input is perspective images from Stage 2
            self.config['stage3_input_dir'] = str(folder_path)
            # Disable Stages 1 & 2, enable Stage 3
            self.config['enable_stage1'] = False
            self.config['enable_stage2'] = False
            self.config['enable_stage3'] = True
            logger.info(f"[OK] Set Stage 3 input: {folder_path}")
    
    def run(self):
        """Execute the pipeline"""
        try:
            results = {
                'stage1': {},
                'stage2': {},
                'stage3': {},
                'success': False,
                'start_time': datetime.now().isoformat()
            }
            
            # Track which stages were executed
            stages_executed = []
            
            # Stage 1: Extract Frames
            if self.config.get('enable_stage1', True):
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before Stage 1")
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return
                
                logger.info("=== Starting Stage 1: Frame Extraction ===")
                stage1_result = self._execute_stage1()
                results['stage1'] = stage1_result
                stages_executed.append(1)
                self.stage_complete.emit(1, stage1_result)
                
                if not stage1_result.get('success'):
                    self.error.emit(f"Stage 1 failed: {stage1_result.get('error')}")
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return
            
            # Stage 2: Split Perspectives
            if self.config.get('enable_stage2', True):
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before Stage 2")
                    results['stages_executed'] = stages_executed
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return
                
                logger.info("=== Starting Stage 2: Perspective Splitting ===")
                stage2_result = self._execute_stage2()
                results['stage2'] = stage2_result
                stages_executed.append(2)
                self.stage_complete.emit(2, stage2_result)
                
                if not stage2_result.get('success'):
                    self.error.emit(f"Stage 2 failed: {stage2_result.get('error')}")
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return
            
            # Stage 3: Generate Masks
            if self.config.get('enable_stage3', True):
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before Stage 3")
                    results['stages_executed'] = stages_executed
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return
                
                logger.info("=== Starting Stage 3: Mask Generation ===")
                stage3_result = self._execute_stage3()
                results['stage3'] = stage3_result
                stages_executed.append(3)
                self.stage_complete.emit(3, stage3_result)
                
                if not stage3_result.get('success'):
                    self.error.emit(f"Stage 3 failed: {stage3_result.get('error')}")
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return
            
            # Success - ALL stages completed
            results['success'] = True
            results['stages_executed'] = stages_executed
            results['end_time'] = datetime.now().isoformat()
            
            # Log completion ONLY after all stages finish
            logger.info(f"=== Pipeline Complete === (Executed stages: {stages_executed})")
            self.finished.emit(results)
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            self.error.emit(str(e))
            self.finished.emit({'success': False, 'error': str(e)})
    
    def _execute_stage1(self) -> Dict:
        """Execute Stage 1: Frame Extraction"""
        try:
            input_file = self.config['input_file']
            output_dir = Path(self.config['output_dir']) / 'stage1_frames'
            fps = self.config.get('fps', DEFAULT_FPS)
            method = self.config.get('extraction_method', 'opencv')
            
            # Get additional parameters
            start_time = self.config.get('start_time', 0.0)
            end_time = self.config.get('end_time', None)
            
            def progress_callback(current, total, message):
                self.progress.emit(current, total, f"Stage 1: {message}")
            
            # Map extraction method names
            # 'dual_fisheye' is raw extraction, so it should use opencv
            # 'ffmpeg' uses v360 stitching filter (RECOMMENDED)
            if method.lower() == 'dual_fisheye':
                logger.info("Dual-Fisheye Export selected - extracting raw fisheye frames")
                method = 'opencv'
            
            # Use SDK extractor if method is 'sdk' or 'sdk_stitching' (PRIMARY METHOD)
            if method.lower() in ['sdk', 'sdk_stitching']:
                logger.info("=== SDK Stitching (PRIMARY METHOD) ===")
                
                if not self.sdk_extractor.is_available():
                    logger.warning("WARNING: Insta360 MediaSDK not available")
                    logger.warning("INFO: Auto-fallback to FFmpeg v360 stitching (SDK-quality)")
                    method = 'ffmpeg'
                else:
                    logger.info("INFO: Insta360 MediaSDK detected - using SDK stitching")
                    
                    # SDK extraction parameters
                    quality = self.config.get('sdk_quality', 'best')  # best/good/draft
                    output_format = self.config.get('output_format', 'jpg')  # jpg/png
                    
                    # Resolution mapping
                    resolution_map = {
                        'original': None,  # SDK will use original
                        '8k': (7680, 3840),
                        '6k': (6080, 3040),
                        '4k': (3840, 1920),
                        '2k': (1920, 960)
                    }
                    resolution_key = str(self.config.get('sdk_resolution', 'original')).lower()
                    resolution = resolution_map.get(resolution_key, None)
                    
                    try:
                        # Call new MediaSDK extractor with all parameters
                        frame_paths = self.sdk_extractor.extract_frames(
                            input_path=str(input_file),
                            output_dir=str(output_dir),
                            fps=fps,
                            quality=quality,
                            resolution=resolution,
                            output_format=output_format,
                            start_time=start_time,
                            end_time=end_time,
                            progress_callback=lambda p: progress_callback(p, 100, "SDK stitching")
                        )
                        
                        logger.info(f"[OK] SDK extraction complete: {len(frame_paths)} frames")
                        return {
                            'success': True,
                            'frames': frame_paths,
                            'method': 'sdk_stitching',
                            'count': len(frame_paths)
                        }
                    
                    except subprocess.TimeoutExpired as timeout_error:
                        # SDK timeout - check if frames were actually extracted
                        logger.warning(f"[WARNING] SDK extraction timeout: {timeout_error}")
                        extracted_dir = Path(output_dir)
                        frame_files = list(extracted_dir.glob('*.*'))  # Find any extracted files
                        
                        if frame_files:
                            # SDK did extract frames before timing out - use them!
                            logger.info(f"[OK] SDK partially completed: {len(frame_files)} frames extracted before timeout")
                            frame_paths = [str(f) for f in sorted(frame_files)]
                            return {
                                'success': True,
                                'frames': frame_paths,
                                'method': 'sdk_stitching',
                                'count': len(frame_paths),
                                'warning': f'Timeout after {len(frame_files)} frames'
                            }
                        else:
                            # No frames produced - fallback to FFmpeg
                            logger.warning("INFO: No frames extracted by SDK timeout - Falling back to FFmpeg method")
                            method = 'ffmpeg'
                    
                    except Exception as sdk_error:
                        # Other SDK errors - check if any frames were created anyway
                        logger.error(f"[ERROR] SDK extraction failed: {sdk_error}")
                        extracted_dir = Path(output_dir)
                        frame_files = list(extracted_dir.glob('*.*'))
                        
                        if frame_files and len(frame_files) > 10:  # At least 10 frames produced
                            logger.warning(f"[WARNING] SDK error but {len(frame_files)} frames were extracted - using them")
                            frame_paths = [str(f) for f in sorted(frame_files)]
                            return {
                                'success': True,
                                'frames': frame_paths,
                                'method': 'sdk_stitching',
                                'count': len(frame_paths),
                                'warning': f'SDK error but {len(frame_files)} frames recovered'
                            }
                        
                        logger.warning("INFO: Falling back to FFmpeg method")
                        method = 'ffmpeg'
            
            # Use standard FrameExtractor (FFmpeg or OpenCV)
            result = self.frame_extractor.extract_frames(
                input_file=input_file,
                output_dir=str(output_dir),
                fps=fps,
                method=method,
                start_time=start_time,
                end_time=end_time,
                progress_callback=progress_callback
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Stage 1 error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_stage2(self) -> Dict:
        """Execute Stage 2: Perspective Splitting"""
        try:
            # Get input frames (auto-discovery runs ONLY ONCE)
            if self.config.get('enable_stage1', True):
                # Stage 1 was enabled - use its output directly
                input_dir = Path(self.config['output_dir']) / 'stage1_frames'
            else:
                # Stage 1 disabled - check for explicit input or auto-discover ONCE
                stage2_input = self.config.get('stage2_input_dir')
                if not stage2_input:
                    # Single auto-discovery attempt
                    discovered = self.discover_stage_input_folder(2, self.config['output_dir'])
                    if discovered:
                        input_dir = discovered
                    else:
                        return {
                            'success': False,
                            'error': 'Stage 2 input directory not specified and auto-discovery failed (Stage 1 is disabled)',
                            'output_files': []
                        }
                else:
                    input_dir = Path(stage2_input)
            
            if not input_dir.exists():
                return {
                    'success': False,
                    'error': f'Input directory does not exist: {input_dir}',
                    'output_files': []
                }
            
            output_dir = Path(self.config['output_dir']) / 'stage2_perspectives'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get transform settings
            transform_type = self.config.get('transform_type', 'perspective')
            output_width = self.config.get('output_width', DEFAULT_OUTPUT_WIDTH)
            output_height = self.config.get('output_height', DEFAULT_OUTPUT_HEIGHT)
            
            # Get all input frames (support many extensions, case-insensitive)
            image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            input_frames = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
            total_frames = len(input_frames)
            
            # Route based on transform type
            if transform_type == 'cubemap':
                logger.info(f"Processing {total_frames} frames in CUBEMAP mode")
                return self._execute_stage2_cubemap(input_frames, output_dir)
            else:
                # Perspective mode - use cameras
                camera_config = self.config.get('camera_config', {})
                cameras = camera_config.get('cameras', self._get_default_cameras())
                total_operations = total_frames * len(cameras)
                logger.info(f"Processing {total_frames} frames with {len(cameras)} cameras (PERSPECTIVE mode)")
                return self._execute_stage2_perspective(input_frames, cameras, output_dir, output_width, output_height)
        
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'output_files': []}
    
    def _execute_stage2_perspective(self, input_frames, cameras, output_dir, output_width, output_height) -> Dict:
        """Execute Stage 2 in perspective mode (E2P)"""
        try:
            output_files = []
            operation_count = 0
            total_operations = len(input_frames) * len(cameras)
            
            
            for frame_idx, frame_path in enumerate(input_frames):
                # Check for cancellation between frames
                if self.is_cancelled:
                    logger.info(f"Pipeline cancelled during perspective generation")
                    return {'success': False, 'error': 'Cancelled by user'}
                
                # Load equirectangular image
                equirect_img = cv2.imread(str(frame_path))
                
                if equirect_img is None:
                    logger.warning(f"Failed to load {frame_path}")
                    continue
                
                # Extract camera metadata
                camera_metadata = self.metadata_handler.extract_camera_metadata(str(frame_path))
                
                # Process each camera view
                for cam_idx, camera in enumerate(cameras):
                    if self.is_cancelled:
                        return {'success': False, 'error': 'Cancelled by user'}
                    
                    yaw = camera['yaw']
                    pitch = camera.get('pitch', 0)
                    roll = camera.get('roll', 0)
                    fov = camera.get('fov', DEFAULT_H_FOV)
                    
                    # Transform to perspective
                    perspective_img = self.e2p_transform.equirect_to_pinhole(
                        equirect_img,
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll,
                        h_fov=fov,
                        output_width=output_width,
                        output_height=output_height
                    )
                    
                    # Save perspective image with configured format
                    image_format = self.config.get('stage2_format', 'png')
                    extension = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'png'
                    output_filename = f"frame_{frame_idx:05d}_cam_{cam_idx:02d}.{extension}"
                    output_path = output_dir / output_filename
                    
                    # Save image - use PIL for PNG (prevents corruption), cv2 for JPEG
                    success = False
                    if extension == 'png':
                        # Use PIL for PNG to prevent chunk corruption
                        try:
                            # Convert BGR to RGB
                            perspective_rgb = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(perspective_rgb)
                            pil_img.save(str(output_path), 'PNG', compress_level=6)
                            success = True
                        except Exception as e:
                            logger.warning(f"PIL PNG save failed for {output_filename}: {e}, falling back to cv2")
                            success = cv2.imwrite(str(output_path), perspective_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    elif extension in ['jpg', 'jpeg']:
                        # JPEG: Use quality parameter
                        success = cv2.imwrite(str(output_path), perspective_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        success = cv2.imwrite(str(output_path), perspective_img)
                    
                    if not success:
                        logger.warning(f"Failed to save {output_filename}")
                        continue
                    
                    # Verify file was written correctly before embedding metadata
                    if not output_path.exists() or output_path.stat().st_size == 0:
                        logger.warning(f"Saved file is empty or missing: {output_filename}")
                        continue
                    
                    # Embed camera orientation in EXIF (skip if embedding fails)
                    try:
                        self.metadata_handler.embed_camera_orientation(
                            str(output_path),
                            yaw=yaw,
                            pitch=pitch,
                            roll=roll,
                            h_fov=fov
                        )
                    except Exception as e:
                        logger.warning(f"Failed to embed metadata in {output_filename}: {e}")
                        # Continue anyway - file is saved, just without metadata
                    
                    output_files.append(str(output_path))
                    operation_count += 1
                    
                    # Check for cancellation between cameras
                    if self.is_cancelled:
                        logger.info(f"Pipeline cancelled during perspective generation")
                        return
                    
                    # Check for pause
                    while self.is_paused:
                        if self.is_cancelled:
                            return
                        self.msleep(100)
                    
                    # Progress update
                    if operation_count % 10 == 0:
                        self.progress.emit(
                            operation_count,
                            total_operations,
                            f"Stage 2: Frame {frame_idx+1}/{len(input_frames)}, Camera {cam_idx+1}/{len(cameras)}"
                        )
            
            return {
                'success': True,
                'perspective_count': len(output_files),
                'output_files': output_files,
                'output_dir': str(output_dir)
            }
        
        except Exception as e:
            logger.error(f"Stage 2 PERSPECTIVE error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_stage2_cubemap(self, input_frames, output_dir) -> Dict:
        """Execute Stage 2 in cubemap mode (E2C)"""
        try:
            output_files = []
            
            # Get cubemap configuration
            cubemap_params = self.config.get('cubemap_params', {})
            cubemap_type = cubemap_params.get('cubemap_type', '6-face')
            face_size = cubemap_params.get('face_size', 2048)
            fov = cubemap_params.get('fov', 90)
            overlap_percent = cubemap_params.get('overlap_percent', 0)
            
            logger.info(f"Cubemap mode: {cubemap_type}, face_size={face_size}, fov={fov}, overlap={overlap_percent}%")
            
            # Define face names for 6-tile cubemap
            face_names_6 = ['front', 'back', 'left', 'right', 'top', 'bottom']
            
            # Calculate overlap degrees for 8-tile mode
            if cubemap_type == '8-tile':
                step_size = 360.0 / 8  # 45°
                overlap_degrees = (overlap_percent / 100.0) * step_size
                # 8 tiles in 4×2 grid: yaw positions 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
                # Two rows: pitch 0° and pitch calculated from user params
                tile_positions = []
                for row in range(2):
                    pitch_val = 0 if row == 0 else 30  # Top row at horizon, bottom row slightly down
                    for col in range(4):
                        yaw_val = col * 90  # 0°, 90°, 180°, 270°
                        tile_positions.append({
                            'yaw': yaw_val,
                            'pitch': pitch_val,
                            'fov': fov,
                            'name': f'tile_{row}_{col}'
                        })
            
            total_operations = len(input_frames)
            
            for frame_idx, frame_path in enumerate(input_frames):
                if self.is_cancelled:
                    return {'success': False, 'error': 'Cancelled by user'}
                
                # Load equirectangular image
                equirect_img = cv2.imread(str(frame_path))
                
                if equirect_img is None:
                    logger.warning(f"Failed to load {frame_path}")
                    continue
                
                # Extract camera metadata
                camera_metadata = self.metadata_handler.extract_camera_metadata(str(frame_path))
                
                if cubemap_type == '6-face':
                    # Generate 6 standard cubemap faces (90° FOV each)
                    # Front, Back, Left, Right, Top, Bottom
                    face_configs = [
                        {'yaw': 0, 'pitch': 0, 'name': 'front'},    # Front
                        {'yaw': 180, 'pitch': 0, 'name': 'back'},   # Back
                        {'yaw': -90, 'pitch': 0, 'name': 'left'},   # Left
                        {'yaw': 90, 'pitch': 0, 'name': 'right'},   # Right
                        {'yaw': 0, 'pitch': 90, 'name': 'top'},     # Top
                        {'yaw': 0, 'pitch': -90, 'name': 'bottom'}  # Bottom
                    ]
                    
                    for face_config in face_configs:
                        # Use E2P transform with 90° FOV for true cubemap projection
                        face_img = self.e2p_transform.equirect_to_pinhole(
                            equirect_img,
                            yaw=face_config['yaw'],
                            pitch=face_config['pitch'],
                            roll=0,
                            h_fov=90,  # Fixed 90° for standard cubemap
                            output_width=face_size,
                            output_height=face_size
                        )
                        
                        # Save face with configured format - use PIL for PNG, cv2 for JPEG
                        image_format = self.config.get('stage2_format', 'png')
                        extension = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'png'
                        output_filename = f"frame_{frame_idx:05d}_{face_config['name']}.{extension}"
                        output_path = output_dir / output_filename
                        
                        # Save image - use PIL for PNG (prevents corruption), cv2 for JPEG
                        success = False
                        if extension == 'png':
                            try:
                                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(face_rgb)
                                pil_img.save(str(output_path), 'PNG', compress_level=6)
                                success = True
                            except Exception as e:
                                logger.warning(f"PIL PNG save failed for {output_filename}: {e}, falling back to cv2")
                                success = cv2.imwrite(str(output_path), face_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                        elif extension in ['jpg', 'jpeg']:
                            success = cv2.imwrite(str(output_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        else:
                            success = cv2.imwrite(str(output_path), face_img)
                        
                        if not success:
                            logger.warning(f"Failed to save {output_filename}")
                            continue
                        
                        # Embed camera orientation in EXIF (skip if fails)
                        try:
                            self.metadata_handler.embed_camera_orientation(
                                str(output_path),
                                yaw=face_config['yaw'],
                                pitch=face_config['pitch'],
                                roll=0,
                                h_fov=90
                            )
                        except Exception as e:
                            logger.warning(f"Failed to embed metadata in {output_filename}: {e}")
                        
                        output_files.append(str(output_path))
                        
                        # Check for cancellation between faces
                        if self.is_cancelled:
                            logger.info(f"Pipeline cancelled during cubemap generation")
                            return {'success': False, 'error': 'Cancelled by user'}
                        
                        # Check for pause
                        while self.is_paused:
                            if self.is_cancelled:
                                return {'success': False, 'error': 'Cancelled by user'}
                            self.msleep(100)
                
                else:  # 8-tile grid
                    # Generate 8 tiles with custom FOV/overlap
                    for tile in tile_positions:
                        tile_img = self.e2p_transform.equirect_to_pinhole(
                            equirect_img,
                            yaw=tile['yaw'],
                            pitch=tile['pitch'],
                            roll=0,
                            h_fov=tile['fov'],
                            output_width=face_size,
                            output_height=face_size
                        )
                        
                        # Save tile with configured format - use PIL for PNG, cv2 for JPEG
                        image_format = self.config.get('stage2_format', 'png')
                        extension = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'png'
                        output_filename = f"frame_{frame_idx:05d}_{tile['name']}.{extension}"
                        output_path = output_dir / output_filename
                        
                        # Save image - use PIL for PNG (prevents corruption), cv2 for JPEG
                        success = False
                        if extension == 'png':
                            try:
                                tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(tile_rgb)
                                pil_img.save(str(output_path), 'PNG', compress_level=6)
                                success = True
                            except Exception as e:
                                logger.warning(f"PIL PNG save failed for {output_filename}: {e}, falling back to cv2")
                                success = cv2.imwrite(str(output_path), tile_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                        elif extension in ['jpg', 'jpeg']:
                            success = cv2.imwrite(str(output_path), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        else:
                            success = cv2.imwrite(str(output_path), tile_img)
                        
                        if not success:
                            logger.warning(f"Failed to save {output_filename}")
                            continue
                        
                        # Embed camera orientation (skip if fails)
                        try:
                            self.metadata_handler.embed_camera_orientation(
                                str(output_path),
                                yaw=tile['yaw'],
                                pitch=tile['pitch'],
                                roll=0,
                                h_fov=tile['fov']
                            )
                        except Exception as e:
                            logger.warning(f"Failed to embed metadata in {output_filename}: {e}")
                        
                        output_files.append(str(output_path))
                        
                        # Check for cancellation between tiles
                        if self.is_cancelled:
                            logger.info(f"Pipeline cancelled during 8-tile generation")
                            return {'success': False, 'error': 'Cancelled by user'}
                        
                        # Check for pause
                        while self.is_paused:
                            if self.is_cancelled:
                                return {'success': False, 'error': 'Cancelled by user'}
                            self.msleep(100)
                
                # Progress update
                self.progress.emit(
                    frame_idx + 1,
                    total_operations,
                    f"Stage 2 CUBEMAP: Frame {frame_idx+1}/{len(input_frames)} ({cubemap_type})"
                )
            
            faces_per_frame = 6 if cubemap_type == '6-face' else 8
            logger.info(f"Generated {len(output_files)} cubemap faces ({faces_per_frame} per frame)")
            
            return {
                'success': True,
                'cubemap_count': len(output_files),
                'output_files': output_files,
                'output_dir': str(output_dir),
                'cubemap_type': cubemap_type
            }
        
        except Exception as e:
            logger.error(f"Stage 2 CUBEMAP error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _execute_stage3(self) -> Dict:
        """Execute Stage 3: AI Masking"""
        try:
            # Initialize masker if not done
            if self.masker is None:
                # Lazy import to avoid loading torch during PyInstaller analysis
                from ..masking import MultiCategoryMasker
                
                model_size = self.config.get('model_size', 'small')
                confidence = self.config.get('confidence_threshold', 0.5)
                use_gpu = self.config.get('use_gpu', True)
                
                self.masker = MultiCategoryMasker(
                    model_size=model_size,
                    confidence_threshold=confidence,
                    use_gpu=use_gpu
                )
                
                # Set enabled categories
                categories = self.config.get('masking_categories', {})
                self.masker.set_enabled_categories(
                    persons=categories.get('persons', True),
                    personal_objects=categories.get('personal_objects', True),
                    animals=categories.get('animals', True)
                )
            
            # Get input images (auto-discovery runs ONLY ONCE)
            if self.config.get('enable_stage2', True):
                # Stage 2 was enabled - use its output directly
                input_dir = Path(self.config['output_dir']) / 'stage2_perspectives'
            else:
                # Stage 2 disabled - check for explicit input or auto-discover ONCE
                stage3_input = self.config.get('stage3_input_dir')
                if not stage3_input:
                    # Single auto-discovery attempt
                    discovered = self.discover_stage_input_folder(3, self.config['output_dir'])
                    if discovered:
                        input_dir = discovered
                    else:
                        return {
                            'success': False,
                            'error': 'Stage 3 input directory not specified and auto-discovery failed',
                            'masks_created': 0,
                            'skipped': 0,
                            'failed': 0
                        }
                else:
                    input_dir = Path(stage3_input)
            
            output_dir = Path(self.config['output_dir']) / 'stage3_masks'
            save_visualization = self.config.get('save_visualization', False)
            
            def progress_callback(current, total, message):
                self.progress.emit(current, total, f"Stage 3: {message}")
            
            def cancellation_check():
                """Check for cancellation or pause"""
                if self.is_cancelled:
                    return True
                # Wait if paused
                while self.is_paused:
                    import time
                    time.sleep(0.5)
                return self.is_cancelled
            
            # Batch process
            result = self.masker.process_batch(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                save_visualization=save_visualization,
                progress_callback=progress_callback,
                cancellation_check=cancellation_check
            )
            
            result['success'] = True
            return result
        
        except Exception as e:
            logger.error(f"Stage 3 error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_default_cameras(self) -> List[Dict]:
        """Get default 8-camera horizontal configuration"""
        cameras = []
        for i in range(DEFAULT_SPLIT_COUNT):
            yaw = (360 / DEFAULT_SPLIT_COUNT) * i
            cameras.append({
                'yaw': yaw,
                'pitch': 0,
                'roll': 0,
                'fov': DEFAULT_H_FOV
            })
        return cameras
    
    def cancel(self):
        """Cancel pipeline execution"""
        self.is_cancelled = True
        # Propagate cancellation to components
        if self.frame_extractor:
            self.frame_extractor.cancel()
        if self.sdk_extractor:
            self.sdk_extractor.cancel()
        logger.info("Pipeline cancellation requested")
    
    def pause(self):
        """Pause pipeline execution (can be resumed)"""
        self.is_paused = True
        logger.info("Pipeline pause requested")
    
    def resume(self):
        """Resume paused pipeline"""
        self.is_paused = False
        logger.info("Pipeline resume requested")


class BatchOrchestrator:
    """
    Main orchestrator for batch pipeline execution.
    Provides simple interface for running the full pipeline.
    """
    
    def __init__(self):
        """Initialize BatchOrchestrator"""
        self.worker = None
    
    def run_pipeline(self, config: Dict, 
                    progress_callback: Optional[Callable] = None,
                    stage_complete_callback: Optional[Callable] = None,
                    finished_callback: Optional[Callable] = None,
                    error_callback: Optional[Callable] = None) -> PipelineWorker:
        """
        Run the pipeline in a separate thread.
        
        Args:
            config: Pipeline configuration dictionary
            progress_callback: Called with (current, total, message)
            stage_complete_callback: Called with (stage_number, results)
            finished_callback: Called with (final_results)
            error_callback: Called with (error_message)
            
        Returns:
            PipelineWorker thread (already started)
        """
        
        self.worker = PipelineWorker(config)
        
        # Connect callbacks
        if progress_callback:
            self.worker.progress.connect(progress_callback)
        if stage_complete_callback:
            self.worker.stage_complete.connect(stage_complete_callback)
        if finished_callback:
            self.worker.finished.connect(finished_callback)
        if error_callback:
            self.worker.error.connect(error_callback)
        
        # Start worker thread
        self.worker.start()
        
        return self.worker
    
    def cancel(self):
        """Cancel running pipeline"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
    
    def pause(self):
        """Pause running pipeline"""
        if self.worker and self.worker.isRunning():
            self.worker.pause()
    
    def resume(self):
        """Resume paused pipeline"""
        if self.worker and self.worker.isRunning():
            self.worker.resume()
