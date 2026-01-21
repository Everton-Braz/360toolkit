"""
Batch Pipeline Orchestrator for 360FrameTools
Coordinates the 3-stage pipeline: Extract → Split → Mask

Uses QThread for non-blocking UI execution with progress signals.
"""

import glob
import logging
import subprocess
import time
import os
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import json
from datetime import datetime
from PIL import Image
import numpy as np

# Try to import PyTorch with CUDA support
HAS_TORCH_TRANSFORM = False
HAS_TORCH_CUDA = False
torch = None
logger_msg = "PyTorch not tested yet"

try:
    # Direct import - handle torch.distributed error gracefully
    import torch as _torch
    torch = _torch
    
    # Check if CUDA is available
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        try:
            # Test actual GPU kernel execution (catches SM compatibility issues)
            device_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            compute_cap_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
            
            # Force kernel compilation/execution
            test_tensor = torch.zeros(10, 10, device='cuda')
            test_result = test_tensor + 1
            test_sum = test_result.sum().item()
            del test_tensor, test_result
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            if test_sum == 100.0:
                HAS_TORCH_CUDA = True
                from ..transforms.e2p_transform import TorchE2PTransform
                HAS_TORCH_TRANSFORM = True
                logger_msg = f"PyTorch CUDA verified: {device_name} ({compute_cap_str})"
            else:
                logger_msg = f"PyTorch CUDA kernel test failed (got {test_sum}, expected 100)"
                
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "no kernel image" in error_msg or "cuda capability" in error_msg:
                logger_msg = f"PyTorch CUDA kernel not available for {device_name} ({compute_cap_str}): {e}"
            else:
                logger_msg = f"PyTorch CUDA test failed: {e}"
            if torch is not None:
                torch.cuda.empty_cache()
        except Exception as e:
            logger_msg = f"PyTorch CUDA test error: {e}"
    else:
        # PyTorch available but no CUDA
        logger_msg = "PyTorch available but CUDA not detected"
except ImportError as e:
    logger_msg = f"PyTorch not available: {e}"
except Exception as e:
    logger_msg = f"PyTorch import error: {e}"

from PyQt6.QtCore import QThread, pyqtSignal

from ..extraction import FrameExtractor
from ..extraction.sdk_extractor import SDKExtractor
from ..transforms import E2PTransform, E2CTransform
from .metadata_handler import MetadataHandler
from ..utils.resource_path import get_resource_path
from ..config.defaults import (
    DEFAULT_FPS, DEFAULT_H_FOV, DEFAULT_SPLIT_COUNT,
    DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT
)

logger = logging.getLogger(__name__)

# Log PyTorch/CUDA status
logger.info(f"[Stage 2 GPU] {logger_msg}")
logger.info(f"[Stage 2 GPU] HAS_TORCH_TRANSFORM={HAS_TORCH_TRANSFORM}, HAS_TORCH_CUDA={HAS_TORCH_CUDA}")


def process_frame_cpu(frame_data):
    """
    Worker function for CPU parallel processing of Stage 2.
    Must be at module level for pickling.
    Uses absolute imports for PyInstaller compatibility.
    """
    frame_path, frame_idx, cameras, output_dir, width, height, fmt = frame_data
    
    # Re-import locally using ABSOLUTE imports for PyInstaller compatibility
    import cv2
    from PIL import Image
    from src.transforms import E2PTransform
    from src.pipeline.metadata_handler import MetadataHandler
    
    transformer = E2PTransform()
    meta_handler = MetadataHandler()
    
    results = []
    
    try:
        equirect_img = cv2.imread(str(frame_path))
        if equirect_img is None:
            return {'success': False, 'error': f"Failed to load {frame_path}"}
            
        for cam_idx, camera in enumerate(cameras):
            yaw = camera['yaw']
            pitch = camera.get('pitch', 0)
            roll = camera.get('roll', 0)
            fov = camera.get('fov', 90)
            
            perspective_img = transformer.equirect_to_pinhole(
                equirect_img, yaw, pitch, roll, fov, None, width, height
            )
            
            ext = fmt if fmt in ['png', 'jpg', 'jpeg'] else 'png'
            out_name = f"frame_{frame_idx:05d}_cam_{cam_idx:02d}.{ext}"
            out_path = Path(output_dir) / out_name
            
            # Save logic
            success = False
            if ext == 'png':
                try:
                    rgb = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    pil_img.save(str(out_path), 'PNG', compress_level=6)
                    pil_img.close()
                    success = True
                except Exception:
                    success = cv2.imwrite(str(out_path), perspective_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            elif ext in ['jpg', 'jpeg']:
                success = cv2.imwrite(str(out_path), perspective_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                success = cv2.imwrite(str(out_path), perspective_img)
                
            if success:
                try:
                    meta_handler.embed_camera_orientation(str(out_path), yaw, pitch, roll, fov)
                except Exception:
                    pass
                results.append(str(out_path))
                
        return {'success': True, 'files': results}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


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
        
        # Temp folder for Stage 1 when skip_intermediate_save is enabled
        self._temp_stage1_dir = None
    
    def _cleanup_temp_stage1(self):
        """Clean up temp Stage 1 folder after Stage 2 completes (if skip_intermediate_save was used)"""
        if self._temp_stage1_dir and Path(self._temp_stage1_dir).exists():
            import shutil
            try:
                shutil.rmtree(self._temp_stage1_dir)
                logger.info(f"[CLEANUP] Deleted temp Stage 1 folder: {self._temp_stage1_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp folder: {e}")
            self._temp_stage1_dir = None
    
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
            
            # Stage 2: Split Perspectives (Skip if direct masking mode)
            skip_transform = self.config.get('skip_transform', False)
            
            if skip_transform:
                logger.info("=== Stage 2: SKIPPED (Direct Masking Mode) ===")
                logger.info("Stage 1 frames will be used directly for Stage 3 masking")
                # Pass Stage 1 output to Stage 3 input
                results['stage2'] = {
                    'success': True,
                    'skipped': True,
                    'message': 'Transform skipped - using equirectangular/fisheye frames directly'
                }
                stages_executed.append(2)  # Mark as executed (but skipped)
                self.stage_complete.emit(2, results['stage2'])
            elif self.config.get('enable_stage2', True):
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
                
                # Clean up temp Stage 1 folder if skip_intermediate_save was enabled
                self._cleanup_temp_stage1()
                
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
            
            # Check if we should skip saving intermediate files
            skip_intermediate = self.config.get('skip_intermediate_save', False)
            
            if skip_intermediate:
                # Use a temp folder that will be cleaned up after Stage 2
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix='360toolkit_stage1_')
                output_dir = Path(temp_dir)
                self._temp_stage1_dir = temp_dir  # Store for cleanup
                logger.info(f"[FAST] Using temp folder for Stage 1: {temp_dir}")
            else:
                output_dir = Path(self.config['output_dir']) / 'stage1_frames'
                self._temp_stage1_dir = None
            
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

                    quality_descriptions = {
                        'best': 'AI Stitching (maximum quality)',
                        'good': 'Optical Flow (balanced)',
                        'draft': 'Template (fast preview)'
                    }
                    logger.info(
                        f"Using SDK quality preset: {quality.upper()} - "
                        f"{quality_descriptions.get(quality, 'custom preset')}"
                    )
                    
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
                        
                        # Check if fallback is allowed
                        if not self.config.get('allow_fallback', True):
                            logger.error("Fallback disabled by configuration. Aborting.")
                            raise sdk_error
                        
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
        """Execute Stage 2 in perspective mode (E2P) with GPU/CPU optimization"""
        try:
            output_files = []
            total_frames = len(input_frames)
            total_operations = total_frames * len(cameras)
            image_format = self.config.get('stage2_format', 'png')
            
            # Check for GPU acceleration with comprehensive compatibility testing
            use_gpu = False
            if HAS_TORCH_TRANSFORM and HAS_TORCH_CUDA:
                logger.info("[Stage 2 GPU] Starting GPU compatibility check...")
                try:
                    # Get GPU info
                    device_name = torch.cuda.get_device_name(0)
                    try:
                        compute_capability = torch.cuda.get_device_capability(0)
                        compute_cap_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
                    except Exception:
                        compute_cap_str = "unknown"
                    
                    logger.info(f"[Stage 2 GPU] Testing GPU: {device_name} ({compute_cap_str})")
                    
                    # Test GPU with actual kernel execution (not just memory allocation)
                    # This catches RTX 50-series "no kernel image available" errors early
                    test_tensor = torch.zeros(10, 10, device='cuda')
                    test_result = test_tensor + 1  # Force CUDA kernel execution
                    test_mean = test_result.mean().item()  # Force device synchronization
                    del test_tensor, test_result
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    
                    use_gpu = True
                    logger.info(f"[Stage 2 GPU] ✅ GPU acceleration ENABLED: {device_name} ({compute_cap_str})")
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    logger.error(f"[Stage 2 GPU] ❌ GPU test failed: {e}")
                    if "no kernel image" in error_msg or "cuda capability" in error_msg or "sm_" in error_msg:
                        logger.error(f"[Stage 2 GPU] GPU incompatibility detected")
                        logger.error(f"[Stage 2 GPU]    GPU: {device_name} ({compute_cap_str})")
                        logger.error(f"[Stage 2 GPU]    PyTorch CUDA kernels not available for this GPU architecture")
                        if "sm_12" in compute_cap_str or "RTX 50" in device_name:
                            logger.error(f"[Stage 2 GPU]    RTX 50-series requires PyTorch nightly build")
                        logger.warning(f"[Stage 2 GPU] => Using CPU multiprocessing instead")
                    else:
                        logger.warning(f"[Stage 2 GPU] GPU check failed: {e}. Falling back to CPU.")
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
                except Exception as e:
                    logger.warning(f"[Stage 2 GPU] GPU test failed: {e}. Falling back to CPU.")
                    try:
                        if hasattr(torch, 'cuda') and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
            else:
                logger.warning("[Stage 2 GPU] GPU transform not available (HAS_TORCH_TRANSFORM or HAS_TORCH_CUDA is False)")
            
            # GPU Execution Block
            if use_gpu:
                logger.info("[Stage 2 GPU] 🚀 Starting GPU-accelerated processing...")
                try:
                    from concurrent.futures import ThreadPoolExecutor
                    import queue
                    import threading
                    
                    # GPU Path: Optimized batch processing with async I/O
                    logger.info("[Stage 2 GPU] Initializing TorchE2PTransform...")
                    transformer = TorchE2PTransform()
                    logger.info(f"[Stage 2 GPU] Transformer initialized on device: {transformer.device}")
                    
                    # Get sample image dimensions for batch size calculation
                    sample_img = cv2.imread(str(input_frames[0]))
                    if sample_img is None:
                        raise RuntimeError("Failed to load sample image for batch size calculation")
                    
                    input_height, input_width = sample_img.shape[:2]
                    
                    # Pass num_cameras to batch size calculation for accurate VRAM estimation
                    num_cameras = len(cameras)
                    auto_batch_size = transformer.get_optimal_batch_size(
                        input_height, input_width, output_height, output_width, num_cameras
                    )
                    
                    # OVERRIDE: Use batch size 16 for optimal GPU utilization (tested)
                    # Auto-detection gives 8, but testing shows 16 is 8% faster (12648 vs 11659 FPS)
                    # Larger batches also amortize I/O overhead better
                    batch_size = 16 if auto_batch_size <= 16 else auto_batch_size
                    logger.info(f"Using batch size: {batch_size} frames (auto: {auto_batch_size}, optimized for I/O)")
                    
                    del sample_img  # Free memory
                    
                    # Use JPEG for faster saving (10x faster than PNG)
                    ext = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'jpg'
                    if ext == 'png':
                        logger.warning("PNG format selected - saving will be slower. Consider JPEG for faster processing.")
                    
                    # Thread pool for async image saving (I/O bound)
                    # Increased to 24 for faster disk writes (I/O bottleneck mitigation)
                    save_executor = ThreadPoolExecutor(max_workers=24)
                    save_futures = []
                    
                    def save_image_async(out_img, out_path, ext, yaw, pitch, roll, fov):
                        """Async image saving function"""
                        try:
                            success = False
                            if ext == 'png':
                                # Fast PNG with minimal compression
                                success = cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                            elif ext in ['jpg', 'jpeg']:
                                success = cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            else:
                                success = cv2.imwrite(str(out_path), out_img)
                            
                            if success:
                                try:
                                    self.metadata_handler.embed_camera_orientation(str(out_path), yaw, pitch, roll, fov)
                                except: pass
                                return str(out_path)
                            return None
                        except Exception as e:
                            logger.warning(f"Failed to save {out_path}: {e}")
                            return None
                    
                    # Thread pool for async image loading (I/O bound)
                    # MAXIMIZED for I/O bottleneck (77.8% of pipeline time is disk I/O!)
                    # More threads = more concurrent disk reads = better throughput
                    # With 4 workers: 4.18s for 30 images (7680x3840)
                    # With 24 workers: 2.30s for 30 images = 1.8x faster
                    # With 32 workers: ~2.0s for 30 images (target)
                    load_executor = ThreadPoolExecutor(max_workers=32)
                    
                    # LRU cache for loaded images (RAM cache to avoid repeated disk reads)
                    # Max size = 4 images × ~90 MB each = 360 MB RAM (acceptable overhead)
                    from functools import lru_cache
                    image_cache = {}
                    cache_lock = threading.Lock()
                    MAX_CACHE_SIZE = 4
                    
                    def load_image(path):
                        """Load image with RAM caching to eliminate repeated disk reads"""
                        path_str = str(path)
                        
                        # Check cache first
                        with cache_lock:
                            if path_str in image_cache:
                                # Cache hit - return copy to avoid mutation
                                return image_cache[path_str].clone()
                        
                        # Cache miss - load from disk
                        img = cv2.imread(path_str)
                        if img is None:
                            return None
                        
                        # Convert to tensor and pin memory for faster GPU transfer
                        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                        tensor = tensor.pin_memory() if tensor.device.type == 'cpu' else tensor
                        
                        # Add to cache (LRU eviction)
                        with cache_lock:
                            if len(image_cache) >= MAX_CACHE_SIZE:
                                # Remove oldest entry
                                oldest_key = next(iter(image_cache))
                                del image_cache[oldest_key]
                            image_cache[path_str] = tensor
                        
                        return tensor
                    
                    # Process frames in batches with PREFETCHING
                    # Overlaps I/O (loading next batch) with GPU compute (processing current batch)
                    total_batches = (total_frames + batch_size - 1) // batch_size
                    
                    # Pre-submit first batch load
                    pending_load_futures = None
                    pending_batch_start = 0
                    
                    for batch_idx, batch_start in enumerate(range(0, total_frames, batch_size)):
                        if self.is_cancelled:
                            save_executor.shutdown(wait=False)
                            load_executor.shutdown(wait=False)
                            return {'success': False, 'error': 'Cancelled by user'}
                        
                        batch_end = min(batch_start + batch_size, total_frames)
                        batch_paths = input_frames[batch_start:batch_end]
                        
                        # If first batch or no prefetch, submit load now
                        if pending_load_futures is None:
                            load_futures = {load_executor.submit(load_image, p): (i, p) 
                                           for i, p in enumerate(batch_paths, start=batch_start)}
                        else:
                            # Use prefetched futures
                            load_futures = pending_load_futures
                        
                        # PREFETCH: Start loading NEXT batch while we process current batch
                        next_batch_start = batch_end
                        if next_batch_start < total_frames:
                            next_batch_end = min(next_batch_start + batch_size, total_frames)
                            next_batch_paths = input_frames[next_batch_start:next_batch_end]
                            pending_load_futures = {load_executor.submit(load_image, p): (i, p) 
                                                   for i, p in enumerate(next_batch_paths, start=next_batch_start)}
                        else:
                            pending_load_futures = None
                        
                        batch_tensors = []
                        batch_frame_indices = []
                        
                        for future in as_completed(load_futures):
                            frame_idx, frame_path = load_futures[future]
                            tensor = future.result()
                            if tensor is not None:
                                batch_tensors.append((frame_idx, tensor))
                        
                        # Sort by frame index to maintain order
                        batch_tensors.sort(key=lambda x: x[0])
                        batch_frame_indices = [x[0] for x in batch_tensors]
                        tensors_only = [x[1] for x in batch_tensors]
                        
                        if not tensors_only:
                            continue
                        
                        # Stack and transfer to GPU with async copy (pinned memory enables this)
                        batch = torch.stack(tensors_only)
                        del tensors_only, batch_tensors
                        
                        batch = batch.to(transformer.device, non_blocking=True) / 255.0
                        try:
                            torch.cuda.synchronize()
                        except:
                            pass
                        
                        # Process ALL cameras at once - collect all outputs then save
                        all_outputs = []  # [(frame_idx, cam_idx, out_img, yaw, pitch, roll, fov), ...]
                        
                        for cam_idx, camera in enumerate(cameras):
                            yaw = camera['yaw']
                            pitch = camera.get('pitch', 0)
                            roll = camera.get('roll', 0)
                            fov = camera.get('fov', DEFAULT_H_FOV)
                            
                            # Batch transform on GPU
                            out_batch = transformer.batch_equirect_to_pinhole(
                                batch, yaw, pitch, roll, fov, None, output_width, output_height
                            )
                            
                            # OPTIMIZED: Convert FP16→uint8 ON GPU before transfer (12.5x faster!)
                            # This reduces transfer size by 8x and eliminates CPU-side conversion
                            out_batch_uint8 = (out_batch.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
                            out_batch_cpu = out_batch_uint8.cpu().numpy()
                            
                            for i, frame_idx in enumerate(batch_frame_indices):
                                all_outputs.append((frame_idx, cam_idx, out_batch_cpu[i], yaw, pitch, roll, fov))
                            
                            del out_batch, out_batch_uint8
                        
                        # Free GPU memory before async saves
                        del batch
                        try:
                            torch.cuda.empty_cache()
                        except:
                            pass
                        
                        # Submit all saves asynchronously
                        for frame_idx, cam_idx, out_img, yaw, pitch, roll, fov in all_outputs:
                            out_name = f"frame_{frame_idx:05d}_cam_{cam_idx:02d}.{ext}"
                            out_path = output_dir / out_name
                            future = save_executor.submit(save_image_async, out_img.copy(), out_path, ext, yaw, pitch, roll, fov)
                            save_futures.append(future)
                        
                        del all_outputs
                        
                        # Progress update
                        self.progress.emit(batch_end * len(cameras), total_operations, 
                            f"Stage 2 (GPU): Batch {batch_idx + 1}/{total_batches}")
                    
                    # Wait for all saves to complete
                    logger.info("Waiting for async saves to complete...")
                    for future in as_completed(save_futures):
                        result = future.result()
                        if result:
                            output_files.append(result)
                    
                    save_executor.shutdown(wait=True)
                    load_executor.shutdown(wait=True)
                    
                    # Completed successfully on GPU
                    logger.info(f"[OK] GPU batch processing complete: {len(output_files)} perspectives created")
                    return {
                        'success': True,
                        'perspective_count': len(output_files),
                        'output_files': output_files,
                        'output_dir': str(output_dir)
                    }

                except RuntimeError as e:
                    if "no kernel image" in str(e) or "CUDA error" in str(e):
                        logger.warning(f"GPU Stage 2 failed (RTX 50-series incompatibility): {e}")
                        logger.warning("Falling back to CPU Multiprocessing...")
                        use_gpu = False # Fall through to CPU block
                        output_files = [] # Reset output files
                        try:
                            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                    else:
                        raise e # Re-raise other errors

            # CPU Execution Block (Fallback or Default)
            if not use_gpu:
                # CPU Path: Multiprocessing
                logger.info("[Stage 2 CPU] ⚠️ Using CPU multiprocessing (GPU not available or failed)")
                # Limit workers to avoid OOM with 8K images. 
                # 8K image ~100MB. 20 workers = 2GB. Safe.
                max_workers = min(os.cpu_count(), 12) 
                logger.info(f"[Stage 2 CPU] Using {max_workers} CPU workers")
                
                # Prepare tasks
                tasks = []
                for idx, path in enumerate(input_frames):
                    tasks.append((
                        str(path), idx, cameras, str(output_dir), 
                        output_width, output_height, image_format
                    ))
                
                completed_ops = 0
                # Use 'spawn' context for PyInstaller compatibility on Windows
                mp_context = multiprocessing.get_context('spawn')
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                    futures = {executor.submit(process_frame_cpu, task): task for task in tasks}
                    
                    for future in as_completed(futures):
                        if self.is_cancelled:
                            executor.shutdown(wait=False)
                            return {'success': False, 'error': 'Cancelled'}
                            
                        result = future.result()
                        if result['success']:
                            output_files.extend(result['files'])
                            completed_ops += len(cameras)
                            self.progress.emit(completed_ops, total_operations, 
                                f"Stage 2 (CPU): {completed_ops}/{total_operations} views")
                        else:
                            logger.error(f"Frame processing failed: {result.get('error')}")
            
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
                                # Explicitly close file handle to prevent WinError 32 on metadata embedding
                                pil_img.close()
                                del pil_img
                                time.sleep(0.01)  # 10ms delay for Windows file lock release
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
                                # Explicitly close file handle to prevent WinError 32 on metadata embedding
                                pil_img.close()
                                del pil_img
                                time.sleep(0.01)  # 10ms delay for Windows file lock release
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
            # Determine input directory based on skip_transform flag
            skip_transform = self.config.get('skip_transform', False)
            
            if skip_transform:
                # Use Stage 1 output (equirectangular/fisheye frames) directly
                input_dir = Path(self.config['output_dir']) / 'stage1_frames'
                logger.info("[Direct Masking] Using Stage 1 frames (equirectangular/fisheye) for masking")
            else:
                # Use Stage 2 output (perspective images)
                input_dir = Path(self.config.get('stage3_input_dir', 
                                               str(Path(self.config['output_dir']) / 'stage2_perspectives')))
                logger.info(f"Using Stage 2 perspectives for masking: {input_dir}")
            
            # Initialize masker if not done
            if self.masker is None:
                model_size = self.config.get('model_size', 'small')
                confidence = self.config.get('confidence_threshold', 0.5)
                use_gpu = self.config.get('use_gpu', True)
                
                # Check for ONNX model availability - prefer YOLO26 (faster, NMS-free)
                # YOLO26 is 3-4x faster than YOLOv8 with same accuracy
                model_map_yolo26 = {
                    'nano': 'yolo26n-seg.onnx',
                    'small': 'yolo26s-seg.onnx',
                    'medium': 'yolo26m-seg.onnx',
                    'large': 'yolo26m-seg.onnx',   # Fallback to medium (no large YOLO26)
                    'xlarge': 'yolo26m-seg.onnx'   # Fallback to medium (no xlarge YOLO26)
                }
                model_map_yolov8 = {
                    'nano': 'yolov8n-seg.onnx',
                    'small': 'yolov8s-seg.onnx',
                    'medium': 'yolov8m-seg.onnx',
                    'large': 'yolov8l-seg.onnx',
                    'xlarge': 'yolov8x-seg.onnx'
                }
                
                # Try YOLO26 first, then YOLOv8
                onnx_model_name = model_map_yolo26.get(model_size, 'yolo26s-seg.onnx')
                onnx_path = get_resource_path(onnx_model_name)
                
                if not onnx_path.exists():
                    # Fallback to YOLOv8
                    onnx_model_name = model_map_yolov8.get(model_size, 'yolov8s-seg.onnx')
                    onnx_path = get_resource_path(onnx_model_name)
                    logger.info(f"YOLO26 not found, falling back to YOLOv8: {onnx_model_name}")
                else:
                    logger.info(f"Using YOLO26 model (3-4x faster): {onnx_model_name}")
                
                use_onnx = False
                onnx_error = "Unknown error"
                
                if onnx_path.exists():
                    try:
                        # CRITICAL FIX: Add DLL directory for ONNX Runtime in frozen app
                        import sys
                        import os
                        
                        # Determine base path for frozen app (onedir mode)
                        if getattr(sys, 'frozen', False):
                            # In onedir mode, sys.executable is the .exe
                            # The _internal folder is usually next to it (PyInstaller 6+)
                            # or dependencies are in the same folder
                            exe_dir = Path(sys.executable).parent
                            internal_dir = exe_dir / '_internal'
                            
                            # DIAGNOSTIC: List what DLLs we have
                            logger.info("=== ONNX Runtime DLL Diagnostics ===")
                            ort_capi_path = internal_dir / 'onnxruntime' / 'capi'
                            if ort_capi_path.exists():
                                logger.info(f"ONNX capi folder exists: {ort_capi_path}")
                                dlls = list(glob.glob(str(ort_capi_path / '*.dll')))
                                logger.info(f"DLLs in capi: {[os.path.basename(d) for d in dlls]}")
                                
                                pyds = list(glob.glob(str(ort_capi_path / '*.pyd')))
                                logger.info(f"PYDs in capi: {[os.path.basename(d) for d in pyds]}")
                            else:
                                logger.warning(f"ONNX capi folder NOT FOUND: {ort_capi_path}")
                            
                            # Check for MSVC runtimes
                            msvc_dlls = list(glob.glob(str(internal_dir / 'msvcp*.dll')))
                            msvc_dlls += list(glob.glob(str(internal_dir / 'vcruntime*.dll')))
                            logger.info(f"MSVC runtimes in _internal: {[os.path.basename(d) for d in msvc_dlls]}")
                            
                            # Check current PATH
                            logger.info(f"Current PATH dirs: {os.environ.get('PATH', '').split(os.pathsep)[:5]}")
                            logger.info("=== End Diagnostics ===")
                            
                            # Add _internal/onnxruntime/capi to DLL search path
                            if ort_capi_path.exists():
                                logger.info(f"Adding ONNX DLL path: {ort_capi_path}")
                                os.add_dll_directory(str(ort_capi_path))
                            
                            # Add _internal root (for msvcp140.dll etc)
                            if internal_dir.exists():
                                logger.info(f"Adding Internal DLL path: {internal_dir}")
                                os.add_dll_directory(str(internal_dir))
                                
                            # Add exe dir (just in case)
                            logger.info(f"Adding Exe DLL path: {exe_dir}")
                            os.add_dll_directory(str(exe_dir))

                        logger.info("Attempting to import onnxruntime...")
                        import onnxruntime
                        logger.info(f"Successfully imported onnxruntime {onnxruntime.__version__}")
                        from ..masking.onnx_masker import ONNXMasker
                        logger.info(f"Found ONNX model {onnx_path}, using ONNXMasker")
                        self.masker = ONNXMasker(
                            model_path=str(onnx_path),
                            confidence_threshold=confidence,
                            use_gpu=use_gpu
                        )
                        use_onnx = True
                    except ImportError as e:
                        onnx_error = f"ImportError: {e}"
                        logger.warning(f"ONNX model found but onnxruntime not installed/working. Error: {e}")
                    except Exception as e:
                        onnx_error = f"InitError: {e}"
                        logger.warning(f"Failed to initialize ONNXMasker: {e}. Falling back to PyTorch.")
                else:
                    onnx_error = f"File not found at {onnx_path}"
                    logger.warning(f"ONNX model file not found at: {onnx_path}")

                if not use_onnx:
                    logger.info(f"ONNX initialization failed ({onnx_error}). Attempting PyTorch fallback.")
                    
                    # Lazy import to avoid loading torch during PyInstaller analysis
                    try:
                        from ..masking import MultiCategoryMasker
                        
                        self.masker = MultiCategoryMasker(
                            model_size=model_size,
                            confidence_threshold=confidence,
                            use_gpu=use_gpu
                        )
                    except ImportError as e:
                        # If we are here, it means we failed to use ONNX AND failed to use PyTorch
                        error_msg = (
                            f"Stage 3 Initialization Failed: Could not load masking engine.\n"
                            f"1. ONNX Model check: Failed ({onnx_error})\n"
                            f"2. PyTorch fallback: Failed ({str(e)})\n"
                            f"Ensure '{onnx_model_name}' is present in the application folder."
                        )
                        logger.error(error_msg)
                        return {
                            'success': False,
                            'error': error_msg,
                            'masks_created': 0,
                            'skipped': 0,
                            'failed': 0
                        }
                
                # Set enabled categories
                categories = self.config.get('masking_categories', {})
                self.masker.set_enabled_categories(
                    persons=categories.get('persons', True),
                    personal_objects=categories.get('personal_objects', True),
                    animals=categories.get('animals', True)
                )
                
                # Set specific class IDs if provided (fine-grained control from UI)
                masking_classes = self.config.get('masking_classes', {})
                if masking_classes:
                    self.masker.set_specific_classes(
                        persons_classes=masking_classes.get('persons'),
                        objects_classes=masking_classes.get('personal_objects'),
                        animals_classes=masking_classes.get('animals')
                    )
            
            # Get input images based on skip_transform flag
            skip_transform = self.config.get('skip_transform', False)
            
            if skip_transform:
                # Direct Masking Mode - use Stage 1 frames (equirectangular/fisheye)
                input_dir = Path(self.config['output_dir']) / 'stage1_frames'
                logger.info(f"[Direct Masking] Using Stage 1 frames directly: {input_dir}")
            elif self.config.get('enable_stage2', True):
                # Stage 2 was enabled - use its output (perspective images)
                input_dir = Path(self.config['output_dir']) / 'stage2_perspectives'
                logger.info(f"Using Stage 2 perspectives: {input_dir}")
            else:
                # Stage 2 disabled - check for explicit input or auto-discover ONCE
                stage3_input = self.config.get('stage3_input_dir')
                if not stage3_input:
                    # Single auto-discovery attempt
                    discovered = self.discover_stage_input_folder(3, self.config['output_dir'])
                    if discovered:
                        input_dir = discovered
                        logger.info(f"Auto-discovered Stage 3 input: {input_dir}")
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
                    logger.info(f"Using specified Stage 3 input: {input_dir}")
            
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
