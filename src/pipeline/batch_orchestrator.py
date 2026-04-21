"""
Batch Pipeline Orchestrator for 360FrameTools
Coordinates the 3-stage pipeline: Extract → Split → Mask

Uses QThread for non-blocking UI execution with progress signals.
"""

import glob
import logging
import shutil
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

from src.utils.runtime_backends import is_usable_torch_module

# Try to import PyTorch with CUDA support
HAS_TORCH_TRANSFORM = False
HAS_TORCH_CUDA = False
torch = None
TorchE2PTransform = None
logger_msg = "PyTorch not tested yet"

def _test_torch_cuda():
    """Test PyTorch CUDA availability. Called lazily to avoid import-time issues."""
    global HAS_TORCH_CUDA, HAS_TORCH_TRANSFORM, torch, TorchE2PTransform, logger_msg
    import sys as _sys
    try:
        import torch as _torch
        if not is_usable_torch_module(_torch):
            raise ImportError("incomplete torch runtime")
        torch = _torch
        
        # Check if TorchE2PTransform is available
        try:
            from src.transforms.e2p_transform import TorchE2PTransform
            HAS_TORCH_TRANSFORM = True
        except (ImportError, Exception):
            pass
        
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                compute_capability = torch.cuda.get_device_capability(0)
                compute_cap_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
                
                test_tensor = torch.zeros(10, 10, device='cuda')
                test_result = test_tensor + 1
                test_sum = test_result.sum().item()
                del test_tensor, test_result
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                if test_sum == 100.0:
                    HAS_TORCH_CUDA = True
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
            logger_msg = "PyTorch available but CUDA not detected"
    except (ImportError, OSError, AttributeError) as e:
        # Clean up partially-initialized torch from sys.modules
        for key in list(_sys.modules.keys()):
            if key == 'torch' or key.startswith('torch.'):
                del _sys.modules[key]
        torch = None
        logger_msg = f"PyTorch not available: {e}"
    except RuntimeError as e:
        # Special case: torch nightly build docstring conflict — torch IS already
        # imported and functional; the error fires when __init__ tries to set
        # docstrings for the second time.  Use the already-loaded module.
        if 'already has a docstring' in str(e) and 'torch' in _sys.modules:
            torch = _sys.modules['torch']
            try:
                from src.transforms.e2p_transform import TorchE2PTransform as _T
                TorchE2PTransform = _T
                HAS_TORCH_TRANSFORM = True
            except Exception:
                pass
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0)
                    cc = torch.cuda.get_device_capability(0)
                    HAS_TORCH_CUDA = True
                    logger_msg = f"PyTorch CUDA verified: {device_name} (sm_{cc[0]}{cc[1]})"
                except Exception:
                    logger_msg = "PyTorch CUDA partially available"
            else:
                logger_msg = "PyTorch available (docstring quirk), CUDA not detected"
        else:
            for key in list(_sys.modules.keys()):
                if key == 'torch' or key.startswith('torch.'):
                    del _sys.modules[key]
            torch = None
            logger_msg = f"PyTorch import error: {e}"
    except Exception as e:
        for key in list(_sys.modules.keys()):
            if key == 'torch' or key.startswith('torch.'):
                del _sys.modules[key]
        torch = None
        logger_msg = f"PyTorch import error: {e}"

from PyQt6.QtCore import QThread, pyqtSignal

from src.extraction import FrameExtractor
from src.extraction.sdk_extractor import SDKExtractor, IncompleteSDKExtractionError
from src.transforms import E2PTransform, E2CTransform, OpenCLE2PTransform
from src.pipeline.metadata_handler import MetadataHandler
from src.utils.dependency_provisioning import resolve_masking_model_path
from src.utils.resource_path import get_resource_path
from src.utils.color_correction import apply_color_corrections
from src.config.defaults import (
    DEFAULT_FPS, DEFAULT_H_FOV, DEFAULT_SPLIT_COUNT,
    DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT
)
from src.pipeline.stage2_naming import (
    build_stage2_frame_records,
    normalize_stage2_layout_mode,
    normalize_stage2_numbering_mode,
    perspective_output_sort_key,
    resolve_cubemap_output_path,
    resolve_perspective_output_path,
    sort_stage2_input_frames,
)

logger = logging.getLogger(__name__)
_DLL_DIR_HANDLES = []

# Log PyTorch/CUDA status
logger.info(f"[Split GPU] {logger_msg}")
logger.info(f"[Split GPU] HAS_TORCH_TRANSFORM={HAS_TORCH_TRANSFORM}, HAS_TORCH_CUDA={HAS_TORCH_CUDA}")


def process_frame_cpu(frame_data):
    """
    Worker function for CPU parallel processing of perspective splitting.
    Must be at module level for pickling.
    Uses absolute imports for PyInstaller compatibility.
    """
    frame_path, output_frame_id, cameras, output_dir, width, height, fmt, layout_mode = frame_data
    
    # Re-import locally using ABSOLUTE imports for PyInstaller compatibility
    import cv2
    from PIL import Image
    from src.transforms import E2PTransform
    from src.pipeline.metadata_handler import MetadataHandler
    from src.pipeline.stage2_naming import resolve_perspective_output_path
    
    transformer = E2PTransform()
    meta_handler = MetadataHandler()
    
    results = []
    
    try:
        equirect_img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if equirect_img is None:
            return {'success': False, 'error': f"Failed to load {frame_path}"}

        # Detect alpha channel — if present, output MUST be PNG (JPEG has no alpha)
        has_alpha = equirect_img.ndim == 3 and equirect_img.shape[2] == 4

        for cam_idx, camera in enumerate(cameras):
            yaw = camera['yaw']
            pitch = camera.get('pitch', 0)
            roll = camera.get('roll', 0)
            fov = camera.get('fov', 90)
            
            perspective_img = transformer.equirect_to_pinhole(
                equirect_img, yaw, pitch, roll, fov, None, width, height
            )
            
            ext = fmt if fmt in ['png', 'jpg', 'jpeg'] else 'png'
            if has_alpha and ext != 'png':
                ext = 'png'  # Force PNG to preserve alpha channel
            out_path = resolve_perspective_output_path(output_dir, output_frame_id, cam_idx, ext, layout_mode)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save logic
            success = False
            if ext == 'png':
                try:
                    if has_alpha:
                        rgba = cv2.cvtColor(perspective_img, cv2.COLOR_BGRA2RGBA)
                        pil_img = Image.fromarray(rgba, 'RGBA')
                    else:
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
        
    def discover_stage_input_folder(self, stage: int, output_dir: str) -> Optional[Path]:
        """
        Smart folder discovery for individual stage processing.
        
        Extraction: Looks for input file (returns None - user must select)
        Split: Looks for extracted_frames folder with equirectangular images
        Masking: Looks for perspective_views folder with perspective images
        
        Returns Path to folder if found, None if not found (user must select manually)
        """
        output_path = Path(output_dir)
        
        if stage == 1:
            # Extraction requires input file - user must select
            return None
        
        elif stage == 2:
            # Look for extracted_frames folder (match common image extensions, case-insensitive)
            stage1_folder = output_path / 'extracted_frames'
            if self._dir_has_images(stage1_folder):
                logger.info(f"[OK] Auto-discovered extraction output: {stage1_folder}")
                return stage1_folder
            return None
        
        elif stage == 3:
            # Look for perspective_views first, then extracted_frames (equirect masking)
            # If output_path itself is an images folder (user set output dir to the images folder)
            if self._dir_has_images(output_path):
                logger.info(f"[OK] output_dir itself is the masking input: {output_path}")
                return output_path
            for candidate in ('perspective_views', 'extracted_frames'):
                folder = output_path / candidate
                if self._dir_has_images(folder):
                    logger.info(f"[OK] Auto-discovered masking input: {folder}")
                    return folder
            return None
        
        return None
    
    def set_stage_input_folder(self, stage: int, folder_path: Path):
        """
        Manually set input folder for individual stage processing.
        Updates config to process only that stage with the provided input.
        """
        folder_path = Path(folder_path)
        
        if stage == 2:
            # Split input is equirectangular images from extraction
            self.config['stage2_input_dir'] = str(folder_path)
            # Disable extraction, enable split
            self.config['enable_stage1'] = False
            self.config['enable_stage2'] = True
            self.config['enable_stage3'] = False
            logger.info(f"[OK] Set split input: {folder_path}")
        
        elif stage == 3:
            # Masking input is perspective images from split
            self.config['stage3_input_dir'] = str(folder_path)
            # Disable extraction & split, enable masking
            self.config['enable_stage1'] = False
            self.config['enable_stage2'] = False
            self.config['enable_stage3'] = True
            logger.info(f"[OK] Set masking input: {folder_path}")

    @staticmethod
    def _image_patterns() -> tuple[str, ...]:
        return ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')

    def _dir_has_images(self, folder: Optional[Path], recursive: bool = True) -> bool:
        if not folder or not folder.exists() or not folder.is_dir():
            return False
        iterator = folder.rglob if recursive else folder.glob
        for pattern in self._image_patterns():
            if any(iterator(pattern)):
                return True
        return False

    def _detect_project_image_source(self, output_root: Path, folder: Optional[Path]) -> str:
        if folder is None:
            return 'custom'
        try:
            resolved = folder.resolve()
        except Exception:
            resolved = folder
        if resolved == (output_root / 'perspective_views'):
            return 'perspective'
        if resolved == (output_root / 'extracted_frames'):
            return 'equirect'
        return 'custom'

    def _resolve_project_image_dir(
        self,
        output_root: Path,
        source: str,
        alignment_mode: str = 'perspective_reconstruction',
    ) -> Optional[Path]:
        perspective_dir = output_root / 'perspective_views'
        equirect_dir = output_root / 'extracted_frames'

        if source == 'perspective':
            return perspective_dir if self._dir_has_images(perspective_dir) else None
        if source == 'equirect':
            return equirect_dir if self._dir_has_images(equirect_dir) else None

        ordered = [equirect_dir, perspective_dir] if alignment_mode == 'panorama_sfm' else [perspective_dir, equirect_dir]
        for folder in ordered:
            if self._dir_has_images(folder):
                return folder
        return None

    def _images_have_alpha_channel(self, folder: Optional[Path], sample_limit: int = 8) -> bool:
        if not folder or not folder.exists() or not folder.is_dir():
            return False

        image_paths: list[Path] = []
        for pattern in self._image_patterns():
            image_paths.extend(path for path in folder.rglob(pattern) if path.is_file())

        for path in sorted(image_paths)[:sample_limit]:
            try:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            except Exception:
                image = None
            if image is not None and image.ndim == 3 and image.shape[2] == 4:
                return True
        return False

    def _resolve_project_masks_dir(
        self,
        output_root: Path,
        source: str,
        image_source: str,
    ) -> Optional[Path]:
        mask_dirs = {
            'perspective': output_root / 'masks_perspective',
            'equirect': output_root / 'masks_equirect',
            'custom': output_root / 'masks_custom',
        }
        legacy_dir = output_root / 'masks'

        def _existing(folder: Optional[Path]) -> Optional[Path]:
            if folder and folder.exists() and any(folder.rglob('*.png')):
                return folder
            return None

        if source in ('match_images', 'match_reconstruction'):
            source = image_source
        if source == 'none':
            return None
        if source in mask_dirs:
            return _existing(mask_dirs[source]) or _existing(legacy_dir)

        ordered = []
        if image_source in mask_dirs:
            ordered.append(mask_dirs[image_source])
        ordered.extend([legacy_dir, mask_dirs['perspective'], mask_dirs['equirect'], mask_dirs['custom']])
        for folder in ordered:
            existing = _existing(folder)
            if existing:
                return existing
        return None

    def _sync_legacy_masks_dir(self, source_dir: Path, output_root: Path):
        legacy_dir = output_root / 'masks'
        if source_dir == legacy_dir or not source_dir.exists():
            return
        import shutil
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)
        shutil.copytree(source_dir, legacy_dir)

    def _should_mask_before_split(self) -> bool:
        if not self.config.get('enable_stage2', True):
            return False
        if not self.config.get('enable_stage3', True):
            return False
        if self.config.get('skip_transform', False):
            return False

        masking_engine = str(self.config.get('masking_engine', 'yolo') or 'yolo').strip().lower()
        if masking_engine == 'sam_vitb':
            masking_engine = 'sam3_cpp'
        elif masking_engine in {'yolo_onnx', 'yolo_pytorch', 'hybrid'}:
            masking_engine = 'yolo'

        if masking_engine not in {'sam3_cpp', 'yolo'}:
            return False

        image_format = str(self.config.get('stage2_format', 'png') or 'png').strip().lower()
        return image_format == 'png'

    def _mask_output_mode(self) -> str:
        mode = str(
            self.config.get('mask_output_mode')
            or self.config.get('sam3_output_mode')
            or ''
        ).strip().lower()
        if mode in {'masks_only', 'alpha_only', 'both'}:
            return mode
        if self.config.get('sam3_alpha_only', False):
            return 'alpha_only'
        if self.config.get('sam3_alpha_export', False):
            return 'both'
        return 'masks_only'

    def _pre_split_alpha_dir(self) -> Path:
        return Path(self.config['output_dir']) / 'alpha_cutouts'

    def _temporary_pre_split_alpha_dir(self) -> Path:
        return Path(self.config['output_dir']) / '_pre_split_alpha_cutouts'

    def _resolve_alpha_output_dir(self, output_root: Path, resolved_source: str) -> Path:
        return {
            'perspective': output_root / 'alpha_cutouts_perspective',
            'equirect': output_root / 'alpha_cutouts',
            'custom': output_root / 'alpha_cutouts_custom',
        }.get(resolved_source, output_root / 'alpha_cutouts_custom')

    def _prepare_pre_split_masking_config(self) -> dict:
        previous = {
            'stage3_input_dir': self.config.get('stage3_input_dir'),
            'stage3_image_source': self.config.get('stage3_image_source'),
            'mask_target': self.config.get('mask_target'),
            'sam3_alpha_export': self.config.get('sam3_alpha_export'),
            'sam3_alpha_only': self.config.get('sam3_alpha_only'),
            'stage2_input_dir': self.config.get('stage2_input_dir'),
        }

        self.config['stage3_input_dir'] = str(Path(self.config['output_dir']) / 'extracted_frames')
        self.config['stage3_image_source'] = 'equirect'
        self.config['mask_target'] = 'equirect'
        masking_engine = str(self.config.get('masking_engine', 'yolo') or 'yolo').strip().lower()
        if masking_engine == 'sam_vitb':
            masking_engine = 'sam3_cpp'
        elif masking_engine in {'yolo_onnx', 'yolo_pytorch', 'hybrid'}:
            masking_engine = 'yolo'

        if masking_engine == 'sam3_cpp':
            self.config['sam3_alpha_export'] = True
            self.config['sam3_alpha_only'] = True
        return previous

    def _materialize_alpha_cutouts_from_masks(
        self,
        input_dir: Path,
        masks_dir: Path,
        output_dir: Path,
    ) -> Dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        failed = 0
        images: list[Path] = []
        for pattern in self._image_patterns():
            images.extend(path for path in input_dir.rglob(pattern) if path.is_file())

        for image_path in sorted(images):
            rel_path = image_path.relative_to(input_dir)
            target_path = (output_dir / rel_path).with_suffix('.png')
            target_path.parent.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                failed += 1
                logger.warning('[Mask Alpha] Failed to load source image: %s', image_path)
                continue

            if image.ndim == 2:
                rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
            elif image.ndim == 3 and image.shape[2] == 4:
                rgba = image.copy()
            else:
                rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            mask_path = masks_dir / f'{image_path.stem}_mask.png'
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    failed += 1
                    logger.warning('[Mask Alpha] Failed to load mask image: %s', mask_path)
                    continue
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                if mask.shape[:2] != rgba.shape[:2]:
                    mask = cv2.resize(mask, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
            else:
                mask = np.full(rgba.shape[:2], 255, dtype=np.uint8)

            rgba[:, :, 3] = mask
            if cv2.imwrite(str(target_path), rgba):
                created += 1
            else:
                failed += 1
                logger.warning('[Mask Alpha] Failed to save alpha cutout: %s', target_path)

        return {
            'success': failed == 0 and (created > 0 or not images),
            'created': created,
            'failed': failed,
            'total': len(images),
            'output_dir': str(output_dir),
        }

    def _remove_generated_mask_files(self, masks_dir: Path) -> None:
        if not masks_dir.exists() or not masks_dir.is_dir():
            return
        for mask_path in masks_dir.rglob('*_mask.png'):
            try:
                mask_path.unlink()
            except Exception as exc:
                logger.warning('[Mask Alpha] Failed to remove mask file %s: %s', mask_path, exc)

    def _restore_config_values(self, previous: dict):
        for key, value in previous.items():
            if value is None:
                self.config.pop(key, None)
            else:
                self.config[key] = value

    def _resolve_stage2_input_dir(self) -> Optional[Path]:
        stage2_input = self.config.get('stage2_input_dir')
        if stage2_input:
            return Path(stage2_input)

        if self.config.get('enable_stage1', True):
            return Path(self.config['output_dir']) / 'extracted_frames'

        discovered = self.discover_stage_input_folder(2, self.config['output_dir'])
        if discovered:
            return discovered
        return None
    
    def run(self):
        """Execute the pipeline"""
        try:
            # Lazy test PyTorch/CUDA - only when pipeline actually runs
            _test_torch_cuda()
            logger.info(f"[Pipeline] PyTorch status: {logger_msg}")
            logger.info(f"[Pipeline] HAS_TORCH_TRANSFORM={HAS_TORCH_TRANSFORM}, HAS_TORCH_CUDA={HAS_TORCH_CUDA}")
            
            results = {
                'stage1': {},
                'stage2': {},
                'stage3': {},
                'stage4': {},
                'stage5': {},
                'success': False,
                'start_time': datetime.now().isoformat()
            }
            
            # Track which stages were executed
            stages_executed = []
            
            # Frame Extraction
            if self.config.get('enable_stage1', True):
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before Frame Extraction")
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return
                
                logger.info("=== Starting Frame Extraction ===")
                stage1_result = self._execute_stage1()
                results['stage1'] = stage1_result
                stages_executed.append(1)
                self.stage_complete.emit(1, stage1_result)
                
                if not stage1_result.get('success'):
                    self.error.emit(f"Frame Extraction failed: {stage1_result.get('error')}")
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return
            
            # Perspective Splitting (Skip if direct masking mode)
            skip_transform = self.config.get('skip_transform', False)
            mask_before_split = self._should_mask_before_split()
            stage3_already_executed = False
            pre_split_cleanup_dir: Optional[Path] = None

            previous_stage2_input_dir = self.config.get('stage2_input_dir')

            if mask_before_split:
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before pre-split masking")
                    results['stages_executed'] = stages_executed
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return

                logger.info("=== Starting Mask Generation (Pre-Split PNG Pipeline) ===")
                previous_config = self._prepare_pre_split_masking_config()
                try:
                    stage3_result = self._execute_stage3()
                finally:
                    self._restore_config_values(previous_config)

                results['stage3'] = stage3_result
                stages_executed.append(3)
                stage3_already_executed = True
                self.stage_complete.emit(3, stage3_result)

                if not stage3_result.get('success'):
                    self.error.emit(f"Masking failed: {stage3_result.get('error')}")
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return

                masking_engine = str(self.config.get('masking_engine', 'yolo') or 'yolo').strip().lower()
                if masking_engine == 'sam_vitb':
                    masking_engine = 'sam3_cpp'
                elif masking_engine in {'yolo_onnx', 'yolo_pytorch', 'hybrid'}:
                    masking_engine = 'yolo'

                alpha_dir = self._pre_split_alpha_dir()
                if masking_engine == 'yolo':
                    alpha_dir_text = stage3_result.get('alpha_dir')
                    if alpha_dir_text:
                        alpha_dir = Path(alpha_dir_text)
                    else:
                        alpha_dir = self._temporary_pre_split_alpha_dir()
                        alpha_result = self._materialize_alpha_cutouts_from_masks(
                            Path(stage3_result['input_dir']),
                            Path(stage3_result['masks_dir']),
                            alpha_dir,
                        )
                        if not alpha_result.get('success'):
                            self.error.emit('Pre-split masking did not produce alpha cutouts for Stage 2 input')
                            results['success'] = False
                            results['stages_executed'] = stages_executed
                            self.finished.emit(results)
                            return
                        pre_split_cleanup_dir = alpha_dir

                if not self._dir_has_images(alpha_dir, recursive=False):
                    self.error.emit('Pre-split masking did not produce alpha cutouts for Stage 2 input')
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return
                self.config['stage2_input_dir'] = str(alpha_dir)
            
            if skip_transform:
                logger.info("=== Perspective Split: SKIPPED (Direct Masking Mode) ===")
                logger.info("Extracted frames will be used directly for masking")
                # Pass extraction output to masking input
                results['stage2'] = {
                    'success': True,
                    'skipped': True,
                    'message': 'Transform skipped - using equirectangular/fisheye frames directly'
                }
                stages_executed.append(2)  # Mark as executed (but skipped)
                self.stage_complete.emit(2, results['stage2'])
            elif self.config.get('enable_stage2', True):
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before Perspective Split")
                    results['stages_executed'] = stages_executed
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return
                
                logger.info("=== Starting Perspective Splitting ===")
                stage2_result = self._execute_stage2()
                if mask_before_split:
                    if previous_stage2_input_dir is None:
                        self.config.pop('stage2_input_dir', None)
                    else:
                        self.config['stage2_input_dir'] = previous_stage2_input_dir
                results['stage2'] = stage2_result
                stages_executed.append(2)
                self.stage_complete.emit(2, stage2_result)

                if pre_split_cleanup_dir and pre_split_cleanup_dir.exists():
                    shutil.rmtree(pre_split_cleanup_dir, ignore_errors=True)

                # Clean up GPU memory after splitting to free VRAM for masking
                if HAS_TORCH_CUDA:
                    try:
                        import torch
                        # Delete any transformer objects
                        if hasattr(self, 'transformer'):
                            del self.transformer
                        # Force GPU memory cleanup
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("[GPU Cleanup] Released split GPU memory before masking")
                    except Exception as e:
                        logger.warning(f"[GPU Cleanup] Failed to clean GPU memory: {e}")

                if not stage2_result.get('success'):
                    self.error.emit(f"Perspective Split failed: {stage2_result.get('error')}")
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return
            
            # AI Masking
            if self.config.get('enable_stage3', True) and not stage3_already_executed:
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before Masking")
                    results['stages_executed'] = stages_executed
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return
                
                logger.info("=== Starting Mask Generation ===")
                stage3_result = self._execute_stage3()
                results['stage3'] = stage3_result
                stages_executed.append(3)
                self.stage_complete.emit(3, stage3_result)
                
                if not stage3_result.get('success'):
                    self.error.emit(f"Masking failed: {stage3_result.get('error')}")
                    results['success'] = False
                    results['stages_executed'] = stages_executed
                    self.finished.emit(results)
                    return
            
            # RealityCapture Export
            if self.config.get('export_realityscan', False):
                if self.is_cancelled:
                    logger.info("Pipeline cancelled before RealityScan export")
                    results['stages_executed'] = stages_executed
                    self.finished.emit({'success': False, 'error': 'Cancelled by user'})
                    return

                logger.info("=== Starting RealityCapture Export ===")
                export_result = self._execute_realityscan_export_only()
                results['realityscan_export'] = export_result
                self.stage_complete.emit(4, export_result)

                if not export_result.get('success'):
                    self.error.emit(f"RealityCapture export failed: {export_result.get('error')}")
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
        """Execute Frame Extraction"""
        try:
            input_file = self.config['input_file']
            
            output_dir = Path(self.config['output_dir']) / 'extracted_frames'
            
            fps = self.config.get('fps', DEFAULT_FPS)
            method = self.config.get('extraction_method', 'opencv')
            
            # Get additional parameters
            start_time = self.config.get('start_time', 0.0)
            end_time = self.config.get('end_time', None)
            
            def progress_callback(current, total, message):
                self.progress.emit(current, total, f"Extraction: {message}")
            
            # Map extraction method names
            # 'dual_fisheye' is raw extraction, so it should use opencv
            # 'ffmpeg' uses v360 stitching filter (RECOMMENDED)
            if method.lower() == 'dual_fisheye':
                logger.info("Dual-Fisheye Export selected - extracting raw fisheye frames")
                method = 'opencv'
            elif method.lower() == 'ffmpeg':
                method = 'ffmpeg_stitched'

            input_suffix = Path(input_file).suffix.lower()
            if input_suffix == '.mp4' and method.lower() in ['sdk', 'sdk_stitching']:
                logger.info("MP4 input detected - forcing FFmpeg extraction instead of SDK stitching")
                method = 'ffmpeg_stitched'

            if self._is_invalid_ffmpeg_stitched_request(method, input_file):
                return {
                    'success': False,
                    'error': self._build_insv_stitched_sdk_error(),
                    'frames': [],
                    'count': 0,
                }
            
            # Use SDK extractor if method is 'sdk' or 'sdk_stitching' (PRIMARY METHOD)
            if method.lower() in ['sdk', 'sdk_stitching']:
                logger.info("=== SDK Stitching (PRIMARY METHOD) ===")
                
                if not self.sdk_extractor.is_available():
                    logger.warning("WARNING: Insta360 MediaSDK not available")
                    fallback_method = self._resolve_stitched_ffmpeg_fallback_method(input_file)
                    if fallback_method is None:
                        return {
                            'success': False,
                            'error': self._build_insv_stitched_sdk_error(),
                            'frames': [],
                            'count': 0,
                        }
                    logger.warning("INFO: Falling back to FFmpeg stitched extraction for non-INSV input")
                    method = fallback_method
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
                        'original': (7680, 3840),  # Explicit 2:1 equirectangular (SDK ignores None)
                        '8k': (7680, 3840),
                        '6k': (6080, 3040),
                        '4k': (3840, 1920),
                        '2k': (1920, 960)
                    }
                    resolution_key = str(self.config.get('sdk_resolution', '8k')).lower()  # Default to 8k
                    resolution = resolution_map.get(resolution_key, (7680, 3840))  # Fallback to 8K
                    
                    try:
                        # Prefer direct sdk_options dict from UI (via get_current_config).
                        # Fall back to building from legacy individual prefixed keys if absent.
                        sdk_options = self.config.get('sdk_options', None)
                        logger.info(f"[SDK] sdk_options from config: {sdk_options}")
                        if sdk_options is None:
                            sdk_options = {}
                            if self.config.get('sdk_direction_lock', False):
                                sdk_options['enable_direction_lock'] = True
                            # Color correction overrides (all default to 0 = neutral)
                            _color_keys = [
                                'exposure', 'highlights', 'shadows', 'contrast',
                                'brightness', 'blackpoint', 'saturation', 'vibrance',
                                'warmth', 'tint', 'definition',
                            ]
                            for _k in _color_keys:
                                _val = self.config.get(f'sdk_{_k}', 0)
                                if _val != 0:
                                    sdk_options[_k] = int(_val)
                            sdk_options = sdk_options if sdk_options else None

                        use_sdk_native_color_corrections = bool(
                            self.config.get('sdk_use_native_color_corrections', True)
                        )
                        if sdk_options is None:
                            sdk_options = {}
                        sdk_options['_use_sdk_native_color_corrections'] = use_sdk_native_color_corrections

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
                            progress_callback=lambda p: progress_callback(p, 100, "SDK stitching"),
                            sdk_options=sdk_options
                        )
                        
                        logger.info(f"[OK] SDK extraction complete: {len(frame_paths)} frames")

                        # ── Post-process: apply OpenCV colour corrections ──────────────────
                        # Colour is applied ONLY via OpenCV (not SDK CLI) so the
                        # output exactly matches the live preview.
                        # Yaw is NOT applied here — the SDK with direction-lock
                        # already orients the panorama correctly.
                        if sdk_options:
                            _color_opts = {k: sdk_options[k] for k in (
                                'exposure','highlights','shadows','contrast','brightness',
                                'blackpoint','saturation','vibrance','warmth','tint','definition'
                            ) if sdk_options.get(k, 0) != 0}
                            if _color_opts and not use_sdk_native_color_corrections:
                                logger.info(f"[Color] Post-processing {len(frame_paths)} frames with OpenCV corrections: {_color_opts}")
                                for _index, _fp in enumerate(frame_paths, start=1):
                                    try:
                                        _img = cv2.imread(_fp)
                                        if _img is not None:
                                            _corrected = apply_color_corrections(_img, _color_opts)
                                            cv2.imwrite(_fp, _corrected)
                                            if _index == 1 or _index % 5 == 0 or _index == len(frame_paths):
                                                progress_callback(_index, len(frame_paths), f"Color post-process {_index}/{len(frame_paths)}")
                                    except Exception as _ce:
                                        logger.warning(f"[Color] Failed to correct {_fp}: {_ce}")
                            elif _color_opts:
                                logger.info("[Color] Native SDK color corrections already applied during extraction")

                        # ── Post-process: apply frame rotation if set in preview ──────
                        _frame_rotation = self.config.get('frame_rotation', 0)
                        if _frame_rotation:
                            self._apply_frame_rotation(frame_paths, _frame_rotation)

                        return {
                            'success': True,
                            'frames': frame_paths,
                            'method': 'sdk_stitching',
                            'count': len(frame_paths)
                        }
                    
                    except IncompleteSDKExtractionError as incomplete_error:
                        logger.warning(f"[WARNING] SDK extraction incomplete: {incomplete_error}")

                        if self._requires_sdk_for_stitched_extraction(input_file):
                            removed = self._purge_stage1_partial_outputs(output_dir)
                            if removed:
                                logger.warning(f"[WARNING] Removed {removed} partial SDK frame(s) after failed stitched INSV extraction")
                            return {
                                'success': False,
                                'error': self._build_insv_stitched_sdk_error(str(incomplete_error)),
                                'frames': [],
                                'count': 0,
                            }

                        if not self.config.get('allow_fallback', True):
                            logger.error("Fallback disabled by configuration. Aborting.")
                            raise

                        removed = self._purge_stage1_partial_outputs(output_dir)
                        if removed:
                            logger.warning(f"[WARNING] Removed {removed} partial SDK frame(s) before FFmpeg fallback")

                        fallback_method = self._resolve_stitched_ffmpeg_fallback_method(input_file)
                        if fallback_method is None:
                            return {
                                'success': False,
                                'error': self._build_insv_stitched_sdk_error(str(incomplete_error)),
                                'frames': [],
                                'count': 0,
                            }

                        logger.warning("INFO: Falling back to FFmpeg stitched extraction after incomplete SDK extraction")
                        method = fallback_method
                    
                    except Exception as sdk_error:
                        logger.error(f"[ERROR] SDK extraction failed: {sdk_error}")

                        if self._requires_sdk_for_stitched_extraction(input_file):
                            removed = self._purge_stage1_partial_outputs(output_dir)
                            if removed:
                                logger.warning(f"[WARNING] Removed {removed} partial SDK frame(s) after failed stitched INSV extraction")
                            return {
                                'success': False,
                                'error': self._build_insv_stitched_sdk_error(str(sdk_error)),
                                'frames': [],
                                'count': 0,
                            }
                        
                        # Check if fallback is allowed
                        if not self.config.get('allow_fallback', True):
                            logger.error("Fallback disabled by configuration. Aborting.")
                            raise sdk_error

                        removed = self._purge_stage1_partial_outputs(output_dir)
                        if removed:
                            logger.warning(f"[WARNING] Removed {removed} partial SDK frame(s) before FFmpeg fallback")

                        fallback_method = self._resolve_stitched_ffmpeg_fallback_method(input_file)
                        if fallback_method is None:
                            return {
                                'success': False,
                                'error': self._build_insv_stitched_sdk_error(str(sdk_error)),
                                'frames': [],
                                'count': 0,
                            }

                        logger.warning("INFO: Falling back to FFmpeg stitched extraction")
                        method = fallback_method
            
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

            # Apply frame rotation (fisheye preview rotation → extracted files)
            _frame_rotation = self.config.get('frame_rotation', 0)
            if _frame_rotation and result.get('success'):
                _all_files = result.get('output_files', result.get('frames', []))
                if not _all_files:
                    # Fallback: scan output_dir for image files
                    _img_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
                    _all_files = [str(p) for p in output_dir.rglob('*')
                                  if p.is_file() and p.suffix.lower() in _img_exts]
                self._apply_frame_rotation(_all_files, _frame_rotation)

            return result
        
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {'success': False, 'error': str(e)}

    def _requires_sdk_for_stitched_extraction(self, input_file: str) -> bool:
        return Path(input_file).suffix.lower() == '.insv'

    def _resolve_stitched_ffmpeg_fallback_method(self, input_file: str) -> Optional[str]:
        return 'ffmpeg_stitched' if Path(input_file).suffix.lower() == '.mp4' else None

    def _is_invalid_ffmpeg_stitched_request(self, method: str, input_file: str) -> bool:
        return method.lower() == 'ffmpeg_stitched' and self._requires_sdk_for_stitched_extraction(input_file)

    def _build_insv_stitched_sdk_error(self, reason: Optional[str] = None) -> str:
        base = (
            'Stitched .insv extraction requires Insta360 MediaSDK. '
            'FFmpeg stitched fallback is disabled for .insv because it produces incorrect stitched results. '
            'Use SDK Stitching for stitched frames, or choose FFmpeg dual-lens/lens-specific methods only for raw fisheye export.'
        )
        if reason:
            return f'{base} SDK detail: {reason}'
        return base

    def _purge_stage1_partial_outputs(self, output_dir: Path) -> int:
        """Remove partial extraction images so fallback methods start from a clean Stage 1 folder."""
        image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        removed = 0

        for path in output_dir.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in image_exts:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning(f"[WARNING] Failed to remove partial Stage 1 output {path}: {exc}")

        return removed
    
    def _apply_frame_rotation(self, frame_paths: list, rotation: int) -> None:
        """
        Rotate extracted image files in-place.

        Args:
            frame_paths: List of absolute file paths to rotate.
            rotation:    Degrees clockwise — must be 90, 180, or 270 (0 = no-op).
        """
        if not rotation or rotation % 90 != 0:
            return

        cv_codes = {
            90:  cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        code = cv_codes.get(rotation % 360)
        if code is None:
            return

        rotated = 0
        for fp in frame_paths:
            try:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                img = cv2.rotate(img, code)
                cv2.imwrite(str(fp), img)
                rotated += 1
            except Exception as _e:
                logger.warning(f"[Rotation] Could not rotate {fp}: {_e}")

        logger.info(f"[Rotation] Rotated {rotated}/{len(frame_paths)} frames by {rotation}° CW")

    def _execute_stage2(self) -> Dict:
        """Execute Perspective Splitting"""
        try:
            # Get input frames (auto-discovery runs ONLY ONCE)
            input_dir = self._resolve_stage2_input_dir()
            if input_dir is None:
                return {
                    'success': False,
                    'error': 'Split input directory not specified and auto-discovery failed',
                    'output_files': []
                }
            
            if not input_dir.exists():
                return {
                    'success': False,
                    'error': f'Input directory does not exist: {input_dir}',
                    'output_files': []
                }
            
            output_dir = Path(self.config['output_dir']) / 'perspective_views'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get transform settings
            transform_type = self.config.get('transform_type', 'perspective')
            output_width = self.config.get('output_width', DEFAULT_OUTPUT_WIDTH)
            output_height = self.config.get('output_height', DEFAULT_OUTPUT_HEIGHT)
            
            # Get all input frames (support many extensions, case-insensitive)
            image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            input_frames = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
            frame_records = build_stage2_frame_records(
                input_frames,
                normalize_stage2_numbering_mode(self.config.get('stage2_numbering_mode')),
            )
            total_frames = len(frame_records)
            if total_frames == 0:
                return {
                    'success': False,
                    'error': f'No input images found for Perspective Split in: {input_dir}',
                    'output_files': []
                }
            
            # Route based on transform type
            if transform_type == 'cubemap':
                logger.info(f"Processing {total_frames} frames in CUBEMAP mode")
                return self._execute_stage2_cubemap(frame_records, output_dir)
            else:
                # Perspective mode - use cameras
                camera_config = self.config.get('camera_config', {})
                cameras = camera_config.get('cameras', self._get_default_cameras())
                total_operations = total_frames * len(cameras)
                logger.info(f"Processing {total_frames} frames with {len(cameras)} cameras (PERSPECTIVE mode)")
                return self._execute_stage2_perspective(frame_records, cameras, output_dir, output_width, output_height)
        
        except Exception as e:
            logger.error(f"Perspective Split failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'output_files': []}
    
    def _execute_stage2_perspective(self, frame_records, cameras, output_dir, output_width, output_height) -> Dict:
        """Execute perspective splitting (E2P) with GPU/CPU optimization"""
        try:
            output_files = []
            total_frames = len(frame_records)
            total_operations = total_frames * len(cameras)
            image_format = self.config.get('stage2_format', 'png')
            layout_mode = normalize_stage2_layout_mode(self.config.get('stage2_perspective_layout'))
            processing_backend = 'cpu'
            
            # Check for GPU acceleration with comprehensive compatibility testing
            use_gpu = False
            if HAS_TORCH_TRANSFORM and HAS_TORCH_CUDA:
                logger.info("[Split GPU] Starting GPU compatibility check...")
                try:

                    # Get GPU info
                    device_name = torch.cuda.get_device_name(0)
                    try:
                        compute_capability = torch.cuda.get_device_capability(0)
                        compute_cap_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
                    except Exception:
                        compute_cap_str = "unknown"

                    logger.info(f"[Split GPU] Testing GPU: {device_name} ({compute_cap_str})")

                    # Test GPU with actual kernel execution (not just memory allocation)
                    # This catches RTX 50-series "no kernel image available" errors early
                    test_tensor = torch.zeros(10, 10, device='cuda')
                    test_result = test_tensor + 1  # Force CUDA kernel execution
                    test_mean = test_result.mean().item()  # Force device synchronization
                    del test_tensor, test_result
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    use_gpu = True
                    logger.info(f"[Split GPU] GPU acceleration ENABLED: {device_name} ({compute_cap_str})")
                except Exception as e:
                    error_msg = str(e).lower()
                    logger.warning(f"[Split GPU] GPU test failed: {e}. Falling back to CPU.")
                    if "no kernel image" in error_msg or "cuda capability" in error_msg or "sm_" in error_msg:
                        logger.error(f"[Split GPU] GPU incompatibility detected")
                        try:
                            logger.error(f"[Split GPU]    GPU: {device_name} ({compute_cap_str})")
                        except:
                            pass
                        logger.error(f"[Split GPU]    PyTorch CUDA kernels not available for this GPU architecture")
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
            else:
                logger.warning("[Split GPU] GPU transform not available (HAS_TORCH_TRANSFORM or HAS_TORCH_CUDA is False)")
            
            # GPU Execution Block
            if use_gpu:
                logger.info("[Split GPU] Starting GPU-accelerated processing...")
                try:
                    from concurrent.futures import ThreadPoolExecutor
                    import queue
                    import threading

                    # GPU Path: Optimized batch processing with async I/O
                    logger.info("[Split GPU] Initializing TorchE2PTransform...")
                    transformer = TorchE2PTransform()
                    logger.info(f"[Split GPU] Transformer initialized on device: {transformer.device}")
                    
                    # Get sample image dimensions for batch size calculation
                    sample_img = cv2.imread(str(frame_records[0][0]), cv2.IMREAD_UNCHANGED)
                    if sample_img is None:
                        raise RuntimeError("Failed to load sample image for batch size calculation")
                    
                    input_height, input_width = sample_img.shape[:2]
                    input_n_channels = sample_img.shape[2] if sample_img.ndim == 3 else 1
                    input_has_alpha = (input_n_channels == 4)
                    
                    # Pass num_cameras to batch size calculation for accurate VRAM estimation
                    num_cameras = len(cameras)
                    auto_batch_size = transformer.get_optimal_batch_size(
                        input_height, input_width, output_height, output_width, num_cameras
                    )
                    
                    # Keep GPU batch size at or below auto-detected safe value.
                    # For large 8K frames and many cameras, forcing larger batches can trigger OOM.
                    user_batch_size = int(self.config.get('split_gpu_batch_size', 0) or 0)
                    if user_batch_size > 0:
                        batch_size = max(1, min(user_batch_size, auto_batch_size))
                        logger.info(
                            f"Using batch size: {batch_size} frames (auto: {auto_batch_size}, user requested: {user_batch_size})"
                        )
                    else:
                        batch_size = max(1, auto_batch_size)
                        logger.info(f"Using batch size: {batch_size} frames (auto-safe)")
                    
                    del sample_img  # Free memory
                    
                    # Use JPEG for faster saving (10x faster than PNG)
                    ext = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'jpg'
                    if input_has_alpha and ext != 'png':
                        logger.info("[Split GPU] Input images have alpha channel — forcing PNG output to preserve transparency")
                        ext = 'png'
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
                        img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
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
                        batch_records = frame_records[batch_start:batch_end]
                        
                        # If first batch or no prefetch, submit load now
                        if pending_load_futures is None:
                            load_futures = {
                                load_executor.submit(load_image, path): (output_frame_id, path)
                                for path, output_frame_id in batch_records
                            }
                        else:
                            # Use prefetched futures
                            load_futures = pending_load_futures
                        
                        # PREFETCH: Start loading NEXT batch while we process current batch
                        next_batch_start = batch_end
                        if next_batch_start < total_frames:
                            next_batch_end = min(next_batch_start + batch_size, total_frames)
                            next_batch_records = frame_records[next_batch_start:next_batch_end]
                            pending_load_futures = {
                                load_executor.submit(load_image, path): (output_frame_id, path)
                                for path, output_frame_id in next_batch_records
                            }
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
                            out_path = resolve_perspective_output_path(output_dir, frame_idx, cam_idx, ext, layout_mode)
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            future = save_executor.submit(save_image_async, out_img.copy(), out_path, ext, yaw, pitch, roll, fov)
                            save_futures.append(future)
                        
                        del all_outputs
                        
                        # Progress update
                        self.progress.emit(batch_end * len(cameras), total_operations, 
                            f"Split (GPU): Batch {batch_idx + 1}/{total_batches}")
                    
                    # Wait for all saves to complete
                    logger.info("Waiting for async saves to complete...")
                    for future in as_completed(save_futures):
                        result = future.result()
                        if result:
                            output_files.append(result)
                    
                    save_executor.shutdown(wait=True)
                    load_executor.shutdown(wait=True)
                    output_files = sorted(output_files, key=perspective_output_sort_key)
                    
                    # Completed successfully on GPU
                    logger.info(f"[OK] GPU batch processing complete: {len(output_files)} perspectives created")
                    processing_backend = 'gpu'
                    return {
                        'success': True,
                        'perspective_count': len(output_files),
                        'output_files': output_files,
                        'output_dir': str(output_dir),
                        'processing_backend': processing_backend,
                    }

                except RuntimeError as e:
                    if "no kernel image" in str(e) or "CUDA error" in str(e):
                        logger.warning(f"GPU Perspective Split failed (RTX 50-series incompatibility): {e}")
                        logger.warning("Falling back to CPU Multiprocessing...")
                        use_gpu = False # Fall through to CPU block
                        output_files = [] # Reset output files
                        try:
                            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                                import gc
                                gc.collect()
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                try:
                                    torch.cuda.ipc_collect()
                                except Exception:
                                    pass
                        except:
                            pass
                    else:
                        raise e # Re-raise other errors

            # CPU Execution Block (Fallback or Default)
            if not use_gpu:
                # Try OpenCL (cv2.UMat) before falling back to plain CPU multiprocessing.
                # OpenCL is available on any GPU via the driver, requires no special install.
                _opencl_ok = cv2.ocl.haveOpenCL()
                if _opencl_ok:
                    logger.info("[Split OpenCL] OpenCL available — using cv2.UMat GPU-accelerated remap")
                    try:
                        from concurrent.futures import ThreadPoolExecutor

                        cv2.ocl.setUseOpenCL(True)
                        opencl_transformer = OpenCLE2PTransform()
                        _opencl_failed = False
                        save_executor = ThreadPoolExecutor(max_workers=12)

                        def save_image_async(out_img, out_path, ext, yaw, pitch, roll, fov):
                            try:
                                if ext == 'png':
                                    ok = cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                                elif ext in ('jpg', 'jpeg'):
                                    ok = cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                else:
                                    ok = cv2.imwrite(str(out_path), out_img)

                                if not ok:
                                    return None

                                try:
                                    self.metadata_handler.embed_camera_orientation(
                                        str(out_path), yaw, pitch, roll, fov
                                    )
                                except Exception:
                                    pass
                                return str(out_path)
                            except Exception as save_error:
                                logger.warning("[Split OpenCL] Failed to save %s: %s", out_path, save_error)
                                return None

                        # Check whether input images have alpha channel
                        _sample_ocl = cv2.imread(str(frame_records[0][0]), cv2.IMREAD_UNCHANGED)
                        _ocl_has_alpha = (_sample_ocl is not None and
                                          _sample_ocl.ndim == 3 and _sample_ocl.shape[2] == 4)
                        del _sample_ocl
                        ext = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'jpg'
                        if _ocl_has_alpha and ext != 'png':
                            logger.info("[Split OpenCL] Alpha channel detected — forcing PNG output")
                            ext = 'png'

                        total_ops_done = 0
                        for frame_position, (frame_path, output_frame_id) in enumerate(frame_records):
                            if self.is_cancelled:
                                return {'success': False, 'error': 'Cancelled by user'}
                            while self.is_paused:
                                if self.is_cancelled:
                                    return {'success': False, 'error': 'Cancelled by user'}
                                self.msleep(100)

                            equirect_img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                            if equirect_img is None:
                                logger.warning("[Split OpenCL] Failed to load %s", frame_path)
                                continue

                            frame_save_futures = []
                            for cam_idx, camera in enumerate(cameras):
                                yaw  = camera['yaw']
                                pitch = camera.get('pitch', 0)
                                roll  = camera.get('roll', 0)
                                fov   = camera.get('fov', DEFAULT_H_FOV)

                                out_img = opencl_transformer.equirect_to_pinhole(
                                    equirect_img, yaw, pitch, roll, fov, None,
                                    output_width, output_height
                                )

                                out_path = resolve_perspective_output_path(output_dir, output_frame_id, cam_idx, ext, layout_mode)
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                frame_save_futures.append(
                                    save_executor.submit(
                                        save_image_async,
                                        out_img.copy(),
                                        out_path,
                                        ext,
                                        yaw,
                                        pitch,
                                        roll,
                                        fov,
                                    )
                                )

                                total_ops_done += 1

                            for future in as_completed(frame_save_futures):
                                saved_path = future.result()
                                if saved_path:
                                    output_files.append(saved_path)
                            self.progress.emit(total_ops_done, total_operations,
                                               f"Split (OpenCL): frame {frame_position + 1}/{total_frames}")

                        save_executor.shutdown(wait=True)
                        logger.info("[Split OpenCL] Done — %d perspective views", len(output_files))
                        processing_backend = 'opencl'

                    except Exception as _ocl_err:
                        logger.warning("[Split OpenCL] Failed: %s — falling back to CPU", _ocl_err)
                        output_files = []  # Reset; CPU block below re-does everything
                        _opencl_failed = True
                else:
                    _opencl_failed = True

                if not _opencl_ok or _opencl_failed:
                    # CPU Path: Multiprocessing
                    logger.info("[Split CPU] Using CPU multiprocessing (GPU not available or failed)")
                    # Limit workers to avoid OOM with 8K images. 
                    # 8K image ~100MB. 20 workers = 2GB. Safe.
                    max_workers = min(os.cpu_count(), 12)
                    logger.info(f"[Split CPU] Using {max_workers} CPU workers")
                    
                    # Prepare tasks
                    tasks = []
                    for path, output_frame_id in frame_records:
                        tasks.append((
                            str(path), output_frame_id, cameras, str(output_dir),
                            output_width, output_height, image_format, layout_mode
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
                                    f"Split (CPU): {completed_ops}/{total_operations} views")
                            else:
                                logger.error(f"Frame processing failed: {result.get('error')}")

            # Clean up GPU resources at end of splitting
            if use_gpu and HAS_TORCH_CUDA:
                try:
                    if 'transformer' in locals():
                        del transformer
                    torch.cuda.empty_cache()
                    logger.info("[GPU Cleanup] Released GPU memory at end of splitting")
                except Exception as e:
                    logger.warning(f"[GPU Cleanup] Failed: {e}")

            output_files = sorted(output_files, key=perspective_output_sort_key)

            return {
                'success': True,
                'perspective_count': len(output_files),
                'output_files': output_files,
                'output_dir': str(output_dir),
                'processing_backend': processing_backend,
            }

        except Exception as e:
            logger.error(f"Perspective split error: {e}")
            # Clean up GPU on error
            if HAS_TORCH_CUDA:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            return {'success': False, 'error': str(e)}
    
    def _execute_stage2_cubemap(self, frame_records, output_dir) -> Dict:
        """Execute cubemap splitting (E2C)"""
        try:
            output_files = []
            processing_backend = 'cpu'
            
            # Get cubemap configuration
            cubemap_params = self.config.get('cubemap_params', {})
            cubemap_type = cubemap_params.get('cubemap_type', '6-face')
            tile_width = cubemap_params.get('tile_width', 2048)
            tile_height = cubemap_params.get('tile_height', 2048)
            fov = cubemap_params.get('fov', 90)
            layout_mode = normalize_stage2_layout_mode(self.config.get('stage2_cubemap_layout'))

            logger.info(f"Cubemap mode: {cubemap_type}, tile_size={tile_width}×{tile_height}, fov={fov}°")
            logger.info("[Split Cubemap] E2C currently runs on the CPU transform path")

            # Setup tile positions for 8-tile mode (4×2 grid)
            if cubemap_type == '8-tile':
                # 8 tiles in 4×2 grid: 4 columns × 2 rows
                # Each tile covers 90° horizontally (360° / 4 = 90°)
                # Overlap is achieved through pixel-based width (tile_width controls this)
                tile_positions = []
                for row in range(2):
                    pitch_val = 0 if row == 0 else 30  # Top row at horizon, bottom row slightly down
                    for col in range(4):
                        yaw_val = col * 90  # 0°, 90°, 180°, 270°
                        tile_positions.append({
                            'yaw': yaw_val,
                            'pitch': pitch_val,
                            'fov': fov,  # Fixed 90° FOV
                            'name': f'tile_{row}_{col}'
                        })
            
            total_operations = len(frame_records)
            
            for frame_idx, (frame_path, output_frame_id) in enumerate(frame_records):
                if self.is_cancelled:
                    return {'success': False, 'error': 'Cancelled by user'}
                
                # Load equirectangular image (UNCHANGED to preserve alpha channel if present)
                equirect_img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                
                if equirect_img is None:
                    logger.warning(f"Failed to load {frame_path}")
                    continue

                frame_has_alpha = (equirect_img.ndim == 3 and equirect_img.shape[2] == 4)
                
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
                            output_width=tile_width,
                            output_height=tile_height
                        )
                        
                        # Save face with configured format - use PIL for PNG, cv2 for JPEG
                        image_format = self.config.get('stage2_format', 'png')
                        extension = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'png'
                        if frame_has_alpha and extension != 'png':
                            extension = 'png'  # JPEG cannot carry alpha
                        output_path = resolve_cubemap_output_path(
                            output_dir,
                            output_frame_id,
                            face_config['name'],
                            extension,
                            layout_mode,
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_filename = output_path.name
                        
                        # Save image - use PIL for PNG (prevents corruption), cv2 for JPEG
                        success = False
                        if extension == 'png':
                            try:
                                if frame_has_alpha:
                                    face_rgba = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGBA)
                                    pil_img = Image.fromarray(face_rgba, 'RGBA')
                                else:
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
                    # Generate 8 tiles with pixel-based overlap (controlled by tile_width)
                    for tile in tile_positions:
                        tile_img = self.e2p_transform.equirect_to_pinhole(
                            equirect_img,
                            yaw=tile['yaw'],
                            pitch=tile['pitch'],
                            roll=0,
                            h_fov=tile['fov'],  # Fixed 90° FOV
                            output_width=tile_width,
                            output_height=tile_height
                        )
                        
                        # Save tile with configured format - use PIL for PNG, cv2 for JPEG
                        image_format = self.config.get('stage2_format', 'png')
                        extension = image_format if image_format in ['png', 'jpg', 'jpeg'] else 'png'
                        if frame_has_alpha and extension != 'png':
                            extension = 'png'  # JPEG cannot carry alpha
                        output_path = resolve_cubemap_output_path(
                            output_dir,
                            output_frame_id,
                            tile['name'],
                            extension,
                            layout_mode,
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_filename = output_path.name
                        
                        # Save image - use PIL for PNG (prevents corruption), cv2 for JPEG
                        success = False
                        if extension == 'png':
                            try:
                                if frame_has_alpha:
                                    tile_rgba = cv2.cvtColor(tile_img, cv2.COLOR_BGRA2RGBA)
                                    pil_img = Image.fromarray(tile_rgba, 'RGBA')
                                else:
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
                    f"Split CUBEMAP: Frame {frame_idx+1}/{len(frame_records)} ({cubemap_type})"
                )
            
            faces_per_frame = 6 if cubemap_type == '6-face' else 8
            logger.info(f"Generated {len(output_files)} cubemap faces ({faces_per_frame} per frame)")
            
            output_files = sorted(output_files, key=perspective_output_sort_key)

            return {
                'success': True,
                'cubemap_count': len(output_files),
                'output_files': output_files,
                'output_dir': str(output_dir),
                'cubemap_type': cubemap_type,
                'processing_backend': processing_backend,
            }
        
        except Exception as e:
            logger.error(f"Split CUBEMAP error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _execute_stage3(self) -> Dict:
        """Execute AI Masking"""
        try:
            if self.config.get('use_gpu', True):
                try:
                    import gc
                    import torch
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                        logger.info("[Stage 3] Cleared CUDA cache before masker initialization")
                except Exception:
                    pass

            output_root = Path(self.config['output_dir'])
            requested_source = self.config.get('stage3_image_source', 'auto')
            # Determine input directory based on mask_target setting
            mask_target = self.config.get('mask_target', 'split')
            skip_transform = self.config.get('skip_transform', False)
            stage3_input = self.config.get('stage3_input_dir')

            if stage3_input:
                input_dir = Path(stage3_input)
                resolved_source = self._detect_project_image_source(output_root, input_dir)
                logger.info(f"[Masking] Using specified input: {input_dir}")
            else:
                if requested_source == 'equirect' or (requested_source == 'auto' and (mask_target == 'equirect' or skip_transform)):
                    input_dir = output_root / 'extracted_frames'
                    resolved_source = 'equirect'
                    logger.info("[Equirect Masking] Masking extracted equirectangular frames")
                elif requested_source == 'perspective':
                    input_dir = output_root / 'perspective_views'
                    resolved_source = 'perspective'
                    logger.info(f"[Split Masking] Masking perspective views: {input_dir}")
                else:
                    input_dir = self._resolve_project_image_dir(output_root, 'auto')
                    resolved_source = self._detect_project_image_source(output_root, input_dir)
                    logger.info(f"[Masking] Auto-selected input: {input_dir}")

            if not input_dir or not input_dir.exists():
                return {
                    'success': False,
                    'error': 'Masking input directory not found',
                    'masks_created': 0,
                    'skipped': 0,
                    'failed': 0
                }

            masking_engine = str(self.config.get('masking_engine', 'yolo')).strip().lower()
            if masking_engine == 'sam_vitb':
                masking_engine = 'sam3_cpp'
            elif masking_engine in {'yolo_onnx', 'yolo_pytorch', 'hybrid'}:
                masking_engine = 'yolo'
            
            # Initialize masker if not done
            if self.masker is None:
                model_size = self.config.get('model_size', 'small')
                confidence = self.config.get('confidence_threshold', 0.5)
                use_gpu = self.config.get('use_gpu', True)
                
                # Handle SAM3 separately
                if masking_engine == 'sam3_cpp':
                    try:
                        logger.info("Initializing SAM3.cpp external masker...")
                        from ..masking.sam3_external_masker import SAM3ExternalMasker

                        self.masker = SAM3ExternalMasker(
                            segment_persons_exe=self.config.get('sam3_segmenter_path', ''),
                            model_path=self.config.get('sam3_model_path', ''),
                            sam3_image_exe=self.config.get('sam3_image_exe_path') or None,
                            use_gpu=use_gpu,
                            feather_radius=self.config.get('sam3_feather_radius', 8),
                            morph_radius=self.config.get('sam3_morph_radius', 0),
                            alpha_export=self.config.get('sam3_alpha_export', False),
                            alpha_only=self.config.get('sam3_alpha_only', False),
                            max_input_width=self.config.get('sam3_max_input_width', 3840),
                            score_threshold=self.config.get('sam3_score_threshold', 0.5),
                            nms_threshold=self.config.get('sam3_nms_threshold', 0.1),
                            enable_refinement=self.config.get('sam3_enable_refinement', True),
                            refine_sky_only=self.config.get('sam3_refine_sky_only', True),
                            seam_aware_refinement=self.config.get('sam3_seam_aware_refinement', True),
                            edge_sharpen_strength=self.config.get('sam3_edge_sharpen_strength', 0.75),
                        )
                        logger.info("SAM3.cpp masker initialized successfully")
                        self.masker.set_enabled_categories(
                            self.config.get('sam3_prompts', {})
                        )
                        self.masker.set_custom_prompts(
                            self.config.get('sam3_custom_prompts', '')
                        )
                        use_onnx = False
                    except Exception as e:
                        logger.error(f"Failed to initialize SAM3.cpp masker: {e}", exc_info=True)
                        return {
                            'success': False,
                            'error': f'SAM3.cpp initialization failed: {e}',
                            'masks_created': 0,
                            'skipped': 0,
                            'failed': 0
                        }

                else:
                    # Single YOLO mode with ONNX-first runtime and PyTorch fallback
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

                    custom_onnx_model = str(self.config.get('yolo_model_path', '') or '').strip()
                    if custom_onnx_model:
                        onnx_path = Path(custom_onnx_model).expanduser()
                        onnx_model_name = onnx_path.name
                        logger.info(f"Using custom YOLO ONNX model: {onnx_path}")
                    else:
                        # Try YOLO26 first, then YOLOv8
                        onnx_model_name = model_map_yolo26.get(model_size, 'yolo26s-seg.onnx')
                        onnx_path = resolve_masking_model_path(onnx_model_name)

                        if not onnx_path.exists():
                            # Fallback to YOLOv8
                            onnx_model_name = model_map_yolov8.get(model_size, 'yolov8s-seg.onnx')
                            onnx_path = resolve_masking_model_path(onnx_model_name)
                            logger.info(f"YOLO26 not found, falling back to YOLOv8: {onnx_model_name}")
                        else:
                            logger.info(f"Using YOLO26 model (3-4x faster): {onnx_model_name}")
                    
                    use_onnx = False
                    onnx_error = "Unknown error"
                    
                    if onnx_path.exists():
                        try:
                            from ..masking.onnx_masker import ONNXMasker
                            logger.info(f"Found ONNX model {onnx_path}, initializing ONNXMasker")
                            self.masker = ONNXMasker(
                                model_path=str(onnx_path),
                                confidence_threshold=confidence,
                                use_gpu=use_gpu,
                                mask_dilation_pixels=15  # Expand boundaries by 15 pixels (includes backpack!)
                            )
                            backend_name = getattr(self.masker, 'backend', 'onnxruntime')
                            logger.info(f"Masking backend ready via ONNXMasker ({backend_name})")
                            use_onnx = True
                        except Exception as e:
                            onnx_error = f"InitError: {e}"
                            logger.warning(f"Failed to initialize ONNXMasker: {e}. Attempting PyTorch fallback.")
                    else:
                        onnx_error = f"File not found at {onnx_path}"
                        logger.warning(f"ONNX model file not found at: {onnx_path}")

                    if not use_onnx:
                        logger.info(f"ONNX initialization failed ({onnx_error}). Attempting PyTorch fallback.")

                        try:
                            from ..utils.runtime_backends import has_usable_torch_runtime
                            if not has_usable_torch_runtime():
                                raise ImportError("PyTorch runtime not bundled in ONNX-only package")
                            from ..masking.multi_category_masker import MultiCategoryMasker
                            
                            self.masker = MultiCategoryMasker(
                                model_size=model_size,
                                confidence_threshold=confidence,
                                use_gpu=use_gpu
                            )
                        except Exception as e:
                            # If we are here, it means we failed to use ONNX AND failed to use PyTorch
                            error_msg = (
                                f"Masking Initialization Failed: Could not load masking engine.\n"
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
                    
                    # Set enabled categories (YOLO only - SAM/Hybrid already set above)
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
            # NOTE: if stage3_input_dir was explicitly provided, the first resolution block
            # already set input_dir correctly — do not overwrite it here.
            skip_transform = self.config.get('skip_transform', False)

            if not self.config.get('stage3_input_dir'):
                if skip_transform:
                    # Direct Masking Mode - use extracted frames (equirectangular/fisheye)
                    input_dir = Path(self.config['output_dir']) / 'extracted_frames'
                    logger.info(f"[Direct Masking] Using extracted frames directly: {input_dir}")
                elif self.config.get('enable_stage2', True):
                    # Split was enabled - use its output (perspective images)
                    input_dir = Path(self.config['output_dir']) / 'perspective_views'
                    logger.info(f"Using perspective views: {input_dir}")
                else:
                    # Split disabled - check for explicit input or auto-discover ONCE
                    stage3_input = self.config.get('stage3_input_dir')
                    if not stage3_input:
                        # Single auto-discovery attempt
                        discovered = self.discover_stage_input_folder(3, self.config['output_dir'])
                        if discovered:
                            input_dir = discovered
                            logger.info(f"Auto-discovered masking input: {input_dir}")
                        else:
                            return {
                                'success': False,
                                'error': 'Masking input directory not specified and auto-discovery failed',
                                'masks_created': 0,
                                'skipped': 0,
                                'failed': 0
                            }
                    else:
                        input_dir = Path(stage3_input)
                        logger.info(f"Using specified masking input: {input_dir}")
                # Re-detect resolved_source from the (possibly updated) input_dir
                resolved_source = self._detect_project_image_source(output_root, input_dir)
            
            # alpha_only mode uses a dedicated subfolder so no _mask.png files appear
            # alongside the source images when the user opens the output folder.
            if self.config.get('sam3_alpha_only', False):
                masks_subdir = 'alpha_cutouts'
            else:
                masks_subdir = {
                    'perspective': 'masks_perspective',
                    'equirect': 'masks_equirect',
                    'custom': 'masks_custom',
                }.get(resolved_source, 'masks_custom')
            output_dir = output_root / masks_subdir
            save_visualization = self.config.get('save_visualization', False)
            
            def progress_callback(current, total, message):
                self.progress.emit(current, total, f"Masking: {message}")
            
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

            output_mode = self._mask_output_mode()
            if masking_engine == 'yolo' and output_mode in {'alpha_only', 'both'}:
                alpha_dir = self._resolve_alpha_output_dir(output_root, resolved_source)
                alpha_result = self._materialize_alpha_cutouts_from_masks(input_dir, output_dir, alpha_dir)
                if not alpha_result.get('success'):
                    return {
                        'success': False,
                        'error': 'Failed to generate alpha cutout PNGs from YOLO masks',
                        'masks_created': int(result.get('successful', 0)),
                        'skipped': int(result.get('skipped', 0)),
                        'failed': int(result.get('failed', 0)),
                    }
                result['alpha_dir'] = str(alpha_dir)
                result['alpha_created'] = alpha_result.get('created', 0)
                if output_mode == 'alpha_only':
                    self._remove_generated_mask_files(output_dir)

            result['input_dir'] = str(input_dir)
            result['mask_source'] = resolved_source
            result['masks_dir'] = str(output_dir)
            result['masks_created'] = int(result.get('successful', 0))
            result['success'] = int(result.get('failed', 0)) == 0
            if result['success']:
                if int(result.get('masks_created', 0)) == 0 and int(result.get('total', 0)) > 0:
                    logger.warning(
                        "Mask generation completed without creating masks: %s image(s) processed, %s skipped",
                        result.get('total', 0),
                        result.get('skipped', 0),
                    )
            else:
                result['error'] = (
                    f"Mask generation failed for {result.get('failed', 0)} image(s) "
                    f"out of {result.get('total', 0)}"
                )
            return result
        
        except Exception as e:
            logger.error(f"Masking error: {e}")
            return {'success': False, 'error': str(e), 'masks_created': 0, 'skipped': 0, 'failed': 0}

    def _execute_realityscan_export_only(self) -> Dict:
        """Export split/masked images for RealityCapture without COLMAP."""
        import shutil as _shutil
        try:
            output_root = Path(self.config['output_dir'])
            realityscan_dir = output_root / 'realityscan_export'
            alignment_mode = self.config.get('alignment_mode', 'perspective_reconstruction')

            # Find images directory
            images_dir = self._resolve_project_image_dir(output_root, 'auto', alignment_mode)
            if not images_dir:
                discovered = self.discover_stage_input_folder(3, self.config['output_dir'])
                if discovered:
                    images_dir = discovered
                else:
                    return {
                        'success': False,
                        'error': 'RealityCapture export requires images. Run Stage 1 or 2 first.',
                    }

            # Find masks directory
            export_masks_dir = None
            if self.config.get('export_include_masks', True):
                image_source = self._detect_project_image_source(output_root, images_dir)
                mask_source = self.config.get('export_mask_source', 'auto')
                images_have_alpha = self._images_have_alpha_channel(images_dir)
                if images_have_alpha and mask_source in {'auto', 'match_images', 'match_reconstruction'}:
                    logger.info('[Export] Selected images already contain alpha channel; skipping automatic separate-mask export')
                else:
                    export_masks_dir = self._resolve_project_masks_dir(
                        output_root,
                        mask_source,
                        image_source,
                    )

            # Create output structure
            out_images = realityscan_dir / 'images'
            out_images.mkdir(parents=True, exist_ok=True)

            # Copy images
            img_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
            copied = 0
            for p in sorted(images_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in img_exts:
                    _shutil.copy2(str(p), str(out_images / p.name))
                    copied += 1

            # Copy masks if available
            mask_count = 0
            if export_masks_dir and export_masks_dir.exists():
                out_masks = realityscan_dir / 'masks'
                out_masks.mkdir(parents=True, exist_ok=True)
                for p in sorted(export_masks_dir.rglob('*')):
                    if p.is_file() and p.suffix.lower() in img_exts:
                        _shutil.copy2(str(p), str(out_masks / p.name))
                        mask_count += 1

            # Write XMP sidecars using orientation from EXIF (yaw/pitch/roll embedded by Stage 2)
            try:
                from .export_formats import export_xmp_from_exif
                export_xmp_from_exif(out_images)
            except Exception as xmp_err:
                logger.warning(f"[Export] XMP sidecar generation skipped: {xmp_err}")

            logger.info(f"[Export] RealityCapture export: {copied} images, {mask_count} masks -> {realityscan_dir}")
            return {
                'success': True,
                'realityscan_export': str(realityscan_dir),
                'images_exported': copied,
                'masks_exported': mask_count,
                'image_count': copied,
                'mask_count': mask_count,
                'mode': 'images_masks_only',
            }

        except Exception as e:
            logger.error(f"RealityCapture export error: {e}", exc_info=True)
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
        
        # Cancel and clean up any existing worker before starting a new one.
        # Without this guard, rapid successive calls (e.g. from the auto-advance loop)
        # would spawn multiple concurrent worker threads while pause/cancel signals
        # only reached the latest worker, leaving orphaned threads running.
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)  # Wait up to 3 s for clean shutdown

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
