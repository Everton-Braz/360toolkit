"""
Multi-Category Masking Module using YOLOv8 Instance Segmentation
Generates binary masks for multiple object categories (persons, personal objects, animals)
suitable for RealityScan and photogrammetry workflows.

Extended from 360toFrame PersonMasker with GPU acceleration and batch processing.

Mask Format (RealityScan Compatible):
- 0 (Black) = Mask/Remove this region (objects to remove)
- 255 (White) = Keep this region (background/valid points)
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict
import sys

# Defer torch/ultralytics imports to avoid PyInstaller analysis issues
# These will be imported at runtime when actually needed
TORCH_AVAILABLE = False
YOLO_AVAILABLE = False
torch = None
YOLO = None

# Only try importing if NOT running under PyInstaller analysis
# PyInstaller sets sys.frozen during build analysis, but we check for _MEIPASS at runtime
if not getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS'):
    # Either running in normal Python OR running from frozen exe
    try:
        import torch
        TORCH_AVAILABLE = True
        logging.info(f"PyTorch loaded successfully. Version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logging.warning(f"PyTorch not available. GPU masking disabled. Error: {e}")
        torch = None
    except Exception as e:
        logging.error(f"PyTorch import failed with unexpected error: {e}", exc_info=True)
        torch = None

    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
        logging.info("Ultralytics loaded successfully")
    except ImportError as e:
        YOLO_AVAILABLE = False
        logging.warning(f"Ultralytics not available. Masking will be disabled. Error: {e}")
    except Exception as e:
        YOLO_AVAILABLE = False
        logging.error(f"Ultralytics import failed with unexpected error: {e}", exc_info=True)
else:
    # Running under PyInstaller analysis - skip imports to avoid build-time errors
    logging.debug("Skipping torch/ultralytics imports during PyInstaller analysis")

from ..config.defaults import (
    YOLOV8_MODELS, DEFAULT_MODEL_SIZE, DEFAULT_CONFIDENCE_THRESHOLD,
    MASKING_CATEGORIES, MASK_VALUE_REMOVE, MASK_VALUE_KEEP,
    DEFAULT_USE_GPU, DEFAULT_BATCH_SIZE
)

logger = logging.getLogger(__name__)


class MultiCategoryMasker:
    """
    YOLOv8-based multi-category masking for photogrammetry workflows.
    Supports persons, personal objects, and animals detection.
    Includes GPU acceleration and batch processing.
    """
    
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, 
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 use_gpu: bool = DEFAULT_USE_GPU,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initialize MultiCategoryMasker with YOLOv8 model.
        
        Args:
            model_size: Model size ('nano', 'small', 'medium', 'large', 'xlarge')
            confidence_threshold: Confidence threshold for detection (0.0-1.0)
            use_gpu: Enable GPU acceleration if available
            batch_size: Batch size for GPU processing
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics package not installed. Run: pip install ultralytics")
        
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.model = None
        self.cancelled = False  # Cancellation flag for batch processing
        
        # Initialize enabled categories (all enabled by default)
        self.enabled_categories = {
            'persons': True,
            'personal_objects': True,
            'animals': True
        }
        
        # Device selection
        self.device = self._select_device(use_gpu)
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self._initialize_model()
        
    def _select_device(self, use_gpu: bool) -> str:
        """Select compute device (CUDA or CPU) with comprehensive compatibility checking"""
        if not TORCH_AVAILABLE:
            if use_gpu:
                logger.warning("GPU requested but PyTorch not available. Using CPU.")
            return 'cpu'
        
        if use_gpu:
            try:
                cuda_available = torch.cuda.is_available()
                cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else 'unknown'
                device_count = torch.cuda.device_count() if cuda_available else 0
                
                logger.info(f"CUDA detection: available={cuda_available}, version={cuda_version}, devices={device_count}")
                
                if cuda_available and device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    
                    # Get compute capability
                    try:
                        compute_capability = torch.cuda.get_device_capability(0)
                        compute_cap_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
                        logger.info(f"GPU Device: {device_name} (Compute Capability: {compute_cap_str})")
                    except Exception:
                        compute_cap_str = "unknown"
                        logger.info(f"GPU Device: {device_name}")
                    
                    # Test GPU compatibility with actual tensor operations
                    try:
                        test_tensor = torch.zeros(1, device='cuda')
                        test_result = test_tensor + 1
                        del test_tensor, test_result
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        logger.info("[OK] GPU compatibility verified - CUDA operations successful")
                        return 'cuda:0'
                    except RuntimeError as e:
                        error_msg = str(e).lower()
                        
                        # Detect specific compatibility issues
                        if "sm_" in error_msg or "cuda capability" in error_msg or "no kernel image" in error_msg:
                            logger.error(f"[!] GPU architecture incompatibility detected")
                            logger.error(f"   GPU: {device_name} ({compute_cap_str})")
                            logger.error(f"   PyTorch was compiled for older CUDA compute capabilities")
                            
                            if "sm_12" in compute_cap_str or "RTX 50" in device_name:
                                logger.error(f"   RTX 50-series (Blackwell) requires PyTorch 2.7+ or nightly build")
                                logger.error(f"   Install: pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128")
                            else:
                                logger.error(f"   Try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
                            
                            logger.warning(f"=> Falling back to CPU for masking operations")
                        else:
                            logger.warning(f"GPU test failed: {e}. Falling back to CPU.")
                        
                        torch.cuda.empty_cache()
                        return 'cpu'
                    except Exception as e:
                        logger.warning(f"GPU initialization error: {e}. Falling back to CPU.")
                        return 'cpu'
                else:
                    logger.warning(f"GPU requested but CUDA not available (version={cuda_version}, devices={device_count}). Using CPU.")
                    return 'cpu'
            except Exception as e:
                logger.warning(f"Error detecting CUDA: {e}. Falling back to CPU.")
                return 'cpu'
        else:
            logger.info("CPU mode selected (GPU disabled in settings)")
            return 'cpu'
    
    def _initialize_model(self):
        """Load YOLOv8 segmentation model"""
        try:
            model_filename = YOLOV8_MODELS[self.model_size]['filename']
            logger.info(f"Loading YOLOv8 model: {model_filename}")
            
            # YOLOv8 will auto-download model if not present
            self.model = YOLO(model_filename)
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.overrides['verbose'] = False  # Reduce console output
            
            model_info = YOLOV8_MODELS[self.model_size]
            logger.info(f"YOLOv8 model loaded successfully: {model_filename} "
                       f"({model_info['size_mb']}MB, ~{model_info['speed_seconds']}s/image)")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def set_enabled_categories(self, persons: bool = True, personal_objects: bool = True, 
                              animals: bool = True):
        """
        Configure which categories to mask.
        
        IMPORTANT: All images are processed ONCE. All enabled categories are detected in a 
        SINGLE pass through the YOLOv8 model. The model detects all enabled object types
        simultaneously and creates a SINGLE binary mask combining all detections.
        
        Example: If persons AND personal_objects are enabled:
        - App processes each image ONE time through YOLOv8
        - Model detects BOTH persons AND backpacks/phones in that single pass
        - One mask file is created with BOTH persons and objects masked
        
        Args:
            persons: Mask persons (COCO class 0)
            personal_objects: Mask personal objects (backpacks, phones, etc.)
            animals: Mask all animals
        """
        self.enabled_categories['persons'] = persons
        self.enabled_categories['personal_objects'] = personal_objects
        self.enabled_categories['animals'] = animals
        
        enabled = [k for k, v in self.enabled_categories.items() if v]
        logger.info(f"Enabled masking categories: {', '.join(enabled)}")
    
    def request_cancellation(self):
        """Request cancellation of current batch processing"""
        self.cancelled = True
        logger.info("Masking cancellation requested")
    
    def get_target_classes(self) -> List[int]:
        """
        Get list of COCO class IDs to detect based on enabled categories.
        
        Returns:
            List of class IDs
        """
        target_classes = []
        
        for category, enabled in self.enabled_categories.items():
            if enabled:
                classes = MASKING_CATEGORIES[category]['classes']
                target_classes.extend(classes)
        
        # Remove duplicates and sort
        target_classes = sorted(list(set(target_classes)))
        
        logger.debug(f"Target class IDs: {target_classes}")
        return target_classes
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to: {self.confidence_threshold}")
    
    def generate_mask(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate binary mask for a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Binary mask as numpy array (uint8, 0 or 255), or None if failed
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            return self.generate_mask_from_array(image)
            
        except Exception as e:
            logger.error(f"Error generating mask for {image_path}: {e}")
            return None
    
    def _safe_predict(self, image, **kwargs):
        """
        Run model prediction with automatic fallback to CPU if CUDA fails.
        Handles 'no kernel image is available' error for unsupported GPUs (e.g. RTX 50-series).
        """
        try:
            # Ensure device is passed in kwargs or use self.device
            if 'device' not in kwargs:
                kwargs['device'] = self.device
                
            return self.model.predict(image, **kwargs)
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA error" in error_msg or "no kernel image" in error_msg:
                if self.device != 'cpu':
                    logger.warning(f"CUDA inference failed: {error_msg}")
                    logger.warning("Falling back to CPU for remaining operations.")
                    self.device = 'cpu'
                    # Retry with CPU
                    kwargs['device'] = 'cpu'
                    return self.model.predict(image, **kwargs)
            # Re-raise if it's not a CUDA error or if we were already on CPU
            raise e

    def generate_mask_from_array(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate binary mask from numpy array.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Binary mask (0=remove, 255=keep) or None if failed
        """
        try:
            h, w = image.shape[:2]
            
            # Get target classes based on enabled categories
            target_classes = self.get_target_classes()
            
            if not target_classes:
                # No categories enabled - return all white (keep everything)
                logger.warning("No masking categories enabled")
                return np.full((h, w), MASK_VALUE_KEEP, dtype=np.uint8)
            
            # Run YOLOv8 inference with safe fallback
            results = self._safe_predict(
                image,
                classes=target_classes,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Extract masks
            if len(results) == 0 or results[0].masks is None:
                # No objects detected - return all white (keep everything)
                logger.debug("No objects detected in image")
                return np.full((h, w), MASK_VALUE_KEEP, dtype=np.uint8)
            
            # Combine all detected object masks
            combined_mask = self._combine_masks(results[0], (h, w))
            
            num_detections = len(results[0].masks.data)
            logger.debug(f"Generated mask with {num_detections} object(s) masked")
            
            return combined_mask
            
        except Exception as e:
            logger.error(f"Error generating mask from array: {e}")
            return None
    
    def _combine_masks(self, result, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract and combine all object masks from YOLO results.
        
        Args:
            result: YOLO result object
            image_shape: Target image shape (height, width)
            
        Returns:
            Combined binary mask (0=mask/remove, 255=keep)
        """
        h, w = image_shape
        # Start with white background (255 = keep everything)
        combined_mask = np.full((h, w), MASK_VALUE_KEEP, dtype=np.uint8)
        
        # Process each detected object
        for i, mask_data in enumerate(result.masks.data):
            # Convert mask to numpy array
            mask = mask_data.cpu().numpy().astype(np.uint8)
            
            # Resize mask to original image size
            mask_resized = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Set object regions to black (0 = mask/remove)
            combined_mask[mask_resized > 0] = MASK_VALUE_REMOVE
        
        return combined_mask
    
    def has_objects(self, image: np.ndarray) -> bool:
        """
        Check if image contains any target objects (fast detection without generating mask).
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            True if objects detected, False otherwise
        """
        try:
            target_classes = self.get_target_classes()
            
            if not target_classes:
                return False
            
            # Run YOLOv8 inference with safe fallback
            results = self._safe_predict(
                image,
                classes=target_classes,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Check if any objects detected
            has_detection = len(results) > 0 and results[0].masks is not None
            
            if has_detection:
                num_objects = len(results[0].masks.data)
                logger.debug(f"Detected {num_objects} object(s)")
            else:
                logger.debug("No objects detected")
                
            return has_detection
            
        except Exception as e:
            logger.error(f"Error checking for objects: {e}")
            return False
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     save_visualization: bool = False,
                     progress_callback = None,
                     cancellation_check = None) -> Dict:
        """
        Process all images in a directory and generate masks.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save masks
            save_visualization: If True, save visualization overlays
            progress_callback: Optional callback(current, total, message)
            cancellation_check: Optional callback() that returns True if cancelled
            
        Returns:
            Dictionary with results: {successful, failed, total, skipped}
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        # Search recursively for images (supports dual-lens subdirectories like lens_1/, lens_2/)
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))  # Direct files
            image_files.extend(input_path.glob(ext.upper()))  # Uppercase
            image_files.extend(input_path.glob(f'**/{ext}'))  # Subdirectories (lowercase)
            image_files.extend(input_path.glob(f'**/{ext.upper()}'))  # Subdirectories (uppercase)
        
        total = len(image_files)
        successful = 0
        failed = 0
        skipped = 0
        
        logger.info(f"Processing {total} images for multi-category masking...")
        
        for idx, img_path in enumerate(image_files):
            # Check cancellation before processing each image
            if cancellation_check and cancellation_check():
                logger.info("Masking batch processing cancelled by user")
                break
            
            try:
                if progress_callback:
                    progress_callback(idx + 1, total, f"Processing {img_path.name}")
                
                # Check if mask already exists
                mask_filename = f"{img_path.stem}_mask.png"
                mask_path = output_path / mask_filename
                
                if mask_path.exists():
                    skipped += 1
                    logger.debug(f"Skipping (already exists): {img_path.name}")
                    continue
                
                # SMART MASK SKIPPING: Check if image has any target objects first
                image = cv2.imread(str(img_path))
                if image is None:
                    failed += 1
                    logger.error(f"Failed to load image: {img_path.name}")
                    continue
                
                has_objects = self.has_objects(image)
                
                if not has_objects:
                    # No target objects detected - skip mask creation
                    skipped += 1
                    logger.debug(f"No detections in {img_path.name} - skipped mask creation")
                    continue
                
                # Objects detected - generate and save mask
                logger.debug(f"Detections found in {img_path.name} - creating mask")
                mask = self.generate_mask_from_array(image)
                
                if mask is not None:
                    # Save mask
                    cv2.imwrite(str(mask_path), mask)
                    
                    # Optionally save visualization
                    if save_visualization:
                        vis_path = output_path / f"{img_path.stem}_masked_vis.jpg"
                        self._save_visualization(str(img_path), mask, str(vis_path))
                    
                    successful += 1
                    logger.debug(f"Mask created: {img_path.name}")
                else:
                    failed += 1
                    logger.warning(f"Failed to generate mask: {img_path.name}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {img_path.name}: {e}")
        
        results = {
            'successful': successful,
            'failed': failed,
            'total': total,
            'skipped': skipped
        }
        
        logger.info(f"Batch processing complete: {successful} masks created, "
                   f"{failed} failed, {skipped} skipped (no detections or already exist)")
        
        return results
    
    def _save_visualization(self, image_path: str, mask: np.ndarray, output_path: str):
        """
        Save visualization of mask overlay on original image.
        
        Args:
            image_path: Path to original image
            mask: Binary mask
            output_path: Path to save visualization
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return
            
            # Create red overlay for masked regions (where mask == 0)
            overlay = image.copy()
            overlay[mask == MASK_VALUE_REMOVE] = [0, 0, 255]  # Red color (BGR)
            
            # Blend original and overlay
            result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            
            cv2.imwrite(output_path, result)
            
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
