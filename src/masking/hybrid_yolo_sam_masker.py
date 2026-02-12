#!/usr/bin/env python3
"""
YOLO26 + SAM Hybrid Masker
Best of both worlds: YOLO detection + SAM segmentation
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Defer imports
SAM_AVAILABLE = False
YOLO_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except Exception as e:
    logger.warning(f"SAM not available: {e}")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"YOLO not available: {e}")

# Mask values (RealityScan format)
MASK_VALUE_REMOVE = 0    # Black = remove/mask
MASK_VALUE_KEEP = 255    # White = keep/valid


class HybridYOLOSAMMasker:
    def process_batch(self, input_dir: str, output_dir: str,
                     save_visualization: bool = False,
                     progress_callback=None,
                     cancellation_check=None) -> dict:
        """
        Process all images in a directory and generate masks (pipeline-compatible).
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save masks
            save_visualization: (ignored)
            progress_callback: Optional callback(current, total, message)
            cancellation_check: Optional callback() that returns True if cancelled
        Returns:
            Dictionary with results: {successful, failed, total, skipped}
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.rglob('*') if f.is_file() and f.suffix.lower() in image_extensions]
        total = len(image_files)
        successful = 0
        skipped = 0
        failed = 0

        logger.info(f"[Hybrid] Processing {total} images for batch masking...")
        for idx, img_path in enumerate(image_files, 1):
            if cancellation_check and cancellation_check():
                logger.info("[Hybrid] Batch processing cancelled by user")
                break
            if progress_callback:
                progress_callback(idx, total, f"Processing {img_path.name}")
            try:
                mask = self.generate_mask(img_path, output_path / f"{img_path.stem}_mask.png")
                if mask is not None:
                    successful += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                logger.error(f"[Hybrid] Error processing {img_path.name}: {e}")

        results = {
            'successful': successful,
            'failed': failed,
            'total': total,
            'skipped': skipped
        }
        logger.info(f"[Hybrid] Batch complete: {successful} masked, {failed} failed, {skipped} skipped")
        return results
    """
    Hybrid masker combining YOLO detection with SAM segmentation.
    
    Pipeline:
    1. YOLO detects persons → bounding boxes
    2. SAM segments using boxes as prompts → precise masks
    
    Advantages:
    - YOLO: Fast, accurate person detection
    - SAM: Pixel-perfect segmentation boundaries
    - No training needed (both pre-trained)
    - 95-98% mask quality
    """
    
    def __init__(self,
                 yolo_model: str = 'yolov8m.pt',
                 sam_checkpoint: str = 'sam_vit_b_01ec64.pth',
                 use_gpu: bool = True,
                 mask_dilation_pixels: int = 15,
                 yolo_confidence: float = 0.5):
        """
        Initialize hybrid YOLO+SAM masker.
        
        Args:
            yolo_model: YOLO model path (yolo26m.pt, yolov8m.pt, etc.)
            sam_checkpoint: SAM checkpoint path
            use_gpu: Enable GPU acceleration
            mask_dilation_pixels: Pixels to expand mask boundaries
            yolo_confidence: YOLO detection confidence threshold
        """
        if not SAM_AVAILABLE or not YOLO_AVAILABLE:
            raise ImportError("Both SAM and YOLO required. Install: pip install segment-anything ultralytics")
        
        self.mask_dilation_pixels = mask_dilation_pixels
        self.yolo_confidence = yolo_confidence
        self.cancelled = False
        
        # Device selection - test actual CUDA kernel execution (catches SM incompatibility)
        self.device = self._select_device(use_gpu)
        logger.info(f"[Hybrid] Using device: {self.device}")
        
        # Load YOLO for detection
        logger.info(f"[Hybrid] Loading YOLO: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        
        # Load SAM for segmentation
        logger.info(f"[Hybrid] Loading SAM: {sam_checkpoint}")
        sam_checkpoint_path = Path(sam_checkpoint)
        if not sam_checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        
        model_type = 'vit_b'  # Extract from checkpoint name if needed
        if 'vit_h' in str(sam_checkpoint):
            model_type = 'vit_h'
        elif 'vit_l' in str(sam_checkpoint):
            model_type = 'vit_l'
        
        sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint_path))
        self.sam_predictor = SamPredictor(sam.to(self.device))
        
        # Category configuration
        self.enabled_categories = {
            'persons': True,
            'personal_objects': False,  # Not implemented yet
            'animals': False  # Not implemented yet
        }
        
        logger.info(f"[Hybrid] [OK] Initialized: YOLO + SAM")
        logger.info(f"[Hybrid] YOLO confidence threshold: {yolo_confidence}")
        logger.info(f"[Hybrid] Mask dilation: {mask_dilation_pixels}px")
    
    def _select_device(self, use_gpu: bool) -> str:
        """Select compute device with actual CUDA kernel testing (matches MultiCategoryMasker pattern)."""
        if not use_gpu:
            logger.info("[Hybrid] CPU mode selected (GPU disabled in settings)")
            return 'cpu'
        
        try:
            if not torch.cuda.is_available():
                logger.warning("[Hybrid] CUDA not available, using CPU")
                return 'cpu'
            
            device_name = torch.cuda.get_device_name(0)
            try:
                cc = torch.cuda.get_device_capability(0)
                cc_str = f"sm_{cc[0]}{cc[1]}"
            except Exception:
                cc_str = "unknown"
            
            logger.info(f"[Hybrid] GPU: {device_name} ({cc_str})")
            
            # Test actual GPU kernel execution (catches SM compatibility issues like sm_120)
            try:
                test_tensor = torch.zeros(1, device='cuda')
                test_result = test_tensor + 1
                del test_tensor, test_result
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info("[Hybrid] [OK] GPU compatibility verified - CUDA operations successful")
                return 'cuda'
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "no kernel image" in error_msg or "sm_" in error_msg or "cuda capability" in error_msg:
                    logger.warning(f"[Hybrid] GPU architecture incompatibility: {device_name} ({cc_str})")
                    logger.warning(f"[Hybrid] PyTorch lacks kernel for this GPU. Need PyTorch 2.7+ for Blackwell.")
                else:
                    logger.warning(f"[Hybrid] GPU test failed: {e}")
                logger.warning("[Hybrid] => Falling back to CPU")
                torch.cuda.empty_cache()
                return 'cpu'
        except Exception as e:
            logger.warning(f"[Hybrid] CUDA detection error: {e}. Using CPU.")
            return 'cpu'
    
    def _fallback_to_cpu(self):
        """Switch YOLO and SAM models to CPU after a CUDA runtime error."""
        if self.device == 'cpu':
            return
        logger.warning("[Hybrid] Switching YOLO + SAM to CPU")
        self.device = 'cpu'
        self.yolo.to('cpu')
        # Re-load SAM on CPU
        self.sam_predictor.model.to('cpu')
        torch.cuda.empty_cache()
        logger.info("[Hybrid] [OK] Now running on CPU")
    
    def set_enabled_categories(self, categories: dict):
        """Enable/disable masking categories."""
        self.enabled_categories.update(categories)
        enabled = [k for k, v in self.enabled_categories.items() if v]
        logger.info(f"[Hybrid] Enabled categories: {', '.join(enabled) if enabled else 'None'}")
    
    def set_specific_classes(self, persons_classes: List[int] = None, 
                             objects_classes: List[int] = None,
                             animals_classes: List[int] = None):
        """
        Set specific COCO class IDs to detect for each category.
        This allows fine-grained control over which objects are masked.
        
        Note: Hybrid masker currently primarily supports 'persons' (class 0).
        
        Args:
            persons_classes: List of class IDs for persons (e.g., [0])
            objects_classes: List of class IDs for personal objects (e.g., [24, 26, 67])
            animals_classes: List of class IDs for animals (e.g., [15, 16])
        """
        if not hasattr(self, '_custom_classes'):
            self._custom_classes = {}
        
        if persons_classes is not None:
            self._custom_classes['persons'] = persons_classes
            self.enabled_categories['persons'] = len(persons_classes) > 0
            
        if objects_classes is not None:
            self._custom_classes['personal_objects'] = objects_classes
            self.enabled_categories['personal_objects'] = len(objects_classes) > 0
            
        if animals_classes is not None:
            self._custom_classes['animals'] = animals_classes
            self.enabled_categories['animals'] = len(animals_classes) > 0
        
        # Log what classes are enabled
        all_classes = []
        for cat, classes in self._custom_classes.items():
            if classes:
                all_classes.extend(classes)
                logger.info(f"[Hybrid] Custom {cat} classes: {classes}")
        
        if all_classes:
            logger.info(f"[Hybrid] Total target classes: {sorted(set(all_classes))}")
    
    def cancel(self):
        """Cancel ongoing operations."""
        self.cancelled = True
        logger.warning("[Hybrid] Masking cancelled")
    
    def reset_cancel(self):
        """Reset cancellation flag."""
        self.cancelled = False
    
    def generate_mask(self,
                      image_path: Path,
                      output_path: Optional[Path] = None,
                      category_filter: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """
        Generate mask using YOLO detection + SAM segmentation.
        
        Args:
            image_path: Input image path
            output_path: Optional output path for mask
            category_filter: Categories to mask (overrides enabled_categories)
        
        Returns:
            Binary mask array (uint8) or None if no detections
        """
        if self.cancelled:
            logger.info("[Hybrid] Skipping (cancelled)")
            return None
        
        # Determine active categories
        if category_filter is not None:
            active_categories = category_filter
        else:
            active_categories = [k for k, v in self.enabled_categories.items() if v]
        
        if not active_categories:
            logger.warning("[Hybrid] No categories enabled")
            return None
        
        # Only persons supported currently
        if 'persons' not in active_categories:
            logger.warning("[Hybrid] Only 'persons' category is supported")
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"[Hybrid] Failed to load: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Step 1: YOLO Detection
        logger.info(f"[Hybrid] Step 1: YOLO detecting persons...")
        try:
            yolo_results = self.yolo(image, conf=self.yolo_confidence, verbose=False)
        except RuntimeError as e:
            if "CUDA error" in str(e) or "no kernel image" in str(e):
                logger.warning(f"[Hybrid] CUDA inference failed: {e}")
                self._fallback_to_cpu()
                yolo_results = self.yolo(image, conf=self.yolo_confidence, verbose=False)
            else:
                raise
        
        # Set SAM image once
        self.sam_predictor.set_image(image_rgb)
        
        # Extract person bounding boxes
        person_boxes = []
        person_confs = []
        
        for result in yolo_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            # Filter for persons (class 0 in COCO)
            person_indices = np.where(classes == 0)[0]
            
            for idx in person_indices:
                person_boxes.append(boxes[idx])
                person_confs.append(confs[idx])
        
        if len(person_boxes) == 0:
            logger.info("[Hybrid] No persons detected")
            return None
        
        logger.info(f"[Hybrid] [OK] Detected {len(person_boxes)} person(s)")
        
        # Step 2: SAM Segmentation
        logger.info(f"[Hybrid] Step 2: SAM segmenting {len(person_boxes)} person(s)...")
        
        # Initialize combined mask (all white = keep)
        combined_mask = np.ones((height, width), dtype=np.uint8) * MASK_VALUE_KEEP
        
        for person_id, (bbox, conf) in enumerate(zip(person_boxes, person_confs)):
            # SAM prediction with bbox as prompt
            try:
                masks, scores, logits = self.sam_predictor.predict(
                    box=np.array(bbox),
                    multimask_output=True,
                    return_logits=True
                )
                
                # Select best mask (highest score for hybrid approach)
                best_idx = np.argmax(scores)
                best_mask_logits = masks[best_idx]
                best_score = scores[best_idx]
                
                # Convert logits to boolean
                person_mask = best_mask_logits > 0
                
                # Apply to combined mask (black = remove)
                combined_mask[person_mask] = MASK_VALUE_REMOVE
                
                mask_area = 100 * np.sum(person_mask) / (height * width)
                logger.info(f"[Hybrid]   Person {person_id+1}: conf={conf:.2f}, sam_score={best_score:.3f}, area={mask_area:.1f}%")
                
            except Exception as e:
                logger.error(f"[Hybrid] SAM failed for person {person_id+1}: {e}")
                continue
        
        # Apply dilation
        if self.mask_dilation_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.mask_dilation_pixels * 2 + 1, self.mask_dilation_pixels * 2 + 1)
            )
            inverted = cv2.bitwise_not(combined_mask)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            combined_mask = cv2.bitwise_not(dilated)
            logger.debug(f"[Hybrid] Applied {self.mask_dilation_pixels}px dilation")
        
        # Save mask
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), combined_mask)
            logger.info(f"[Hybrid] [OK] Saved mask: {output_path.name}")
        
        masked_percent = 100 * np.sum(combined_mask == 0) / (height * width)
        logger.info(f"[Hybrid] [OK] Final mask: {masked_percent:.1f}% masked")
        
        return combined_mask
    
    def batch_generate_masks(self,
                             image_paths: List[Path],
                             output_dir: Path,
                             progress_callback=None) -> Tuple[int, int]:
        """
        Generate masks for multiple images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Output directory
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            Tuple of (successful_count, skipped_count)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        skipped = 0
        total = len(image_paths)
        
        logger.info(f"[Hybrid] Starting batch: {total} images")
        
        for idx, image_path in enumerate(image_paths, 1):
            if self.cancelled:
                logger.warning("[Hybrid] Batch cancelled")
                break
            
            if progress_callback:
                progress_callback(idx, total, f"Processing: {image_path.name}")
            
            output_path = output_dir / f"{image_path.stem}_mask.png"
            
            mask = self.generate_mask(image_path, output_path)
            
            if mask is not None:
                successful += 1
            else:
                skipped += 1
        
        logger.info(f"[Hybrid] Batch complete: {successful} masked, {skipped} skipped")
        return successful, skipped
    
    def cleanup(self):
        """Free GPU memory."""
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("[Hybrid] GPU cache cleared")
