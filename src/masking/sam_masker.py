"""
SAM (Segment Anything Model) ViT-B Masking Module
Alternative to YOLO for Stage 3 masking with improved segmentation quality.

SAM advantages over YOLO:
- Superior segmentation quality (especially edges and complex boundaries)
- Better handling of overlapping objects
- More precise mask boundaries
- Works well with prompt-free automatic mask generation

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

# Defer SAM imports to avoid build-time issues
SAM_AVAILABLE = False
sam_model_registry = None
SamPredictor = None
torch = None

try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
    logging.info(f"SAM loaded successfully. PyTorch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    logging.warning(f"SAM not available. Install with: pip install segment-anything. Error: {e}")
except Exception as e:
    logging.error(f"SAM import failed: {e}", exc_info=True)

from ..config.defaults import (
    MASK_VALUE_REMOVE, MASK_VALUE_KEEP, DEFAULT_USE_GPU
)

logger = logging.getLogger(__name__)


class SAMMasker:
    """
    SAM ViT-B based masking for photogrammetry workflows.
    Uses PROMPT-BASED segmentation with bounding boxes (recommended approach).
    Provides high-quality segmentation for persons, objects, and animals.
    """
    
    def __init__(self, 
                 model_checkpoint: str = 'sam_vit_b_01ec64.pth',
                 use_gpu: bool = DEFAULT_USE_GPU,
                 mask_dilation_pixels: int = 15,
                 confidence_threshold: float = 0.8):
        """
        Initialize SAMMasker with SAM ViT-B model.
        
        Args:
            model_checkpoint: Path to SAM checkpoint file (.pth)
            use_gpu: Enable GPU acceleration if available
            mask_dilation_pixels: Pixels to expand mask boundaries
            confidence_threshold: Minimum score for accepting masks (0.0-1.0)
        """
        if not SAM_AVAILABLE:
            raise ImportError("Segment Anything (SAM) not installed. Run: pip install segment-anything")
        
        self.model_checkpoint = Path(model_checkpoint)
        self.mask_dilation_pixels = mask_dilation_pixels
        self.confidence_threshold = confidence_threshold
        self.cancelled = False
        
        # Check if checkpoint exists
        if not self.model_checkpoint.exists():
            logger.error(f"SAM checkpoint not found: {self.model_checkpoint}")
            logger.info("Download SAM ViT-B checkpoint from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            raise FileNotFoundError(f"SAM checkpoint not found: {self.model_checkpoint}")
        
        # Initialize enabled categories
        self.enabled_categories = {
            'persons': True,
            'personal_objects': True,
            'animals': True
        }
        
        # Device selection
        self.device = self._select_device(use_gpu)
        logger.info(f"Using device: {self.device}")
        
        # Load SAM model
        self.sam = self._load_model()
        
        # Initialize predictor (PROMPT-BASED, not automatic)
        self.predictor = SamPredictor(self.sam)
        
        logger.info(f"[SAM] Model loaded: ViT-B (PROMPT-BASED), dilation: {mask_dilation_pixels}px")
        logger.info(f"[SAM] Confidence threshold: {confidence_threshold}")
    
    def _select_device(self, use_gpu: bool) -> str:
        """Select compute device (CUDA or CPU)"""
        if not torch:
            return 'cpu'
        
        if use_gpu and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"[SAM] GPU Device: {device_name}")
            
            # Test GPU compatibility
            try:
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info("[SAM] ✅ GPU compatibility verified")
                return 'cuda'
            except Exception as e:
                logger.warning(f"[SAM] GPU test failed: {e}. Falling back to CPU.")
                return 'cpu'
        else:
            if use_gpu:
                logger.warning("[SAM] GPU requested but not available. Using CPU.")
            return 'cpu'
    
    def _load_model(self):
        """Load SAM model from checkpoint"""
        try:
            logger.info(f"[SAM] Loading model from: {self.model_checkpoint}")
            sam = sam_model_registry["vit_b"](checkpoint=str(self.model_checkpoint))
            sam.to(device=self.device)
            logger.info("[SAM] Model loaded successfully")
            return sam
        except Exception as e:
            logger.error(f"[SAM] Failed to load model: {e}", exc_info=True)
            raise
    
    def set_enabled_categories(self, categories: Dict[str, bool]):
        """
        Set which object categories to mask.
        
        Args:
            categories: Dict with keys 'persons', 'personal_objects', 'animals'
        """
        self.enabled_categories.update(categories)
        enabled = [k for k, v in self.enabled_categories.items() if v]
        logger.info(f"[SAM] Enabled categories: {', '.join(enabled) if enabled else 'None'}")
    
    def cancel(self):
        """Cancel ongoing batch operations"""
        self.cancelled = True
        logger.warning("[SAM] Masking cancelled by user")
    
    def reset_cancel(self):
        """Reset cancellation flag"""
        self.cancelled = False
    
    def _get_person_bbox(self, image: np.ndarray) -> np.ndarray:
        """
        Get bounding box for person segmentation.
        For photogrammetry, we assume person is in foreground (use full image as bbox).
        SAM will automatically segment the person from the full image.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            np.array([x1, y1, x2, y2]) - full image bbox
        """
        height, width = image.shape[:2]
        
        # Use full image as bounding box - SAM will segment foreground objects
        # This works well for photogrammetry where person is main subject
        return np.array([0, 0, width, height])
    
    def generate_mask(self, 
                      image_path: Path,
                      output_path: Optional[Path] = None,
                      category_filter: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """
        Generate binary mask for image using SAM PROMPT-BASED segmentation.
        
        Uses RECOMMENDED approach: bounding box + multimask_output + select best score.
        This is the correct way to use SAM for person masking (not automatic segmentation).
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save mask (auto-generated if None)
            category_filter: List of categories to mask (overrides self.enabled_categories)
        
        Returns:
            Binary mask array (uint8) or None if no objects detected
        """
        if self.cancelled:
            logger.info("[SAM] Skipping mask generation (cancelled)")
            return None
        
        # Determine active categories
        if category_filter is not None:
            active_categories = category_filter
        else:
            active_categories = [k for k, v in self.enabled_categories.items() if v]
        
        if not active_categories:
            logger.warning("[SAM] No categories enabled. Skipping mask generation.")
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"[SAM] Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB (SAM expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Set image for predictor
        self.predictor.set_image(image_rgb)
        
        # Get bounding box around foreground object
        bbox = self._get_person_bbox(image)
        logger.info(f"[SAM] Auto-detected bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
        # SAM prediction with RECOMMENDED PARAMETERS
        try:
            masks, scores, logits = self.predictor.predict(
                box=bbox,
                multimask_output=True,  # ← KEY: Get 3 candidate masks
                return_logits=True       # ← KEY: Get confidence scores
            )
            logger.info(f"[SAM] Generated {len(masks)} candidate masks with scores: {scores}")
        except Exception as e:
            logger.error(f"[SAM] Mask generation failed: {e}", exc_info=True)
            return None
        
        # SAM returns 3 masks with different granularities:
        # - Mask 0 (highest score): Most inclusive (may include background)
        # - Mask 1 (mid score): Intermediate coverage
        # - Mask 2 (lowest score): Most specific (tightest fit to object)
        #
        # For person removal in photogrammetry, we want the TIGHTEST mask
        # that captures just the person, not extra background.
        # Solution: Select mask with MINIMUM area (most specific)
        
        # Convert all masks to boolean
        masks_bool = [mask > 0 for mask in masks]
        areas = [np.sum(mask_bool) for mask_bool in masks_bool]
        
        # Select mask with MINIMUM area (most specific to person)
        best_idx = np.argmin(areas)
        best_mask = masks_bool[best_idx]
        best_score = scores[best_idx]
        best_area_percent = 100 * areas[best_idx] / (height * width)
        
        logger.info(f"[SAM] Generated 3 masks with areas: {[f'{100*a/(height*width):.1f}%' for a in areas]}")
        logger.info(f"[SAM] Selected mask #{best_idx} (most specific): area={best_area_percent:.1f}%, score={best_score:.3f}")
        
        # Check if quality is acceptable
        if best_score < self.confidence_threshold:
            logger.warning(f"[SAM] Low quality mask (score {best_score:.3f} < {self.confidence_threshold})")
            # Still return it, but log warning
        
        # Convert to binary mask (RealityScan format)
        # SAM mask: True = object (to remove), False = background (to keep)
        # RealityScan: 0 (black) = remove/mask, 255 (white) = keep/valid
        combined_mask = np.ones((height, width), dtype=np.uint8) * MASK_VALUE_KEEP  # Start with all white (keep everything)
        combined_mask[best_mask] = MASK_VALUE_REMOVE  # Set masked regions to black (remove objects)
        
        masked_count = 1
        mask_area_percent = np.sum(best_mask) / (height * width) * 100
        logger.info(f"[SAM] Applied mask: {mask_area_percent:.1f}% of image area masked")
        
        # Apply dilation to expand mask boundaries
        if self.mask_dilation_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (self.mask_dilation_pixels * 2 + 1, 
                                                self.mask_dilation_pixels * 2 + 1))
            # Dilate the masked regions (black areas)
            inverted = cv2.bitwise_not(combined_mask)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            combined_mask = cv2.bitwise_not(dilated)
            logger.debug(f"[SAM] Applied {self.mask_dilation_pixels}px dilation")
        
        # Save mask
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), combined_mask)
            logger.info(f"[SAM] Saved mask: {output_path.name}")
        
        return combined_mask
    
    def batch_generate_masks(self,
                             image_paths: List[Path],
                             output_dir: Path,
                             progress_callback=None) -> Tuple[int, int]:
        """
        Generate masks for multiple images (batch processing).
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save masks
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            Tuple of (successful_count, skipped_count)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        skipped = 0
        total = len(image_paths)
        
        logger.info(f"[SAM] Starting batch masking: {total} images")
        
        for idx, image_path in enumerate(image_paths, 1):
            if self.cancelled:
                logger.warning("[SAM] Batch processing cancelled")
                break
            
            # Progress callback
            if progress_callback:
                progress_callback(idx, total, f"Processing: {image_path.name}")
            
            # Generate output path
            output_path = output_dir / f"{image_path.stem}_mask.png"
            
            # Generate mask
            mask = self.generate_mask(image_path, output_path)
            
            if mask is not None:
                successful += 1
            else:
                skipped += 1
        
        logger.info(f"[SAM] Batch complete: {successful} masked, {skipped} skipped")
        return successful, skipped
    
    def cleanup(self):
        """Free GPU memory"""
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[SAM] GPU memory cleared")


def download_sam_checkpoint(model_type: str = 'vit_b', output_dir: Path = Path('.')) -> Path:
    """
    Helper function to download SAM checkpoint.
    
    Args:
        model_type: 'vit_b', 'vit_l', or 'vit_h'
        output_dir: Directory to save checkpoint
    
    Returns:
        Path to downloaded checkpoint
    """
    import urllib.request
    
    checkpoints = {
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    }
    
    if model_type not in checkpoints:
        raise ValueError(f"Invalid model type: {model_type}. Choose from {list(checkpoints.keys())}")
    
    url = checkpoints[model_type]
    filename = url.split('/')[-1]
    output_path = output_dir / filename
    
    if output_path.exists():
        logger.info(f"[SAM] Checkpoint already exists: {output_path}")
        return output_path
    
    logger.info(f"[SAM] Downloading {model_type} checkpoint from {url}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"[SAM] Downloaded successfully: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"[SAM] Download failed: {e}", exc_info=True)
        raise
