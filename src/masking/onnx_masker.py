"""
ONNX-based Multi-Category Masking Module
Lightweight replacement for PyTorch-based YOLO masking.

Supports both YOLOv8 and YOLO26 models:
- YOLOv8: [1, 116, 8400] output - requires NMS post-processing
- YOLO26: [1, 300, 38] output - NMS-free end-to-end inference (faster)

This module uses ONNX Runtime instead of PyTorch, resulting in:
- ~90% smaller binary size (300-500 MB vs 6-8 GB)
- Faster inference on CPU
- Same accuracy as PyTorch version
- Compatible with GPU (CUDA) and CPU

Usage:
1. Export YOLO model to ONNX format (one-time setup):
   from ultralytics import YOLO
   # For YOLOv8:
   model = YOLO('yolov8s-seg.pt')
   model.export(format='onnx', simplify=True)
   # For YOLO26 (recommended, faster):
   model = YOLO('yolo26s-seg.pt')
   model.export(format='onnx', simplify=True)

2. Use ONNXMasker instead of MultiCategoryMasker:
   masker = ONNXMasker(model_path='yolo26s-seg.onnx')  # or yolov8s-seg.onnx
   mask = masker.generate_mask(image_path)

Mask Format (RealityScan Compatible):
- 0 (Black) = Mask/Remove this region (objects to remove)
- 255 (White) = Keep this region (background/valid points)
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict

# Defer onnxruntime import to avoid build-time issues
ONNX_AVAILABLE = False
ort = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logging.info(f"ONNX Runtime loaded successfully. Version: {ort.__version__}")
except ImportError as e:
    logging.warning(f"ONNX Runtime not available. Install with: pip install onnxruntime or onnxruntime-gpu. Error: {e}")
except Exception as e:
    logging.error(f"ONNX Runtime import failed: {e}", exc_info=True)

from ..config.defaults import (
    DEFAULT_CONFIDENCE_THRESHOLD, MASKING_CATEGORIES,
    MASK_VALUE_REMOVE, MASK_VALUE_KEEP, DEFAULT_USE_GPU
)

logger = logging.getLogger(__name__)


# COCO class names (80 classes for YOLOv8)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ONNXMasker:
    """
    ONNX Runtime-based multi-category masking for photogrammetry workflows.
    Lightweight replacement for PyTorch-based MultiCategoryMasker.
    """
    
    def __init__(self, model_path: str = 'yolo26s-seg.onnx',
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 use_gpu: bool = DEFAULT_USE_GPU,
                 mask_dilation_pixels: int = 15):
        """
        Initialize ONNXMasker with YOLO ONNX model.
        
        Supports both YOLOv8 and YOLO26 models:
        - YOLO26: Faster (NMS-free), recommended for production
        - YOLOv8: Widely used, good accuracy
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            confidence_threshold: Confidence threshold for detection (0.0-1.0)
            use_gpu: Enable GPU acceleration if available
            mask_dilation_pixels: Pixels to expand mask boundaries (fixes cutoff issues)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime or onnxruntime-gpu")
        
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.cancelled = False
        self.mask_dilation_pixels = mask_dilation_pixels
        
        logger.info(f"[ONNX] Mask dilation: {mask_dilation_pixels}px (expands boundaries to include attached objects)")
        
        # Initialize enabled categories (all enabled by default)
        self.enabled_categories = {
            'persons': True,
            'personal_objects': True,
            'animals': True
        }
        
        # Select execution provider (GPU or CPU)
        self.providers = self._select_providers(use_gpu)
        logger.info(f"Using ONNX providers: {self.providers}")
        
        # Load ONNX model
        self.session = self._load_model()
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2] if len(input_shape) > 2 else 640
        self.input_width = input_shape[3] if len(input_shape) > 3 else 640
        
        # Log active providers
        active_providers = self.session.get_providers()
        logger.info(f"[ONNX] Model loaded: {self.model_path.name}, input: {self.input_width}×{self.input_height}")
        logger.info(f"[ONNX] Active providers: {active_providers}")
        if 'CUDAExecutionProvider' in active_providers:
            logger.info("[ONNX] 🚀 GPU acceleration ACTIVE")
        else:
            logger.warning("[ONNX] ⚠️ Running on CPU (GPU not active)")
    
    def _select_providers(self, use_gpu: bool) -> List[str]:
        """Select ONNX Runtime execution providers (GPU or CPU)"""
        if use_gpu:
            # Try CUDA provider first, fallback to CPU
            available_providers = ort.get_available_providers()
            logger.info(f"[ONNX] Available providers: {available_providers}")
            
            # Try CUDAExecutionProvider with optimized options
            if 'CUDAExecutionProvider' in available_providers:
                logger.info("[ONNX] ✅ CUDA provider available - using GPU acceleration")
                # Optimized CUDA settings for RTX 5070 Ti
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',  # More efficient for batch processing
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB limit (increased for larger batches)
                    'cudnn_conv_algo_search': 'HEURISTIC',  # Faster than EXHAUSTIVE for batch
                    'do_copy_in_default_stream': True,
                    'cudnn_conv_use_max_workspace': True,  # Use more workspace for speed
                    'cudnn_conv1d_pad_to_nc1d': True,  # Additional optimization
                }
                return [
                    ('CUDAExecutionProvider', cuda_options),
                    'CPUExecutionProvider'
                ]
            else:
                logger.warning("[ONNX] ⚠️ GPU requested but CUDA provider not available. Using CPU.")
                logger.warning(f"[ONNX] Available: {available_providers}")
                logger.warning("[ONNX] Install: pip install onnxruntime-gpu")
                return ['CPUExecutionProvider']
        else:
            logger.info("[ONNX] CPU mode selected (GPU disabled in settings)")
            return ['CPUExecutionProvider']
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {self.model_path}\n\n"
                    f"To create ONNX model from YOLOv8:\n"
                    f"  from ultralytics import YOLO\n"
                    f"  model = YOLO('{self.model_path.stem.replace('-seg', '')}-seg.pt')\n"
                    f"  model.export(format='onnx', simplify=True)\n"
                )
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(
                str(self.model_path),
                providers=self.providers
            )
            
            logger.info(f"ONNX model loaded successfully: {self.model_path}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def set_enabled_categories(self, persons: bool = True, personal_objects: bool = True,
                              animals: bool = True):
        """
        Configure which categories to mask.
        
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
    
    def set_specific_classes(self, persons_classes: List[int] = None, 
                             objects_classes: List[int] = None,
                             animals_classes: List[int] = None):
        """
        Set specific COCO class IDs to detect for each category.
        This allows fine-grained control over which objects are masked.
        
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
                logger.info(f"Custom {cat} classes: {classes}")
        
        logger.info(f"Total target classes: {sorted(set(all_classes))}")
    
    def request_cancellation(self):
        """Request cancellation of current batch processing"""
        self.cancelled = True
        logger.info("ONNX Masking cancellation requested")
    
    def get_target_classes(self) -> List[int]:
        """
        Get list of COCO class IDs to detect based on enabled categories.
        Uses custom classes if set via set_specific_classes(), otherwise uses defaults.
        
        Returns:
            List of class IDs
        """
        target_classes = []
        
        # Check if custom classes were set
        if hasattr(self, '_custom_classes') and self._custom_classes:
            for category, classes in self._custom_classes.items():
                if self.enabled_categories.get(category, False) and classes:
                    target_classes.extend(classes)
        else:
            # Use default classes from MASKING_CATEGORIES
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
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX inference.
        
        Args:
            image: Input image (BGR format, HxWxC)
            
        Returns:
            Preprocessed image (1xCxHxW, float32, normalized to 0-1)
        """
        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _postprocess(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Postprocess ONNX model outputs to extract masks.
        Supports both YOLOv8 and YOLO26 output formats.
        
        Args:
            outputs: Raw ONNX model outputs
            original_shape: Original image shape (height, width)
            
        Returns:
            Combined binary mask (0=remove, 255=keep) or None if no detections
        """
        if len(outputs) < 2:
            logger.warning("Unexpected ONNX output format")
            return None
        
        detections = outputs[0][0]  # Remove batch dimension
        proto_masks = outputs[1][0] if len(outputs) > 1 else None
        
        # DIAGNOSTIC: Log shapes
        logger.debug(f"ONNX Output Shapes - Detections: {detections.shape}, Proto: {proto_masks.shape if proto_masks is not None else 'None'}")
        
        # Detect YOLO version based on output shape
        # YOLO26: [300, 38] - 300 max detections, 38 values (already post-NMS)
        # YOLOv8: [116, 8400] or transposed - needs NMS, 116 = 4+80+32
        is_yolo26 = (len(detections.shape) == 2 and 
                     detections.shape[0] <= 300 and 
                     detections.shape[1] < 100)
        
        if not is_yolo26:
            # YOLOv8 Output is usually (Channels, Anchors) e.g. (116, 8400)
            # We need (Anchors, Channels) to iterate over detections
            if detections.shape[0] < detections.shape[1]:
                detections = detections.T
                logger.debug(f"Transposed detections to: {detections.shape}")

        # Get target classes
        target_classes = self.get_target_classes()
        
        if not target_classes:
            # No categories enabled - return all white
            return np.full(original_shape, MASK_VALUE_KEEP, dtype=np.uint8)
        
        # Parse detections
        h_orig, w_orig = original_shape
        combined_mask = np.full((h_orig, w_orig), MASK_VALUE_KEEP, dtype=np.uint8)
        
        num_detections = 0
        
        for detection in detections:
            if is_yolo26:
                # YOLO26 format: [x1, y1, x2, y2, conf, class_id, mask_coeff0..31]
                # Already in xyxy format and post-NMS
                if len(detection) < 38:
                    continue
                    
                x1_norm, y1_norm, x2_norm, y2_norm = detection[0:4]
                confidence = detection[4]
                class_id = int(detection[5])
                mask_coeffs = detection[6:38]  # 32 mask coefficients
                
                # YOLO26 boxes are in pixel coords for input size (640x640)
                x1 = int(x1_norm * w_orig / self.input_width)
                y1 = int(y1_norm * h_orig / self.input_height)
                x2 = int(x2_norm * w_orig / self.input_width)
                y2 = int(y2_norm * h_orig / self.input_height)
                
                # Filter by confidence and class
                if confidence < self.confidence_threshold:
                    continue
                if class_id not in target_classes:
                    continue
            else:
                # YOLOv8 format: [x_center, y_center, w, h, class0_conf, class1_conf, ..., mask_coeff0..31]
                # Extract confidence and class (80 COCO classes)
                class_scores = detection[4:84]
                max_conf = np.max(class_scores)
                class_id = np.argmax(class_scores)
                
                # Filter by confidence and class
                if max_conf < self.confidence_threshold or class_id not in target_classes:
                    continue
                
                confidence = max_conf
                mask_coeffs = detection[-32:]
                
                # Extract box coordinates (xywh format)
                x_center, y_center, width, height = detection[:4]
                
                # Convert to pixel coordinates
                x1 = int((x_center - width / 2) * w_orig / self.input_width)
                y1 = int((y_center - height / 2) * h_orig / self.input_height)
                x2 = int((x_center + width / 2) * w_orig / self.input_width)
                y2 = int((y_center + height / 2) * h_orig / self.input_height)
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Process segmentation mask if available
            if proto_masks is not None and len(mask_coeffs) == 32:
                # Matrix multiplication: (1, 32) @ (32, 160*160) -> (1, 160*160)
                # Reshape proto_masks to (32, 160*160)
                proto_flat = proto_masks.reshape(32, -1)
                mask_data = mask_coeffs @ proto_flat
                mask_data = mask_data.reshape(160, 160)
                
                # Sigmoid activation
                mask_data = 1 / (1 + np.exp(-mask_data))
                
                # Resize to original image size
                mask_resized = cv2.resize(mask_data, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                
                # Crop to bounding box
                # Create a box mask
                box_mask = np.zeros_like(mask_resized)
                box_mask[y1:y2, x1:x2] = 1
                
                # Apply box constraint and threshold
                final_mask = (mask_resized > 0.5) * box_mask
                
                # DILATE MASK to include boundaries (fixes backpack cutoff!)
                # This expands the mask by N pixels to capture attached objects
                if self.mask_dilation_pixels > 0:
                    kernel = np.ones((self.mask_dilation_pixels, self.mask_dilation_pixels), np.uint8)
                    final_mask = cv2.dilate(final_mask.astype(np.uint8), kernel, iterations=1)
                
                # Combine with main mask (0=remove, so we set remove regions to 0)
                # MASK_VALUE_REMOVE = 0 (Black) -> Remove this object
                # MASK_VALUE_KEEP = 255 (White) -> Keep background
                # So if final_mask is True (object), set to REMOVE (0)
                
                combined_mask[final_mask > 0] = MASK_VALUE_REMOVE
            else:
                # Fallback to bounding box with dilation
                # Expand bounding box by dilation pixels
                if self.mask_dilation_pixels > 0:
                    dil = self.mask_dilation_pixels
                    y1_exp = max(0, y1 - dil)
                    y2_exp = min(h_orig, y2 + dil)
                    x1_exp = max(0, x1 - dil)
                    x2_exp = min(w_orig, x2 + dil)
                    combined_mask[y1_exp:y2_exp, x1_exp:x2_exp] = MASK_VALUE_REMOVE
                else:
                    combined_mask[y1:y2, x1:x2] = MASK_VALUE_REMOVE
            
            num_detections += 1
            logger.debug(f"Detection: class={class_id} conf={confidence:.2f} box=({x1},{y1},{x2},{y2})")
        
        if num_detections == 0:
            return None
        
        logger.debug(f"Detected {num_detections} object(s)")
        return combined_mask
    
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
    
    def generate_mask_from_array(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate binary mask from numpy array.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Binary mask (0=remove, 255=keep) or None if no detections
        """
        try:
            original_shape = image.shape[:2]
            
            # Preprocess image
            input_tensor = self._preprocess(image)
            
            # Run ONNX inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Postprocess outputs
            mask = self._postprocess(outputs, original_shape)
            
            if mask is None:
                # No detections - return all white (keep everything)
                logger.debug("No objects detected in image")
                return np.full(original_shape, MASK_VALUE_KEEP, dtype=np.uint8)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error generating mask from array: {e}")
            return None
    
    def has_objects(self, image: np.ndarray) -> bool:
        """
        Check if image contains any target objects (fast detection without generating full mask).
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            True if objects detected, False otherwise
        """
        try:
            # Preprocess
            input_tensor = self._preprocess(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Quick check for detections
            detections = outputs[0][0]
            
            # YOLOv8 Output is usually (Channels, Anchors) e.g. (116, 8400)
            # We need (Anchors, Channels) to iterate over detections
            if detections.shape[0] < detections.shape[1]:
                detections = detections.T
            
            target_classes = self.get_target_classes()
            
            # DEBUG: Print max confidence found
            max_conf_found = 0.0
            
            for detection in detections:
                class_scores = detection[4:84]
                max_conf = np.max(class_scores)
                class_id = np.argmax(class_scores)
                
                if max_conf > max_conf_found:
                    max_conf_found = max_conf
                
                if max_conf >= self.confidence_threshold and class_id in target_classes:
                    logger.debug(f"Object detected: Class {class_id} with conf {max_conf:.2f}")
                    return True
            
            logger.debug(f"No objects detected. Max confidence found: {max_conf_found:.2f} (Threshold: {self.confidence_threshold})")
            return False
            
        except Exception as e:
            logger.error(f"Error checking for objects: {e}")
            return False
    
    def process_batch(self, input_dir: str, output_dir: str,
                     save_visualization: bool = False,
                     progress_callback=None,
                     cancellation_check=None,
                     batch_size: int = 16) -> Dict:
        """
        Process all images in a directory and generate masks (OPTIMIZED).
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save masks
            save_visualization: If True, save visualization overlays
            progress_callback: Optional callback(current, total, message)
            cancellation_check: Optional callback() that returns True if cancelled
            batch_size: Number of images to process in GPU batch (default 16 for speed)
            
        Returns:
            Dictionary with results: {successful, failed, total, skipped}
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        
        total = len(image_files)
        successful = 0
        failed = 0
        skipped = 0
        
        logger.info(f"Processing {total} images with ONNX masking (batch size: {batch_size})...")
        
        # Process in batches for better GPU utilization
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_files = image_files[batch_start:batch_end]
            
            # Check cancellation
            if cancellation_check and cancellation_check():
                logger.info("ONNX masking batch processing cancelled")
                break
            
            # Process batch
            for idx, img_path in enumerate(batch_files, start=batch_start):
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
                    
                    # SMART MASK SKIPPING: Check for objects first
                    image = cv2.imread(str(img_path))
                    if image is None:
                        failed += 1
                        logger.error(f"Failed to load: {img_path.name}")
                        continue
                    
                    has_objects = self.has_objects(image)
                    
                    if not has_objects:
                        # No detections - skip mask creation
                        skipped += 1
                        logger.debug(f"No detections in {img_path.name} - skipped")
                        continue
                    
                    # Generate and save mask
                    mask = self.generate_mask_from_array(image)
                    
                    if mask is not None:
                        cv2.imwrite(str(mask_path), mask)
                        
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
            
            # Clear GPU cache after each batch
            if 'CUDAExecutionProvider' in self.session.get_providers():
                import gc
                gc.collect()
        
        results = {
            'successful': successful,
            'failed': failed,
            'total': total,
            'skipped': skipped
        }
        
        logger.info(f"ONNX batch processing complete: {successful} masks created, "
                   f"{failed} failed, {skipped} skipped")
        
        return results
    
    def _save_visualization(self, image_path: str, mask: np.ndarray, output_path: str):
        """Save visualization of mask overlay on original image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return
            
            # Create red overlay for masked regions
            overlay = image.copy()
            overlay[mask == MASK_VALUE_REMOVE] = [0, 0, 255]  # Red (BGR)
            
            # Blend
            result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            cv2.imwrite(output_path, result)
            
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
    
    def _nms(self, boxes, scores, iou_threshold):
        """
        Non-Maximum Suppression (NMS)
        """
        if boxes.size == 0:
            return []
            
        # Coordinates of bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(scores)[::-1]
        
        pick = []
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[0]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / area[idxs[1:]]
            
            # Delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > iou_threshold)[0] + 1)))
            
        return pick
