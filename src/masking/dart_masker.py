"""
DART (Detect Anything in Real Time) Multi-Class Open-Vocabulary Masking Module.
Converts SAM3 into a high-performance open-vocabulary detector and instance segmenter
for photogrammetry (RealityScan / COLMAP) and 360 video workflows.

Based on mkturkcan/DART (https://github.com/mkturkcan/DART)
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import PIL.Image

from ..config.defaults import (
    MASK_VALUE_KEEP,
    MASK_VALUE_REMOVE,
)
from ..utils.fisheye_mask import (
    build_fisheye_keep_mask,
    clamp_radius_percent,
    combine_with_keep_mask,
)
from ..utils.mask_output import build_related_output_path
from .mask_refinement import MaskRefinementSettings, refine_detected_mask

logger = logging.getLogger(__name__)

# Predefined open-vocabulary categories and rich synonyms
DEFAULT_DART_CATEGORY_PROMPTS: Dict[str, List[str]] = {
    'persons': ['person', 'pedestrian', 'human body'],
    'vehicles': ['car', 'automobile', 'truck', 'bus', 'motorcycle', 'bicycle', 'van'],
    'rigs': ['tripod', 'monopod', 'selfie stick', 'camera mount', 'camera pole', 'stand'],
    'personal_objects': ['backpack', 'handbag', 'bag', 'suitcase', 'hat', 'helmet', 'phone'],
    'animals': ['dog', 'cat', 'horse', 'bird', 'animal', 'pet'],
    'sky': ['sky', 'clouds'],
}

_MASK_FRAME_RE = re.compile(r'^frame_(\d+)_cam_(\d+)\.[^.]+$', re.IGNORECASE)


def _mask_input_sort_key(path_or_name: Union[str, Path]) -> Tuple[int, int, int, str]:
    candidate = Path(path_or_name)
    match = _MASK_FRAME_RE.match(candidate.name)
    if match:
        return (0, int(match.group(1)), int(match.group(2)), candidate.as_posix().lower())
    numbers = re.findall(r'\d+', candidate.stem)
    frame_id = int(numbers[-1]) if numbers else 0
    return (1 if numbers else 2, frame_id, 0, candidate.as_posix().lower())


def _read_image_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Read images robustly on Windows paths with non-ASCII characters."""
    if os.name == 'nt':
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size > 0:
                decoded = cv2.imdecode(data, flags)
                if decoded is not None:
                    return decoded
        except Exception:
            pass
    try:
        img = cv2.imread(str(path), flags)
    except Exception:
        img = None
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _write_image_unicode(path: Path, image: np.ndarray, params: Optional[List[int]] = None) -> bool:
    """Write images robustly on Windows paths with non-ASCII characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or '.png'
    encode_ext = suffix if suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'} else '.png'
    try:
        if params:
            ok, encoded = cv2.imencode(encode_ext, image, params)
        else:
            ok, encoded = cv2.imencode(encode_ext, image)
        if ok:
            encoded.tofile(str(path))
            return True
    except Exception:
        pass
    try:
        if params:
            return bool(cv2.imwrite(str(path), image, params))
        return bool(cv2.imwrite(str(path), image))
    except Exception:
        return False


def _resolve_sam3_checkpoint(model_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Resolve SAM3 checkpoint path locally or return candidate for download."""
    if model_path:
        p = Path(model_path).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)

    candidates = [
        Path('sam3.pt'),
        Path('models') / 'sam3.pt',
        Path('downloads') / 'sam3.pt',
        Path('downloads') / 'sam3cpp' / 'models' / 'sam3.pt',
        Path.home() / '.cache' / 'sam3' / 'sam3.pt',
        Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--facebook--sam3' / 'snapshots' / 'main' / 'sam3.pt',
    ]

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand.resolve())

    return None


class DARTMasker:
    """
    High-performance open-vocabulary multi-class masking using DART (Detect Anything in Real Time).
    Converts SAM3 into an open-vocabulary instance segmenter for RealityScan and COLMAP photogrammetry.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.70,
        use_gpu: bool = True,
        fast: bool = True,
        single_pass: bool = False,
        resolution: int = 1008,
        text_cache: Optional[str] = None,
        feather_radius: int = 8,
        morph_radius: int = 0,
        alpha_export: bool = False,
        alpha_only: bool = False,
        enable_refinement: bool = True,
        refine_sky_only: bool = True,
        seam_aware_refinement: bool = True,
        edge_sharpen_strength: float = 0.75,
        fisheye_circle_mask_enabled: bool = False,
        fisheye_circle_mask_radius_percent: float = 94.0,
    ):
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.use_gpu = bool(use_gpu)
        self.fast = bool(fast)
        self.single_pass = bool(single_pass)
        self.resolution = int(resolution) if int(resolution) % 14 == 0 else 1008
        self.text_cache = text_cache

        # Post-processing / refinement settings
        self.feather_radius = int(feather_radius)
        self.morph_radius = int(morph_radius)
        self.alpha_export = bool(alpha_export)
        self.alpha_only = bool(alpha_only)
        self.enable_refinement = bool(enable_refinement)
        self.refine_sky_only = bool(refine_sky_only)
        self.seam_aware_refinement = bool(seam_aware_refinement)
        self.edge_sharpen_strength = float(edge_sharpen_strength)

        # Fisheye circular crop settings
        self.fisheye_circle_mask_enabled = bool(fisheye_circle_mask_enabled)
        self.fisheye_circle_mask_radius_percent = clamp_radius_percent(fisheye_circle_mask_radius_percent)

        # Cancellation state for batch operations
        self.cancelled = False

        # Category enabling
        self.enabled_categories: Dict[str, bool] = {
            'persons': True,
            'vehicles': True,
            'rigs': True,
            'personal_objects': True,
            'animals': False,
            'sky': False,
        }
        self.custom_prompts: List[str] = []

        # Internal model and predictor handles
        self.device = self._select_device(use_gpu)
        self._model = None
        self._predictor = None
        self._cached_prompt_list: Optional[List[str]] = None

    def _select_device(self, use_gpu: bool) -> str:
        """Select execution device (cuda / cpu)."""
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"DARTMasker using CUDA device: {device_name}")
                    return "cuda"
            except Exception as exc:
                logger.warning(f"CUDA check failed ({exc}), falling back to CPU.")
        logger.info("DARTMasker using CPU device.")
        return "cpu"

    # -----------------------------------------------------------------------
    # Category and Prompt Configuration
    # -----------------------------------------------------------------------

    def set_enabled_categories(self, categories_dict: Dict[str, bool]) -> None:
        """Update enabled prompt categories."""
        if not categories_dict:
            return
        for key, val in categories_dict.items():
            norm_key = key.lower().strip()
            if norm_key in self.enabled_categories:
                self.enabled_categories[norm_key] = bool(val)
        self._invalidate_prompts()

    def set_custom_prompts(self, prompts: Union[str, Sequence[str]]) -> None:
        """Set custom open-vocabulary text prompts (comma-separated or list)."""
        if isinstance(prompts, str):
            items = [p.strip() for p in prompts.split(',') if p.strip()]
        elif isinstance(prompts, (list, tuple)):
            items = [str(p).strip() for p in prompts if str(p).strip()]
        else:
            items = []
        self.custom_prompts = items
        self._invalidate_prompts()

    def _invalidate_prompts(self) -> None:
        self._cached_prompt_list = None

    def get_active_prompts(self) -> List[str]:
        """Collect all active unique prompt strings from enabled categories + custom prompts."""
        if self._cached_prompt_list is not None:
            return self._cached_prompt_list

        collected: List[str] = []
        for cat, enabled in self.enabled_categories.items():
            if enabled and cat in DEFAULT_DART_CATEGORY_PROMPTS:
                collected.extend(DEFAULT_DART_CATEGORY_PROMPTS[cat])

        collected.extend(self.custom_prompts)

        # Deduplicate while preserving order
        seen = set()
        unique_prompts = []
        for p in collected:
            norm = p.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                unique_prompts.append(p.strip())

        if not unique_prompts:
            # Default fallback to person if nothing was enabled
            unique_prompts = ['person']

        self._cached_prompt_list = unique_prompts
        return self._cached_prompt_list

    # -----------------------------------------------------------------------
    # Lazy Model Initialization
    # -----------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazily load the SAM3 image model and instantiate DART predictor."""
        if self._predictor is not None:
            return

        import torch
        from .dart.model_builder import build_sam3_image_model, load_pruned_config, build_pruned_sam3_image_model
        from .dart.model.sam3_multiclass import Sam3MultiClassPredictor
        from .dart.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

        ckpt_path = _resolve_sam3_checkpoint(self.model_path)
        logger.info(f"Loading DART SAM3 model (checkpoint: {ckpt_path or 'auto-download'}) on {self.device}...")

        pruned_config = load_pruned_config(ckpt_path) if ckpt_path else None
        if pruned_config is not None:
            logger.info(f"Detected pruned SAM3 checkpoint: {pruned_config}")
            self._model = build_pruned_sam3_image_model(
                checkpoint_path=ckpt_path,
                pruning_config=pruned_config,
                device=self.device,
                eval_mode=True,
            )
        else:
            self._model = build_sam3_image_model(
                checkpoint_path=ckpt_path,
                device=self.device,
                eval_mode=True,
            )

        if self.fast:
            logger.info("Initializing Sam3MultiClassPredictorFast (batched + FP16 + presence early-exit)")
            self._predictor = Sam3MultiClassPredictorFast(
                self._model,
                device=self.device,
                resolution=self.resolution,
                use_fp16=(self.device == "cuda"),
                presence_threshold=0.05,
                single_pass=self.single_pass,
                detection_only=False,
            )
        else:
            logger.info("Initializing standard Sam3MultiClassPredictor (per-class sequential)")
            self._predictor = Sam3MultiClassPredictor(
                self._model,
                device=self.device,
                resolution=self.resolution,
                detection_only=False,
            )

        # Set initial class prompts
        prompts = self.get_active_prompts()
        logger.info(f"DART initialized with {len(prompts)} active classes: {prompts}")
        self._predictor.set_classes(prompts)

    def _sync_classes_if_needed(self) -> None:
        """Ensure predictor has the latest prompts cached."""
        self._ensure_model_loaded()
        active = self.get_active_prompts()
        if getattr(self._predictor, '_class_names', None) != active:
            logger.info(f"Updating DART classes to {len(active)} active prompts: {active}")
            self._predictor.set_classes(active)

    # -----------------------------------------------------------------------
    # Mask Generation (Single Image)
    # -----------------------------------------------------------------------

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, PIL.Image.Image],
    ) -> Dict:
        """Run DART detection + segmentation on a single image and return predictions dict."""
        self._sync_classes_if_needed()

        if isinstance(image_input, (str, Path)):
            img_bgr = _read_image_unicode(Path(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(img_rgb)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input
            pil_image = PIL.Image.fromarray(img_rgb)
        elif isinstance(image_input, PIL.Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        state = self._predictor.set_image(pil_image)
        results = self._predictor.predict(
            state,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )
        return results

    def generate_mask(
        self,
        image_path: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Generate a RealityScan/COLMAP compatible mask (0=remove, 255=keep) for an image.

        Args:
            image_path: Path to image or image numpy array (BGR/RGB).
            output_path: Optional destination to save the mask PNG.

        Returns:
            2D uint8 numpy array of shape (H, W) where 0=remove, 255=keep.
        """
        # Load image array for dimension inspection and refinement
        if isinstance(image_path, (str, Path)):
            path_obj = Path(image_path)
            img_bgr = _read_image_unicode(path_obj)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            orig_h, orig_w = img_bgr.shape[:2]
            results = self.predict_image(path_obj)
        else:
            img_bgr = image_path
            orig_h, orig_w = img_bgr.shape[:2]
            results = self.predict_image(image_path)

        # Composite detected instance masks (detected regions = 255, background = 0)
        detected_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if results.get("masks") is not None and len(results["masks"]) > 0:
            masks_tensor = results["masks"]
            for i in range(len(masks_tensor)):
                mask_np = masks_tensor[i].detach().cpu().numpy().astype(bool)
                if mask_np.shape != (orig_h, orig_w):
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                detected_mask[mask_np] = 255

        # Morphological operations if requested
        if self.morph_radius != 0 and np.any(detected_mask > 0):
            k_size = abs(int(self.morph_radius)) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            if self.morph_radius > 0:
                # Expand detected regions (dilate detected)
                detected_mask = cv2.dilate(detected_mask, kernel)
            else:
                # Shrink detected regions (erode detected)
                detected_mask = cv2.erode(detected_mask, kernel)

        # Boundary refinement (edge sharpening, feathering, 360 wrap)
        if self.enable_refinement and np.any(detected_mask > 0):
            refine_settings = MaskRefinementSettings(
                enabled=True,
                edge_band_radius=self.feather_radius,
                sharpen_strength=self.edge_sharpen_strength,
                seam_aware=self.seam_aware_refinement,
            )
            detected_mask = refine_detected_mask(img_bgr, detected_mask, refine_settings)

        # Invert to RealityScan / 360toolkit convention:
        # 0 (Black) = detected/remove
        # 255 (White) = keep background
        final_keep_mask = np.where(detected_mask > 127, MASK_VALUE_REMOVE, MASK_VALUE_KEEP).astype(np.uint8)

        # Apply circular fisheye keep mask if enabled
        if self.fisheye_circle_mask_enabled:
            circle_keep = build_fisheye_keep_mask(
                width=orig_w,
                height=orig_h,
                radius_percent=self.fisheye_circle_mask_radius_percent,
            )
            final_keep_mask = combine_with_keep_mask(final_keep_mask, circle_keep)

        # Save to disk if requested
        if output_path:
            out_p = Path(output_path)
            _write_image_unicode(out_p, final_keep_mask)

        return final_keep_mask

    def process_image_to_alpha(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate an RGBA image where detected objects have alpha=0 (transparent) and kept areas alpha=255.

        Args:
            image_path: Path to source image.
            output_path: Optional destination to save the alpha PNG.

        Returns:
            RGBA uint8 numpy array of shape (H, W, 4).
        """
        path_obj = Path(image_path)
        img_bgr = _read_image_unicode(path_obj)
        if img_bgr is None:
            return None

        # Ensure 3-channel BGR
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]

        keep_mask = self.generate_mask(path_obj)
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = keep_mask

        if output_path:
            _write_image_unicode(Path(output_path), rgba)

        return rgba

    # -----------------------------------------------------------------------
    # Batch Processing
    # -----------------------------------------------------------------------

    def process_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_visualization: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
        output_mode: Optional[str] = None,
    ) -> Dict:
        """
        Process all images in a directory, creating masks and/or alpha cutouts.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory to save generated masks/cutouts.
            save_visualization: Whether to save color visualization overlays.
            progress_callback: Callback receiving (current_idx, total_count, current_filename).
            cancellation_check: Optional callable returning True to abort batch.
            cancel_flag: Alias for cancellation_check.
            output_mode: 'masks_only', 'alpha_only', or 'both'.

        Returns:
            Summary dict with 'success', 'successful', 'masks_created', 'total', 'skipped', 'failed', 'cancelled', 'output_dir'.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        self.cancelled = False
        cancel_fn = cancellation_check or cancel_flag
        mode = (output_mode or ('alpha_only' if self.alpha_only else ('both' if self.alpha_export else 'masks_only'))).lower()

        # Discover image files
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        image_files = [
            f for f in in_path.rglob('*')
            if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith('.')
        ]
        image_files.sort(key=_mask_input_sort_key)

        total_files = len(image_files)
        if total_files == 0:
            logger.warning(f"No image files found in {input_dir}")
            return {
                'success': True,
                'successful': 0,
                'masks_created': 0,
                'total': 0,
                'skipped': 0,
                'failed': 0,
                'cancelled': False,
                'output_dir': str(out_path),
            }

        logger.info(f"DARTMasker starting batch of {total_files} images (mode: {mode})...")
        self._ensure_model_loaded()
        self._sync_classes_if_needed()

        masks_created = 0
        skipped = 0
        failed = 0
        was_cancelled = False

        for idx, img_file in enumerate(image_files):
            if self.cancelled or (cancel_fn and cancel_fn()):
                logger.info("Batch processing cancelled by user.")
                was_cancelled = True
                break

            filename = img_file.name
            if progress_callback:
                progress_callback(idx + 1, total_files, filename)

            try:
                # Resolve output paths
                mask_dest = build_related_output_path(
                    input_path=img_file,
                    input_root=in_path,
                    output_root=out_path,
                    suffix='_mask.png',
                )
                alpha_dest = build_related_output_path(
                    input_path=img_file,
                    input_root=in_path,
                    output_root=out_path,
                    suffix='_alpha.png',
                )

                if mode in {'masks_only', 'both'}:
                    self.generate_mask(img_file, output_path=mask_dest)

                if mode in {'alpha_only', 'both'}:
                    self.process_image_to_alpha(img_file, output_path=alpha_dest)

                masks_created += 1

            except Exception as exc:
                logger.error(f"Failed to process image {img_file}: {exc}", exc_info=True)
                failed += 1

        return {
            'success': (failed == 0) and not was_cancelled,
            'successful': masks_created,
            'masks_created': masks_created,
            'total': total_files,
            'skipped': skipped,
            'failed': failed,
            'cancelled': was_cancelled,
            'output_dir': str(out_path),
        }
