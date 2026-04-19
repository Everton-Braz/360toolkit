from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskRefinementSettings:
    enabled: bool = False
    edge_band_radius: int = 8
    sharpen_strength: float = 0.75
    seam_aware: bool = True
    guidance_filter: str = 'bilateral'


def is_likely_equirectangular(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    aspect = width / float(height)
    return 1.85 <= aspect <= 2.15


def refine_detected_mask(
    image_bgr: np.ndarray,
    detected_mask: np.ndarray,
    settings: MaskRefinementSettings,
) -> np.ndarray:
    """
    Refine a detected-region mask (255=detected/remove, 0=keep).

    The refinement is designed for offline-quality cleanup of SAM3 masks,
    especially sky masks. It uses a seam-aware wrap pad for equirectangular
    images, GrabCut initialized from the current mask for image-guided edge
    refinement, and a local unsharp pass on the boundary band.
    """
    binary_mask = _ensure_binary_mask(detected_mask)
    if not settings.enabled or settings.edge_band_radius <= 0:
        return binary_mask

    if image_bgr is None or image_bgr.size == 0:
        return binary_mask

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    if image_bgr.shape[:2] != binary_mask.shape[:2]:
        raise ValueError('image and mask dimensions must match for refinement')

    radius = max(1, int(settings.edge_band_radius))
    seam_aware = bool(settings.seam_aware and is_likely_equirectangular(binary_mask.shape[1], binary_mask.shape[0]))
    pad = max(radius * 2, 8) if seam_aware else 0

    work_image = _wrap_pad_horizontal(image_bgr, pad) if pad else image_bgr
    work_mask = _wrap_pad_horizontal(binary_mask, pad) if pad else binary_mask

    refined = _grabcut_refine(work_image, work_mask, radius, settings.guidance_filter)
    sharpened = _sharpen_boundary_band(refined, radius, settings.sharpen_strength)

    if pad:
        sharpened = sharpened[:, pad:-pad]

    return _ensure_binary_mask(sharpened)


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError('mask_refinement expects a single-channel mask')
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _wrap_pad_horizontal(image: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return image
    width = image.shape[1]
    if width < 2:
        return image
    pad = min(pad, max(1, width - 1))
    return np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)


def _prepare_guidance_image(image_bgr: np.ndarray, guidance_filter: str) -> np.ndarray:
    if guidance_filter == 'bilateral':
        return cv2.bilateralFilter(image_bgr, d=5, sigmaColor=30, sigmaSpace=30)
    return image_bgr


def _grabcut_refine(
    image_bgr: np.ndarray,
    detected_mask: np.ndarray,
    radius: int,
    guidance_filter: str,
) -> np.ndarray:
    detected = (detected_mask > 127).astype(np.uint8)
    if not np.any(detected):
        return detected_mask
    if np.all(detected):
        return detected_mask

    fg_dist = cv2.distanceTransform(detected, cv2.DIST_L2, 5)
    bg_dist = cv2.distanceTransform(1 - detected, cv2.DIST_L2, 5)

    gc_mask = np.full(detected.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[detected > 0] = cv2.GC_PR_FGD

    core_radius = max(1, radius // 2)
    gc_mask[bg_dist > core_radius] = cv2.GC_BGD
    gc_mask[fg_dist > core_radius] = cv2.GC_FGD

    guide = _prepare_guidance_image(image_bgr, guidance_filter)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(guide, gc_mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)

    probability = np.zeros(gc_mask.shape, dtype=np.float32)
    probability[gc_mask == cv2.GC_BGD] = 0.0
    probability[gc_mask == cv2.GC_PR_BGD] = 0.25
    probability[gc_mask == cv2.GC_PR_FGD] = 0.75
    probability[gc_mask == cv2.GC_FGD] = 1.0

    return np.where(probability >= 0.5, 255, 0).astype(np.uint8)


def _sharpen_boundary_band(mask: np.ndarray, radius: int, strength: float) -> np.ndarray:
    detected = (mask > 127).astype(np.uint8)
    fg_dist = cv2.distanceTransform(detected, cv2.DIST_L2, 5)
    bg_dist = cv2.distanceTransform(1 - detected, cv2.DIST_L2, 5)
    band = (fg_dist <= radius) | (bg_dist <= radius)

    probability = detected.astype(np.float32)
    sigma = max(0.75, radius * 0.35)
    blurred = cv2.GaussianBlur(probability, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = np.clip(probability + (probability - blurred) * float(max(0.0, strength)), 0.0, 1.0)
    probability[band] = sharpened[band]

    refined = np.where(probability >= 0.5, 255, 0).astype(np.uint8)
    cleanup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, cleanup)
    return refined