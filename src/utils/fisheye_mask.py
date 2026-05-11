from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Rect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class FisheyeCircleMaskSettings:
    enabled: bool = False
    radius_percent: int = 94


def is_dual_fisheye_method(method: str | None) -> bool:
    return str(method or '').strip().lower() in {'ffmpeg_dual_lens', 'ffmpeg_lens1', 'ffmpeg_lens2'}


def clamp_radius_percent(value: int | float | None, default: int = 94) -> int:
    try:
        numeric = int(round(float(value)))
    except Exception:
        numeric = default
    return max(40, min(100, numeric))


def infer_fisheye_regions(image_shape: Sequence[int]) -> List[Rect]:
    if len(image_shape) < 2:
        return []

    height = int(image_shape[0])
    width = int(image_shape[1])
    if width <= 0 or height <= 0:
        return []

    tolerance = max(4, int(round(min(width, height) * 0.08)))

    half_width = width // 2
    if width >= int(height * 1.9) and abs(half_width - height) <= tolerance:
        return [
            (0, 0, half_width, height),
            (width - half_width, 0, half_width, height),
        ]

    half_height = height // 2
    if height >= int(width * 1.9) and abs(half_height - width) <= tolerance:
        return [
            (0, 0, width, half_height),
            (0, height - half_height, width, half_height),
        ]

    if abs(width - height) <= tolerance:
        return [(0, 0, width, height)]

    return []


def build_mask_from_regions(
    image_shape: Sequence[int],
    regions: Iterable[Rect],
    radius_percent: int | float,
) -> np.ndarray:
    height = int(image_shape[0])
    width = int(image_shape[1])
    keep_mask = np.zeros((height, width), dtype=np.uint8)
    clamped_percent = clamp_radius_percent(radius_percent)

    for x, y, region_width, region_height in regions:
        if region_width <= 0 or region_height <= 0:
            continue
        radius_limit = min(region_width, region_height) / 2.0
        radius = max(1, int(round(radius_limit * (clamped_percent / 100.0))))
        center_x = int(round(x + (region_width / 2.0)))
        center_y = int(round(y + (region_height / 2.0)))
        cv2.circle(keep_mask, (center_x, center_y), radius, 255, -1, lineType=cv2.LINE_AA)

    return keep_mask


def build_fisheye_keep_mask(
    image_shape: Sequence[int],
    radius_percent: int | float,
    *,
    regions: Iterable[Rect] | None = None,
) -> np.ndarray | None:
    selected_regions = list(regions) if regions is not None else infer_fisheye_regions(image_shape)
    if not selected_regions:
        return None
    return build_mask_from_regions(image_shape, selected_regions, radius_percent)


def combine_with_keep_mask(existing_mask: np.ndarray, keep_mask: np.ndarray | None) -> np.ndarray:
    if keep_mask is None:
        return existing_mask
    if keep_mask.shape[:2] != existing_mask.shape[:2]:
        keep_mask = cv2.resize(keep_mask, (existing_mask.shape[1], existing_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(existing_mask, keep_mask)


def overlay_fisheye_mask(
    image: np.ndarray,
    radius_percent: int | float,
    *,
    regions: Iterable[Rect] | None = None,
    overlay_color: Tuple[int, int, int] = (48, 64, 255),
    overlay_alpha: float = 0.28,
) -> np.ndarray:
    if image is None:
        return image

    keep_mask = build_fisheye_keep_mask(image.shape, radius_percent, regions=regions)
    if keep_mask is None:
        return image.copy()

    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        base = image.copy()

    result = base.copy()
    overlay = np.full_like(base, overlay_color, dtype=np.uint8)
    outside = keep_mask == 0
    if np.any(outside):
        blended = cv2.addWeighted(base, 1.0 - overlay_alpha, overlay, overlay_alpha, 0.0)
        result[outside] = blended[outside]

    edges = cv2.Canny(keep_mask, 50, 150)
    result[edges > 0] = (255, 255, 255)
    return result


def apply_circle_mask_to_image(
    image: np.ndarray,
    radius_percent: int | float,
    *,
    regions: Iterable[Rect] | None = None,
    force_alpha: bool = True,
    fill_value: int = 0,
) -> np.ndarray:
    keep_mask = build_fisheye_keep_mask(image.shape, radius_percent, regions=regions)
    if keep_mask is None:
        return image.copy()

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image.copy()

    masked_bgr = bgr.copy()
    masked_bgr[keep_mask == 0] = fill_value

    if force_alpha or (image.ndim == 3 and image.shape[2] == 4):
        bgra = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = keep_mask
        return bgra

    return masked_bgr