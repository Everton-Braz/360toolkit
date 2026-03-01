"""
Shared OpenCV colour-correction pipeline.

Matches the Insta360-style sliders exposed in MediaProcessingPanel.
Used by both the live preview (preview_panels.py) and the batch
post-processing step in batch_orchestrator.py so the result is
always identical regardless of how the SDK CLI behaves.

Parameter ranges (matching SDK):
  exposure, highlights, shadows, contrast, brightness,
  blackpoint, saturation, vibrance, warmth, tint  →  -100 .. +100
  definition                                       →     0 .. +100
"""

import cv2
import numpy as np


def apply_color_corrections(img: np.ndarray, opts: dict) -> np.ndarray:
    """
    Apply Insta360-style colour corrections to a BGR uint8 numpy image.

    Returns the corrected image as a uint8 BGR array.
    If *opts* is empty / None, returns the original image unchanged.
    """
    if not opts:
        return img

    # Check whether any non-zero value is present (skip for pure-neutral state)
    color_keys = ('exposure', 'highlights', 'shadows', 'contrast', 'brightness',
                  'blackpoint', 'saturation', 'vibrance', 'warmth', 'tint', 'definition')
    if not any(opts.get(k, 0) for k in color_keys):
        return img

    f = img.astype(np.float32) / 255.0

    # ── Exposure: photographic stops [-100..100] ─────────────────────────────
    exp = opts.get('exposure', 0)
    if exp:
        f *= 2.0 ** (exp / 100.0)

    # ── Brightness: linear lift / cut ───────────────────────────────────────
    br = opts.get('brightness', 0)
    if br:
        f += br / 400.0

    # ── Contrast: S-curve around mid-grey ────────────────────────────────────
    con = opts.get('contrast', 0)
    if con:
        f = (f - 0.5) * (1.0 + con / 100.0) + 0.5

    # ── Highlights: compress / expand bright pixels ──────────────────────────
    hl = opts.get('highlights', 0)
    if hl:
        mask = np.clip((f - 0.5) * 2.0, 0.0, 1.0)
        f += mask * (hl / 100.0) * 0.25

    # ── Shadows: lift / cut dark pixels ─────────────────────────────────────
    sh = opts.get('shadows', 0)
    if sh:
        mask = np.clip((0.5 - f) * 2.0, 0.0, 1.0)
        f += mask * (sh / 100.0) * 0.25

    # ── Black point ──────────────────────────────────────────────────────────
    bp = opts.get('blackpoint', 0)
    if bp:
        f = np.where(f < 0.5, f + bp / 500.0, f)

    f = np.clip(f, 0.0, 1.0)

    # ── Saturation & Vibrance (HSV) ──────────────────────────────────────────
    sat = opts.get('saturation', 0)
    vib = opts.get('vibrance', 0)
    if sat or vib:
        u8 = (f * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        if sat:
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat / 100.0), 0.0, 255.0)
        if vib:
            sat_norm = hsv[..., 1] / 255.0
            factor = 1.0 + (vib / 100.0) * (1.0 - sat_norm) * 0.8
            hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 255.0)
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # ── Warmth: colour temperature shift ─────────────────────────────────────
    # +100 = warm / orange, -100 = cool / blue
    wm = opts.get('warmth', 0)
    if wm:
        delta = wm / 500.0
        f[..., 0] = f[..., 0] - delta   # blue  channel (BGR index 0)
        f[..., 2] = f[..., 2] + delta   # red   channel (BGR index 2)

    # ── Tint: +100 = green, -100 = magenta ──────────────────────────────────
    ti = opts.get('tint', 0)
    if ti:
        delta = ti / 500.0
        f[..., 1] = f[..., 1] + delta          # green boost
        f[..., 0] = f[..., 0] - delta * 0.4    # slight blue  suppress for magenta
        f[..., 2] = f[..., 2] - delta * 0.4    # slight red   suppress for magenta

    result = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    # ── Definition (sharpness): 0..100 ───────────────────────────────────────
    defi = opts.get('definition', 0)
    if defi > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.5)
        amount  = defi / 100.0
        result  = cv2.addWeighted(result, 1.0 + amount, blurred, -amount, 0)
        result  = np.clip(result, 0, 255).astype(np.uint8)

    return result
