r"""
test_a1_blend_stitch.py - Prototype blended stitch for Antigravity A1.

This avoids FFmpeg's dfisheye hard hemisphere split by:
  1. Projecting each fisheye stream independently to equirectangular.
  2. Using a feathered latitude blend around the equator.
  3. Sweeping a small set of FOV and yaw offsets for visual review.

Output: C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\a1_blend_experiments\
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


A1_INSV = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\VID_20260327_162728_005_Antigravity_A1_Sample.insv")
OUT_BASE = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\a1_blend_experiments")
FFMPEG = "ffmpeg"
THUMB_W = 960
THUMB_H = 480


def run_ffmpeg_projection(stream_idx: int, pitch: int, yaw: int, fov: int, out_file: Path) -> None:
    cmd = [
        FFMPEG,
        "-y",
        "-i", str(A1_INSV),
        "-filter_complex",
        (
            f"[0:v:{stream_idx}]v360=fisheye:equirect:ih_fov={fov}:iv_fov={fov}"
            f":pitch={pitch}:yaw={yaw},scale={THUMB_W}:{THUMB_H}[out]"
        ),
        "-map", "[out]",
        "-frames:v", "1",
        str(out_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:])


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    scaled = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def match_overlap_exposure(
    reference_img: np.ndarray,
    candidate_img: np.ndarray,
    overlap_mask: np.ndarray,
) -> np.ndarray:
    adjusted = candidate_img.astype(np.float32).copy()
    valid_pixels = overlap_mask > 0
    if not np.any(valid_pixels):
        return adjusted

    ref_samples = reference_img[valid_pixels].astype(np.float32)
    cand_samples = candidate_img[valid_pixels].astype(np.float32)

    ref_mean = ref_samples.mean(axis=0)
    cand_mean = cand_samples.mean(axis=0)
    scale = np.clip(ref_mean / np.maximum(cand_mean, 1.0), 0.85, 1.15)
    adjusted *= scale.reshape(1, 1, 3)
    return adjusted.clip(0, 255)


def build_blend(up_img: np.ndarray, down_img: np.ndarray, band_top: float, band_bottom: float) -> np.ndarray:
    height = up_img.shape[0]
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)
    down_weight_1d = smoothstep(band_top, band_bottom, rows)
    down_weight = np.repeat(down_weight_1d[:, None], up_img.shape[1], axis=1)
    up_weight = 1.0 - down_weight

    up_valid = (up_img.mean(axis=2) > 6).astype(np.float32)
    down_valid = (down_img.mean(axis=2) > 6).astype(np.float32)

    overlap_mask = (up_valid > 0) & (down_valid > 0) & (down_weight > 0.0) & (up_weight > 0.0)
    matched_down = match_overlap_exposure(up_img, down_img, overlap_mask)

    up_weight *= up_valid
    down_weight *= down_valid

    total = up_weight + down_weight
    fallback_up = (total == 0) & (up_valid > 0)
    fallback_down = (total == 0) & (down_valid > 0)
    up_weight[fallback_up] = 1.0
    down_weight[fallback_down] = 1.0
    total = up_weight + down_weight
    total[total == 0] = 1.0

    blended = (
        up_img.astype(np.float32) * up_weight[..., None]
        + matched_down * down_weight[..., None]
    ) / total[..., None]

    return blended.clip(0, 255).astype(np.uint8)


def write_gallery(results: list[tuple[str, Path]]) -> None:
    gallery_path = OUT_BASE / "gallery.html"
    with gallery_path.open("w", encoding="utf-8") as handle:
        handle.write("<html><head><title>A1 Blend Experiments</title></head><body>\n")
        handle.write("<h1>A1 Blend Experiments</h1>\n")
        handle.write("<p>Goal: reduce the hard equator seam by blending separate up/down lens projections.</p>\n")
        for label, image_path in results:
            rel = os.path.relpath(image_path, OUT_BASE).replace("\\", "/")
            handle.write('<div style="display:inline-block;margin:8px;text-align:center">\n')
            handle.write(
                f'  <img src="{rel}" width="{THUMB_W}" height="{THUMB_H}" style="border:2px solid #ccc"/><br/>\n'
            )
            handle.write(f"  <small>{label}</small>\n")
            handle.write("</div>\n")
        handle.write("</body></html>\n")


def main() -> None:
    if not A1_INSV.exists():
        raise SystemExit(f"Missing input: {A1_INSV}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_BASE / "cache"
    cache_dir.mkdir(exist_ok=True)

    projection_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    results: list[tuple[str, Path]] = []

    variants = []
    for fov in [185, 190, 200]:
        for sky_yaw in [0, -15, 15]:
            for ground_yaw in [0, -15, 15]:
                for band_top, band_bottom in [(0.49, 0.51), (0.48, 0.52), (0.46, 0.54)]:
                    label = (
                        f"blend_fov{fov}_skyY{sky_yaw:+d}_groundY{ground_yaw:+d}_"
                        f"band{int(band_top * 100):02d}-{int(band_bottom * 100):02d}"
                    )
                    variants.append((label, fov, sky_yaw, ground_yaw, band_top, band_bottom))

    print(f"Running {len(variants)} blend variants -> {OUT_BASE}")

    for label, fov, sky_yaw, ground_yaw, band_top, band_bottom in variants:
        keys = {
            "up": (1, -90, sky_yaw, fov),
            "down": (0, 90, ground_yaw, fov),
        }

        for key in keys.values():
            if key in projection_cache:
                continue
            stream_idx, pitch, yaw, current_fov = key
            cache_file = cache_dir / f"s{stream_idx}_p{pitch:+d}_y{yaw:+d}_fov{current_fov}.png"
            run_ffmpeg_projection(stream_idx, pitch, yaw, current_fov, cache_file)
            image = cv2.imread(str(cache_file), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read projection: {cache_file}")
            projection_cache[key] = image

        up_img = projection_cache[keys["up"]]
        down_img = projection_cache[keys["down"]]
        blended = build_blend(up_img, down_img, band_top, band_bottom)

        out_dir = OUT_BASE / label
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "frame.jpg"
        cv2.imwrite(str(out_file), blended, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        results.append((label, out_file))
        print(f"  [OK] {label}")

    write_gallery(results)
    print(f"HTML gallery: {OUT_BASE / 'gallery.html'}")


if __name__ == "__main__":
    main()