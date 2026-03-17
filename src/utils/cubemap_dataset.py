from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTS = {".png", ".tif", ".tiff", ".bmp"}

_CUBEMAP_IMAGE_RE = re.compile(
    r"^frame_(\d+)_(front|back|left|right|top|bottom)(\.[^.]+)$",
    re.IGNORECASE,
)
_CUBEMAP_MASK_RE = re.compile(
    r"^frame_(\d+)_(front|back|left|right|top|bottom)_mask(\.[^.]+)$",
    re.IGNORECASE,
)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def is_mask_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MASK_EXTS


def is_cubemap_image_name(name: str) -> bool:
    return _CUBEMAP_IMAGE_RE.match(name) is not None


def is_cubemap_mask_name(name: str) -> bool:
    return _CUBEMAP_MASK_RE.match(name) is not None


def list_cubemap_images(images_dir: Path) -> List[Path]:
    return sorted(
        [path for path in images_dir.iterdir() if is_image_file(path) and is_cubemap_image_name(path.name)]
    )


def list_cubemap_masks(masks_dir: Path) -> List[Path]:
    return sorted(
        [path for path in masks_dir.iterdir() if is_mask_file(path) and is_cubemap_mask_name(path.name)]
    )


def expected_mask_name(image_name: str) -> str:
    match = _CUBEMAP_IMAGE_RE.match(image_name)
    if match is None:
        raise ValueError(f"Not a cubemap image name: {image_name}")
    frame_id, face_name, _ = match.groups()
    return f"frame_{frame_id}_{face_name.lower()}_mask.png"


def marker_path(dataset_root: Path) -> Path:
    return dataset_root / ".orientation_fixed_180.json"


def dataset_orientation_fixed(dataset_root: Path) -> bool:
    return marker_path(dataset_root).exists()


def _rotate_in_place(path: Path) -> None:
    with Image.open(path) as image:
        rotated = ImageOps.exif_transpose(image).transpose(Image.Transpose.ROTATE_180)
        save_kwargs: Dict[str, object] = {}
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": 95, "subsampling": 0})
        rotated.save(path, **save_kwargs)


def rotate_dataset_orientation_180(
    dataset_root: Path,
    images_subdir: str = "perspective_views",
    masks_subdir: str = "masks",
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / images_subdir
    masks_dir = dataset_root / masks_subdir
    marker = marker_path(dataset_root)

    if marker.exists() and not force:
        return {
            "success": True,
            "skipped": True,
            "reason": "orientation already fixed",
            "images_rotated": 0,
            "masks_rotated": 0,
            "marker": str(marker),
        }

    if not images_dir.exists():
        return {
            "success": False,
            "error": f"Missing images directory: {images_dir}",
        }

    image_files = list_cubemap_images(images_dir)
    if not image_files:
        return {
            "success": False,
            "error": f"No cubemap images found in {images_dir}",
        }

    mask_files: List[Path] = []
    if masks_dir.exists():
        mask_files = list_cubemap_masks(masks_dir)

    if not dry_run:
        for image_path in image_files:
            _rotate_in_place(image_path)
        for mask_path in mask_files:
            _rotate_in_place(mask_path)

        payload = {
            "rotation": 180,
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "images_rotated": len(image_files),
            "masks_rotated": len(mask_files),
        }
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "success": True,
        "skipped": False,
        "images_rotated": len(image_files),
        "masks_rotated": len(mask_files),
        "marker": str(marker),
        "dry_run": dry_run,
    }


def collect_frame_prefixes(image_paths: Iterable[Path]) -> List[str]:
    prefixes = sorted({path.name.rsplit("_", 1)[0] for path in image_paths})
    return prefixes


def copy_cubemap_subset(
    source_dataset_root: Path,
    target_dataset_root: Path,
    max_frames: int | None = None,
    include_masks: bool = True,
) -> Dict[str, object]:
    source_dataset_root = Path(source_dataset_root)
    target_dataset_root = Path(target_dataset_root)
    source_images_dir = source_dataset_root / "perspective_views"
    source_masks_dir = source_dataset_root / "masks"
    target_images_dir = target_dataset_root / "perspective_views"
    target_masks_dir = target_dataset_root / "masks"

    image_files = list_cubemap_images(source_images_dir)
    frame_prefixes = collect_frame_prefixes(image_files)
    if max_frames is not None:
        frame_prefixes = frame_prefixes[: max(1, max_frames)]
    selected_prefixes = set(frame_prefixes)

    target_images_dir.mkdir(parents=True, exist_ok=True)
    if include_masks:
        target_masks_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_masks = 0

    for image_path in image_files:
        if image_path.name.rsplit("_", 1)[0] not in selected_prefixes:
            continue
        target_path = target_images_dir / image_path.name
        target_path.write_bytes(image_path.read_bytes())
        copied_images += 1

    if include_masks and source_masks_dir.exists():
        for mask_path in list_cubemap_masks(source_masks_dir):
            frame_prefix = mask_path.name.rsplit("_", 2)[0]
            if frame_prefix not in selected_prefixes:
                continue
            target_path = target_masks_dir / mask_path.name
            target_path.write_bytes(mask_path.read_bytes())
            copied_masks += 1

    return {
        "frames": len(selected_prefixes),
        "images": copied_images,
        "masks": copied_masks,
        "target_dataset_root": str(target_dataset_root),
    }