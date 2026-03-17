from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.cubemap_dataset import (
    copy_cubemap_subset,
    dataset_orientation_fixed,
    expected_mask_name,
    rotate_dataset_orientation_180,
)


def _write_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype("uint8"), mode="RGB").save(path)


def _write_gray(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype("uint8"), mode="L").save(path)


def test_rotate_dataset_orientation_180(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "perspective_views"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    image_pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    mask_pixels = np.array(
        [
            [0, 255],
            [128, 64],
        ],
        dtype=np.uint8,
    )
    _write_rgb(images_dir / "frame_00000_front.png", image_pixels)
    _write_gray(masks_dir / expected_mask_name("frame_00000_front.png"), mask_pixels)

    result = rotate_dataset_orientation_180(dataset_root)
    assert result["success"] is True
    assert result["images_rotated"] == 1
    assert result["masks_rotated"] == 1
    assert dataset_orientation_fixed(dataset_root) is True

    rotated_image = np.array(Image.open(images_dir / "frame_00000_front.png"))
    rotated_mask = np.array(Image.open(masks_dir / expected_mask_name("frame_00000_front.png")))

    assert rotated_image[0, 0].tolist() == image_pixels[1, 1].tolist()
    assert int(rotated_mask[0, 0]) == int(mask_pixels[1, 1])


def test_copy_cubemap_subset(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_images = source_root / "perspective_views"
    source_masks = source_root / "masks"
    source_images.mkdir(parents=True)
    source_masks.mkdir(parents=True)

    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=np.uint8)
    for frame_idx in range(3):
        for face in ("front", "back"):
            _write_rgb(source_images / f"frame_{frame_idx:05d}_{face}.jpeg", sample)
            _write_gray(source_masks / f"frame_{frame_idx:05d}_{face}_mask.png", mask)

    target_root = tmp_path / "target"
    result = copy_cubemap_subset(source_root, target_root, max_frames=2, include_masks=True)

    assert result["frames"] == 2
    assert result["images"] == 4
    assert result["masks"] == 4
    assert len(list((target_root / "perspective_views").iterdir())) == 4
    assert len(list((target_root / "masks").iterdir())) == 4