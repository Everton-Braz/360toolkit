import numpy as np

from src.masking.mask_refinement import MaskRefinementSettings, refine_detected_mask


def test_refine_detected_mask_is_noop_when_disabled():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    mask = np.zeros((32, 64), dtype=np.uint8)
    mask[:12, :] = 255

    refined = refine_detected_mask(
        image,
        mask,
        MaskRefinementSettings(enabled=False, edge_band_radius=8),
    )

    assert np.array_equal(refined, mask)


def test_refine_detected_mask_preserves_binary_shape():
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    image[:30, :, :] = (220, 180, 80)
    image[30:, :, :] = (45, 60, 70)

    mask = np.zeros((64, 128), dtype=np.uint8)
    mask[:28, :] = 255
    mask[24:32, 40:44] = 0
    mask[26:31, 90:94] = 0

    refined = refine_detected_mask(
        image,
        mask,
        MaskRefinementSettings(enabled=True, edge_band_radius=6, sharpen_strength=0.75, seam_aware=True),
    )

    assert refined.shape == mask.shape
    assert refined.dtype == np.uint8
    assert set(np.unique(refined)).issubset({0, 255})