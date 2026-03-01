"""
Prepare a fresh 200-image test subset from the full perspective_views folder.
Copies evenly-spaced images + their corresponding masks to a new test folder.
"""
import os
import shutil
from pathlib import Path

SRC_VIEWS = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\teste_17022026-1515\perspective_views")
SRC_MASKS = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\teste_17022026-1515\masks")
DEST_ROOT = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\test_200_images")

DEST_VIEWS = DEST_ROOT / "perspective_views"
DEST_MASKS = DEST_ROOT / "masks"
N = 200  # images to pick

def main():
    # Collect all perspective view files, sorted
    all_images = sorted(SRC_VIEWS.glob("*"))
    all_images = [f for f in all_images if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    total = len(all_images)
    print(f"Total available images: {total}")

    if total < N:
        print(f"WARNING: Only {total} images available, using all of them.")
        selected = all_images
    else:
        # Evenly spaced indices
        step = total / N
        indices = [int(i * step) for i in range(N)]
        selected = [all_images[i] for i in indices]

    print(f"Selected {len(selected)} images (every ~{total/len(selected):.1f}th)")

    # Clean and recreate destination
    if DEST_ROOT.exists():
        print(f"Removing existing {DEST_ROOT} ...")
        shutil.rmtree(DEST_ROOT)

    DEST_VIEWS.mkdir(parents=True, exist_ok=True)
    DEST_MASKS.mkdir(parents=True, exist_ok=True)

    # Copy images + masks
    copied_images = 0
    copied_masks = 0

    for img_path in selected:
        # Copy image
        dest_img = DEST_VIEWS / img_path.name
        shutil.copy2(img_path, dest_img)
        copied_images += 1

        # Look for matching mask: frame_XXXXX_cam_YY_mask.png
        stem = img_path.stem  # e.g., frame_00000_cam_00
        mask_name = stem + "_mask.png"
        mask_src = SRC_MASKS / mask_name
        if mask_src.exists():
            shutil.copy2(mask_src, DEST_MASKS / mask_name)
            copied_masks += 1

    print(f"\n✅ Copied {copied_images} images to:  {DEST_VIEWS}")
    print(f"✅ Copied {copied_masks} masks to:    {DEST_MASKS}")
    print(f"\nTest folder ready: {DEST_ROOT}")


if __name__ == "__main__":
    main()
