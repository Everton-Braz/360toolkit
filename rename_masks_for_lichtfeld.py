"""
Rename masks from COLMAP format to LichtFeld Studio format.

COLMAP format:    image_name_mask.png  (e.g., "36_mask.png")
LichtFeld format: image_name.jpg.png   (e.g., "36.jpg.png")
"""

import os
from pathlib import Path
import shutil

def rename_masks(masks_dir: str):
    """Rename all masks in directory to LichtFeld Studio format."""
    masks_path = Path(masks_dir)
    
    if not masks_path.exists():
        print(f"❌ Directory not found: {masks_dir}")
        return
    
    # Find all mask files with "_mask.png" pattern
    mask_files = list(masks_path.glob("*_mask.png"))
    
    if not mask_files:
        print(f"❌ No mask files found in {masks_dir}")
        return
    
    print(f"Found {len(mask_files)} masks to rename\n")
    
    renamed_count = 0
    
    for mask_file in mask_files:
        # Extract the base name (remove "_mask.png")
        # Example: "36_mask.png" → "36"
        stem = mask_file.stem.replace("_mask", "")
        
        # Assume original image was .jpg (most common)
        # LichtFeld expects: "36.jpg.png"
        new_name = f"{stem}.jpg.png"
        new_path = mask_file.parent / new_name
        
        try:
            # Rename the file
            mask_file.rename(new_path)
            print(f"✓ {mask_file.name} → {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"✗ Failed to rename {mask_file.name}: {e}")
    
    print(f"\n✓ Renamed {renamed_count}/{len(mask_files)} masks")
    print(f"\nMasks are now in LichtFeld Studio format:")
    print(f"  Image: image.jpg")
    print(f"  Mask:  image.jpg.png")


if __name__ == "__main__":
    masks_dir = r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\pipeline_with_masking\20260202_204122\lichtfeld\masks"
    
    print("=" * 70)
    print("  Rename Masks for LichtFeld Studio")
    print("=" * 70)
    print(f"Target directory: {masks_dir}\n")
    
    rename_masks(masks_dir)
