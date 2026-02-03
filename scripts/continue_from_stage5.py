#!/usr/bin/env python3
"""
Continue pipeline from Stage 5 (perspectives) using existing Stage 1-4 data.
Runs: Stage 5 (E2P) → Stage 6 (Mask perspectives) → Stage 7 (Lichtfeld) → Stage 8 (RealityScan)
"""

import os
import sys
import cv2
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION - Use existing Stage 1-4 output
# ============================================================
BASE_DIR = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\full_pipeline_20260117_164012")

# Camera configuration - 8 cameras around compass (default)
CAMERAS = [
    {"name": "cam_N",  "yaw": 0,    "pitch": 0, "fov": 110},
    {"name": "cam_NE", "yaw": 45,   "pitch": 0, "fov": 110},
    {"name": "cam_E",  "yaw": 90,   "pitch": 0, "fov": 110},
    {"name": "cam_SE", "yaw": 135,  "pitch": 0, "fov": 110},
    {"name": "cam_S",  "yaw": 180,  "pitch": 0, "fov": 110},
    {"name": "cam_SW", "yaw": -135, "pitch": 0, "fov": 110},
    {"name": "cam_W",  "yaw": -90,  "pitch": 0, "fov": 110},
    {"name": "cam_NW", "yaw": -45,  "pitch": 0, "fov": 110},
]

PERSPECTIVE_WIDTH = 1920
PERSPECTIVE_HEIGHT = 1080


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def create_xmp_sidecar(image_path: Path, metadata: dict) -> Path:
    """Create XMP sidecar file for RealityScan compatibility."""
    pos = metadata.get("position", {})
    cam = metadata.get("camera", {})
    
    xmp_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:Camera="http://ns.360toolkit.com/camera/1.0/"
      xmlns:Position="http://ns.360toolkit.com/position/1.0/">
      <Camera:FocalLength>{cam.get("fov", 110)}</Camera:FocalLength>
      <Camera:Yaw>{cam.get("yaw", 0)}</Camera:Yaw>
      <Camera:Pitch>{cam.get("pitch", 0)}</Camera:Pitch>
      <Camera:Roll>0</Camera:Roll>
      <Camera:Width>{cam.get("width", PERSPECTIVE_WIDTH)}</Camera:Width>
      <Camera:Height>{cam.get("height", PERSPECTIVE_HEIGHT)}</Camera:Height>
      <Position:X>{pos.get("x", 0)}</Position:X>
      <Position:Y>{pos.get("y", 0)}</Position:Y>
      <Position:Z>{pos.get("z", 0)}</Position:Z>
      <Position:AlignmentMethod>{metadata.get("alignment_method", "colmap")}</Position:AlignmentMethod>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>'''
    
    xmp_path = image_path.with_suffix('.xmp')
    with open(xmp_path, 'w') as f:
        f.write(xmp_content)
    return xmp_path


def stage5_extract_perspectives():
    """Stage 5: Transform equirectangular to perspective views."""
    logger.info("=" * 60)
    logger.info("STAGE 5: TRANSFORM EQUIRECTANGULAR TO PERSPECTIVES (E2P)")
    logger.info("=" * 60)
    
    from src.transforms.e2p_transform import E2PTransform
    
    transform = E2PTransform()
    equirect_dir = BASE_DIR / "stage1_equirectangular"
    perspectives_dir = BASE_DIR / "stage3_perspectives"
    metadata_dir = BASE_DIR / "metadata"
    
    # Ensure output dir exists
    perspectives_dir.mkdir(parents=True, exist_ok=True)
    
    # Get frames
    frames = sorted(equirect_dir.glob("*.png"))
    logger.info(f"  Processing {len(frames)} equirectangular frames")
    logger.info(f"  Cameras: {len(CAMERAS)}")
    logger.info(f"  Output size: {PERSPECTIVE_WIDTH}x{PERSPECTIVE_HEIGHT}")
    
    start_time = time.time()
    perspectives_created = 0
    
    # Load positions from metadata
    positions = {}
    for meta_file in metadata_dir.glob("*_metadata.json"):
        meta = load_json(meta_file)
        frame_name = meta.get("frame", "")
        if frame_name:
            pos = meta.get("position", {})
            positions[frame_name] = (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
    
    for i, frame_path in enumerate(frames):
        frame_name = frame_path.name
        frame_stem = frame_path.stem
        
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.warning(f"  Could not read: {frame_name}")
            continue
        
        # Get parent position
        parent_pos = positions.get(frame_name, (0, 0, 0))
        
        for cam in CAMERAS:
            # Extract perspective using equirect_to_pinhole
            perspective = transform.equirect_to_pinhole(
                img,
                yaw=cam["yaw"],
                pitch=cam["pitch"],
                roll=0,
                h_fov=cam["fov"],
                v_fov=None,  # Auto-calculated
                output_width=PERSPECTIVE_WIDTH,
                output_height=PERSPECTIVE_HEIGHT
            )
            
            # Save perspective image
            persp_name = f"{frame_stem}_{cam['name']}.png"
            persp_path = perspectives_dir / persp_name
            cv2.imwrite(str(persp_path), perspective)
            perspectives_created += 1
            
            # Create metadata for this perspective
            persp_metadata = {
                "source_frame": frame_name,
                "position": {
                    "x": parent_pos[0],
                    "y": parent_pos[1],
                    "z": parent_pos[2]
                },
                "camera": {
                    "yaw": cam["yaw"],
                    "pitch": cam["pitch"],
                    "fov": cam["fov"],
                    "width": PERSPECTIVE_WIDTH,
                    "height": PERSPECTIVE_HEIGHT
                },
                "alignment_method": "colmap"
            }
            
            # Save XMP sidecar for RealityScan
            create_xmp_sidecar(persp_path, persp_metadata)
            
            # Also save JSON metadata
            json_path = metadata_dir / f"{persp_path.stem}_metadata.json"
            save_json(persp_metadata, json_path)
        
        logger.info(f"  [{i+1}/{len(frames)}] {frame_name}: {len(CAMERAS)} perspectives")
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Created {perspectives_created} perspectives in {elapsed:.1f}s")
    
    return list(perspectives_dir.glob("*.png"))


def stage6_mask_perspectives(perspectives):
    """Stage 6: Mask perspective images."""
    logger.info("=" * 60)
    logger.info("STAGE 6: MASK PERSPECTIVE IMAGES")
    logger.info("=" * 60)
    
    from src.masking.multi_category_masker import MultiCategoryMasker
    
    masks_dir = BASE_DIR / "stage3_perspective_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    masker = MultiCategoryMasker(
        model_size='medium',
        confidence_threshold=0.5,
        use_gpu=True
    )
    # Enable categories
    masker.enabled_categories = {
        'persons': True,
        'personal_objects': False,  # Skip personal objects
        'animals': True
    }
    
    logger.info(f"  Processing {len(perspectives)} perspectives")
    
    start_time = time.time()
    masks_created = 0
    skipped = 0
    
    for i, persp_path in enumerate(perspectives):
        persp_path = Path(persp_path)
        
        # Generate mask
        mask = masker.generate_mask(str(persp_path))
        
        if mask is not None:
            # Check if mask has any black pixels (detections)
            has_detections = (mask < 128).any() if len(mask.shape) == 2 else (mask.min(axis=2) < 128).any()
            
            if has_detections:
                mask_name = f"{persp_path.stem}_mask.png"
                mask_path = masks_dir / mask_name
                cv2.imwrite(str(mask_path), mask)
                masks_created += 1
                if (i + 1) % 20 == 0:
                    logger.info(f"  [{i+1}/{len(perspectives)}] Processing...")
            else:
                skipped += 1
        else:
            skipped += 1
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Created {masks_created} masks, skipped {skipped} in {elapsed:.1f}s")
    
    return list(masks_dir.glob("*.png"))


def stage7_export_lichtfeld(perspectives, perspective_masks, equirect_masks):
    """Stage 7: Export to Lichtfeld Studio format."""
    logger.info("=" * 60)
    logger.info("STAGE 7: EXPORT TO LICHTFELD STUDIO")
    logger.info("=" * 60)
    
    export_dir = BASE_DIR / "export_lichtfeld"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    equirect_dir = BASE_DIR / "stage1_equirectangular"
    equirect_mask_dir = BASE_DIR / "stage1_equirectangular_masks"
    
    # Lichtfeld uses equirectangular images with masks
    start_time = time.time()
    copied = 0
    
    # Copy equirectangular images
    for frame in sorted(equirect_dir.glob("*.png")):
        shutil.copy2(frame, export_dir / frame.name)
        copied += 1
    
    # Copy equirectangular masks
    for mask in sorted(equirect_mask_dir.glob("*.png")):
        shutil.copy2(mask, export_dir / mask.name)
        copied += 1
    
    # Copy COLMAP sparse model
    colmap_dir = BASE_DIR / "stage2_colmap" / "sparse" / "0"
    lichtfeld_colmap = export_dir / "colmap"
    lichtfeld_colmap.mkdir(parents=True, exist_ok=True)
    
    for colmap_file in colmap_dir.glob("*.*"):
        shutil.copy2(colmap_file, lichtfeld_colmap / colmap_file.name)
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Exported {copied} files + COLMAP model in {elapsed:.1f}s")
    
    return export_dir


def stage8_export_realityscan(perspectives, perspective_masks):
    """Stage 8: Export to RealityScan format (perspectives + XMP)."""
    logger.info("=" * 60)
    logger.info("STAGE 8: EXPORT TO REALITYSCAN")
    logger.info("=" * 60)
    
    export_dir = BASE_DIR / "export_realityscan"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    perspectives_dir = BASE_DIR / "stage3_perspectives"
    masks_dir = BASE_DIR / "stage3_perspective_masks"
    
    start_time = time.time()
    copied = 0
    
    # Copy perspective images
    for persp in sorted(perspectives_dir.glob("*.png")):
        shutil.copy2(persp, export_dir / persp.name)
        copied += 1
    
    # Copy XMP sidecars
    for xmp in sorted(perspectives_dir.glob("*.xmp")):
        shutil.copy2(xmp, export_dir / xmp.name)
        copied += 1
    
    # Copy perspective masks
    for mask in sorted(masks_dir.glob("*.png")):
        shutil.copy2(mask, export_dir / mask.name)
        copied += 1
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Exported {copied} files (images + XMP + masks) in {elapsed:.1f}s")
    
    return export_dir


def main():
    """Continue pipeline from Stage 5."""
    print("=" * 70)
    print("360TOOLKIT - CONTINUE PIPELINE FROM STAGE 5")
    print("=" * 70)
    print(f"Base directory: {BASE_DIR}")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Verify existing data
    equirect_dir = BASE_DIR / "stage1_equirectangular"
    equirect_frames = list(equirect_dir.glob("*.png"))
    logger.info(f"Found {len(equirect_frames)} equirectangular frames")
    
    equirect_masks_dir = BASE_DIR / "stage1_equirectangular_masks"
    equirect_masks = list(equirect_masks_dir.glob("*.png"))
    logger.info(f"Found {len(equirect_masks)} equirectangular masks")
    
    colmap_dir = BASE_DIR / "stage2_colmap" / "sparse" / "0"
    if colmap_dir.exists():
        logger.info(f"Found COLMAP sparse model at {colmap_dir}")
    
    # Stage 5: Extract perspectives
    perspectives = stage5_extract_perspectives()
    
    # Stage 6: Mask perspectives
    perspective_masks = stage6_mask_perspectives(perspectives)
    
    # Stage 7: Export to Lichtfeld
    lichtfeld_dir = stage7_export_lichtfeld(perspectives, perspective_masks, equirect_masks)
    
    # Stage 8: Export to RealityScan
    realityscan_dir = stage8_export_realityscan(perspectives, perspective_masks)
    
    # Summary
    total_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print("PIPELINE CONTINUATION COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()
    print("Outputs:")
    print(f"  Perspectives: {BASE_DIR / 'stage3_perspectives'}")
    print(f"  Perspective masks: {BASE_DIR / 'stage3_perspective_masks'}")
    print()
    print("Exports:")
    print(f"  Lichtfeld Studio: {lichtfeld_dir}")
    print(f"  RealityScan: {realityscan_dir}")
    print("=" * 70)
    print()
    print("✓ Pipeline continuation completed successfully!")


if __name__ == "__main__":
    main()
