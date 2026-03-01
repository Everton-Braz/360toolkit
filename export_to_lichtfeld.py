#!/usr/bin/env python3
r"""
Export SphereSFM Reconstruction to LichtFeld Studio Format

Converts COLMAP sparse reconstruction to LichtFeld Studio format for 3DGS training.

Usage:
    python export_to_lichtfeld.py <colmap_sparse_dir> <images_dir> <output_dir> [masks_dir]
    
Example:
    python export_to_lichtfeld.py 
        "C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\spheresfm_equirect_20260202_195436\sparse\0"
        "C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\INPUT_TEST_360_IMAGES\stage1_frames"
        "C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\lichtfeld_export"
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.export_formats import LichtfeldExporter


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def count_files(directory: Path, extensions=['.jpg', '.png', '.txt', '.ply', '.json']):
    """Count files by extension in a directory."""
    counts = {}
    if not directory.exists():
        return counts
    
    for ext in extensions:
        files = list(directory.glob(f"*{ext}"))
        if files:
            counts[ext] = len(files)
    
    return counts


def verify_colmap_model(sparse_dir: Path):
    """Verify COLMAP sparse model exists and is valid."""
    required_files = ['cameras.txt', 'images.txt', 'points3D.txt']
    
    for filename in required_files:
        filepath = sparse_dir / filename
        if not filepath.exists():
            print(f"  ✗ Missing: {filename}")
            return False
        print(f"  ✓ Found: {filename}")
    
    # Count data
    with open(sparse_dir / 'cameras.txt', 'r') as f:
        cameras = len([l for l in f if l.strip() and not l.startswith('#')])
    
    with open(sparse_dir / 'images.txt', 'r') as f:
        image_lines = len([l for l in f if l.strip() and not l.startswith('#')])
    images = image_lines // 2
    
    with open(sparse_dir / 'points3D.txt', 'r') as f:
        points = len([l for l in f if l.strip() and not l.startswith('#')])
    
    print(f"\n  Model Statistics:")
    print(f"    Cameras: {cameras}")
    print(f"    Registered Images: {images}")
    print(f"    3D Points: {points:,}")
    
    return True


def export_lichtfeld(
    sparse_dir: Path,
    images_dir: Path,
    output_dir: Path,
    fix_rotation: bool = True,
    masks_dir: Path | None = None,
):
    """Export COLMAP reconstruction to LichtFeld Studio format."""
    
    print_section("Exporting to LichtFeld Studio Format")
    
    print(f"Input:")
    print(f"  COLMAP Model: {sparse_dir}")
    print(f"  Images: {images_dir}")
    print(f"Output:")
    print(f"  Directory: {output_dir}")
    print(f"Settings:")
    print(f"  Fix Rotation: {fix_rotation} (recommended for 360 images)")
    if masks_dir:
        print(f"  Masks: {masks_dir}")
    else:
        print(f"  Masks: disabled")
    
    # Verify inputs
    print_section("Verifying Inputs")
    
    if not sparse_dir.exists():
        print(f"✗ ERROR: Sparse directory not found: {sparse_dir}")
        return False
    
    if not images_dir.exists():
        print(f"✗ ERROR: Images directory not found: {images_dir}")
        return False

    if masks_dir and not masks_dir.exists():
        print(f"✗ ERROR: Masks directory not found: {masks_dir}")
        return False
    
    # Check COLMAP model
    print("\nCOLMAP Model:")
    if not verify_colmap_model(sparse_dir):
        print(f"\n✗ ERROR: Invalid COLMAP model!")
        return False
    
    # Check images
    print("\nImages:")
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"  Found: {len(image_files)} images")
    
    if not image_files:
        print(f"  ✗ ERROR: No images found in {images_dir}")
        return False
    
    print(f"  ✓ Sample: {image_files[0].name}")
    
    # Create exporter
    print_section("Running Export")
    
    start_time = time.perf_counter()
    
    try:
        # LichtfeldExporter expects the directory containing cameras.txt, images.txt, points3D.txt
        # If sparse_dir is .../sparse/0, pass that directly
        
        print(f"Initializing exporter...")
        exporter = LichtfeldExporter(
            colmap_dir=str(sparse_dir),
            output_dir=str(output_dir)
        )
        
        print(f"Exporting transforms.json and point cloud...")
        success = exporter.export(
            images_dir=str(images_dir),
            fix_rotation=fix_rotation,
            masks_dir=str(masks_dir) if masks_dir else None,
        )
        
        elapsed = time.perf_counter() - start_time
        
        if success:
            print_section("Export Successful ✓")
            print(f"Time: {format_time(elapsed)}")
            
            # Verify outputs
            print(f"\nOutput Files:")
            
            transforms_json = output_dir / "transforms.json"
            if transforms_json.exists():
                print(f"  ✓ {transforms_json.name}")
                
                # Parse and show info
                import json
                with open(transforms_json, 'r') as f:
                    data = json.load(f)
                print(f"    - Frames: {len(data.get('frames', []))}")
                print(f"    - Camera Model: {data.get('camera_model', 'N/A')}")
                if 'ply_file_path' in data:
                    print(f"    - PLY: {data['ply_file_path']}")
            else:
                print(f"  ✗ Missing: transforms.json")
            
            pointcloud_ply = output_dir / "pointcloud.ply"
            if pointcloud_ply.exists():
                size_mb = pointcloud_ply.stat().st_size / (1024 * 1024)
                print(f"  ✓ {pointcloud_ply.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ⚠ Missing: pointcloud.ply")
            
            images_out = output_dir / "images"
            if images_out.exists():
                copied_images = list(images_out.glob("*.jpg")) + list(images_out.glob("*.png"))
                print(f"  ✓ images/ ({len(copied_images)} files)")
            else:
                print(f"  ✗ Missing: images/")

            masks_out = output_dir / "masks"
            if masks_out.exists():
                copied_masks = list(masks_out.glob("*.png"))
                print(f"  ✓ masks/ ({len(copied_masks)} files, format=image.jpg.png)")
            elif masks_dir:
                print(f"  ⚠ Missing: masks/")
            
            print(f"\nLichtFeld Studio Usage:")
            print(f"  1. Open LichtFeld Studio")
            print(f"  2. Click 'Import dataset'")
            print(f"  3. Select: {output_dir}")
            print(f"  4. Start training!")
            
            print(f"\nCommand line training:")
            print(f"  ./LichtFeld-Studio -d \"{output_dir}\" -o output/my_scene")
            
            return True
        else:
            print(f"\n✗ Export failed after {format_time(elapsed)}")
            return False
            
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print_section("Export Error ✗")
        print(f"Failed after {format_time(elapsed)}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main export function."""
    
    # Default test configuration
    default_sparse = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\spheresfm_equirect_20260202_195436\sparse\0")
    default_images = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\INPUT_TEST_360_IMAGES\stage1_frames")
    default_output = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\lichtfeld_export")
    
    # Parse arguments
    if len(sys.argv) in (4, 5):
        sparse_dir = Path(sys.argv[1])
        images_dir = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])
        masks_dir = Path(sys.argv[4]) if len(sys.argv) == 5 else None
    else:
        # Use defaults
        sparse_dir = default_sparse
        images_dir = default_images
        output_dir = default_output
        masks_dir = images_dir.parent / "masks"
        if not masks_dir.exists():
            masks_dir = None
        
        print_section("LichtFeld Studio Export Tool")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nUsing default paths:")
        print(f"  COLMAP: {sparse_dir}")
        print(f"  Images: {images_dir}")
        print(f"  Output: {output_dir}")
        print()
        
        if len(sys.argv) > 1:
            print("Usage: python export_to_lichtfeld.py <sparse_dir> <images_dir> <output_dir> [masks_dir]")
            print()
        
        response = input("Continue with these paths? (y/n): ")
        if response.lower() != 'y':
            print("\nProvide paths as arguments:")
            print('  python export_to_lichtfeld.py "path/to/sparse/0" "path/to/images" "path/to/output"')
            return 1
    
    # Run export
    total_start = time.perf_counter()
    
    if masks_dir is None:
        auto_masks = images_dir.parent / "masks"
        if auto_masks.exists():
            masks_dir = auto_masks
            print(f"Auto-detected masks directory: {masks_dir}")

    success = export_lichtfeld(
        sparse_dir,
        images_dir,
        output_dir,
        fix_rotation=True,
        masks_dir=masks_dir,
    )
    
    total_elapsed = time.perf_counter() - total_start
    
    if success:
        print_section("COMPLETE ✓")
        print(f"Total time: {format_time(total_elapsed)}")
        print(f"\nExport directory: {output_dir}")
        return 0
    else:
        print_section("FAILED ✗")
        print(f"Total time: {format_time(total_elapsed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
