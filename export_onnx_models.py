"""
Export YOLOv8 Segmentation Models to ONNX Format

This script exports YOLOv8 segmentation models to ONNX format for use with
ONNX Runtime, eliminating the need for PyTorch at runtime.

Benefits:
- ~90% smaller executable (500 MB vs 6+ GB)
- Faster loading time
- Compatible with CPU and GPU (CUDA/DirectML/TensorRT)
- Works on machines without PyTorch installed

Usage:
    python export_onnx_models.py

Output:
    yolov8n-seg.onnx  (~15 MB, fastest, ~85% accuracy)
    yolov8s-seg.onnx  (~30 MB, balanced, ~90% accuracy) [DEFAULT]
    yolov8m-seg.onnx  (~55 MB, better accuracy, ~92%)
"""

import sys
from pathlib import Path

def export_models():
    """Export YOLOv8 segmentation models to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics package not installed.")
        print("Install with: pip install ultralytics")
        return False
    
    # Models to export (nano, small, medium)
    models_to_export = [
        ('yolov8n-seg.pt', 'yolov8n-seg.onnx'),  # Nano - fastest
        ('yolov8s-seg.pt', 'yolov8s-seg.onnx'),  # Small - recommended
        ('yolov8m-seg.pt', 'yolov8m-seg.onnx'),  # Medium - best accuracy
    ]
    
    export_settings = {
        'format': 'onnx',
        'simplify': True,      # Simplify ONNX graph for faster inference
        'dynamic': False,      # Static shapes for better optimization
        'imgsz': 640,          # Input image size
        'half': False,         # FP32 for compatibility (can use FP16 for GPU only)
        'opset': 17,           # ONNX opset version (17 is well supported)
    }
    
    print("="*70)
    print("YOLOv8 Segmentation Model Export to ONNX")
    print("="*70)
    print(f"Export settings: {export_settings}")
    print()
    
    success_count = 0
    
    for pt_name, onnx_name in models_to_export:
        onnx_path = Path(onnx_name)
        
        # Skip if already exists
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"✓ {onnx_name} already exists ({size_mb:.1f} MB) - skipping")
            success_count += 1
            continue
        
        print(f"\nExporting {pt_name} -> {onnx_name}...")
        
        try:
            # Load model (downloads automatically if not present)
            model = YOLO(pt_name)
            
            # Export to ONNX
            model.export(**export_settings)
            
            # ultralytics saves to same name with .onnx extension
            exported_path = Path(pt_name).with_suffix('.onnx')
            
            if exported_path.exists():
                size_mb = exported_path.stat().st_size / (1024 * 1024)
                print(f"✓ Exported {onnx_name} ({size_mb:.1f} MB)")
                success_count += 1
            else:
                print(f"✗ Failed to export {onnx_name}")
                
        except Exception as e:
            print(f"✗ Error exporting {pt_name}: {e}")
    
    print()
    print("="*70)
    print(f"Export complete: {success_count}/{len(models_to_export)} models exported")
    print("="*70)
    
    return success_count == len(models_to_export)


if __name__ == '__main__':
    success = export_models()
    sys.exit(0 if success else 1)
