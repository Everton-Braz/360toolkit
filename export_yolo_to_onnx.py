"""
YOLOv8 to ONNX Converter
One-time script to export YOLOv8 segmentation models to ONNX format.

Run this script once to create ONNX models for use with ONNXMasker.
This reduces final binary size from 6-8 GB to ~300-500 MB.

Usage:
    python export_yolo_to_onnx.py

Output:
    - yolov8n-seg.onnx (nano model, ~7 MB)
    - yolov8s-seg.onnx (small model, ~23 MB)  
    - yolov8m-seg.onnx (medium model, ~52 MB)
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_model_to_onnx(model_name: str, simplify: bool = True):
    """
    Export YOLOv8 model to ONNX format.
    
    Args:
        model_name: YOLOv8 model name (e.g., 'yolov8s-seg.pt')
        simplify: Whether to simplify ONNX model (recommended)
    """
    try:
        from ultralytics import YOLO
        
        logger.info(f"Loading YOLOv8 model: {model_name}")
        model = YOLO(model_name)
        
        # Export to ONNX
        onnx_path = model_name.replace('.pt', '.onnx')
        logger.info(f"Exporting to ONNX: {onnx_path}")
        
        model.export(format='onnx', simplify=simplify, dynamic=False)
        
        # Check if export successful
        onnx_file = Path(onnx_path)
        if onnx_file.exists():
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Export successful: {onnx_path} ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ Export failed: {onnx_path} not created")
        
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        logger.error(f"Error exporting {model_name}: {e}")


def main():
    """Export all YOLOv8 segmentation models to ONNX"""
    
    print("="*70)
    print("YOLOv8 to ONNX Converter")
    print("="*70)
    print()
    print("This script will export YOLOv8 models to ONNX format.")
    print("ONNX models can be used with ONNXMasker for ~90% size reduction.")
    print()
    
    # Models to export
    models = [
        'yolov8n-seg.pt',  # Nano (7 MB → ~7 MB ONNX)
        'yolov8s-seg.pt',  # Small (23 MB → ~23 MB ONNX)
        'yolov8m-seg.pt',  # Medium (52 MB → ~52 MB ONNX)
    ]
    
    print(f"Exporting {len(models)} models...")
    print()
    
    for model_name in models:
        export_model_to_onnx(model_name, simplify=True)
        print()
    
    print("="*70)
    print("Export complete!")
    print()
    print("Next steps:")
    print("1. Update your code to use ONNXMasker instead of MultiCategoryMasker")
    print("2. Update requirements.txt: add 'onnxruntime' or 'onnxruntime-gpu'")
    print("3. Remove torch/torchvision from requirements.txt")
    print("4. Rebuild with PyInstaller for smaller binary")
    print()
    print("Expected size reduction: 6-8 GB → 300-500 MB (90% smaller!)")
    print("="*70)


if __name__ == '__main__':
    main()
