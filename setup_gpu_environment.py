#!/usr/bin/env python3
"""
360toolkit GPU Environment Setup and Diagnostics
Ensures the app runs optimally on both GTX and RTX GPUs.
Supports: GTX 1650, RTX 30/40 series, and RTX 5070 Ti
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_cuda_availability():
    """Check if CUDA is available and return GPU information."""
    logger.info("=" * 60)
    logger.info("CHECKING CUDA AND GPU AVAILABILITY")
    logger.info("=" * 60)
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA is available")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_prop = torch.cuda.get_device_properties(i)
                logger.info(f"   GPU {i}: {gpu_name}")
                logger.info(f"         Compute Capability: {gpu_prop.major}.{gpu_prop.minor}")
                logger.info(f"         Total Memory: {gpu_prop.total_memory / 1e9:.2f} GB")
            
            return True
        else:
            logger.warning("⚠️  CUDA not available - will run on CPU")
            return False
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False

def check_onnx_runtime():
    """Check if ONNX Runtime is available."""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING ONNX RUNTIME")
    logger.info("=" * 60)
    
    try:
        import onnxruntime as ort
        
        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            logger.info("✅ ONNX Runtime with GPU support available")
            return True
        elif 'CPUExecutionProvider' in available_providers:
            logger.warning("⚠️  ONNX Runtime available (CPU only)")
            return False
        else:
            logger.error("❌ ONNX Runtime not properly configured")
            return False
    except ImportError:
        logger.warning("⚠️  ONNX Runtime not installed")
        return False

def check_dependencies():
    """Check all critical dependencies."""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("=" * 60)
    
    dependencies = {
        'PyQt6': 'GUI Framework',
        'numpy': 'Numerical Computing',
        'cv2': 'Computer Vision (OpenCV)',
        'PIL': 'Image Processing (Pillow)',
        'torch': 'PyTorch (Deep Learning)',
        'ultralytics': 'YOLOv8 Detection',
        'segment_anything': 'SAM Segmentation',
    }
    
    missing = []
    for module, description in dependencies.items():
        try:
            __import__(module)
            logger.info(f"✅ {module:20s} - {description}")
        except ImportError:
            logger.warning(f"⚠️  {module:20s} - NOT INSTALLED ({description})")
            missing.append(module)
    
    return len(missing) == 0, missing

def setup_recommended_settings(has_cuda: bool, has_onnx: bool):
    """Create recommended settings for the current system."""
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED SETTINGS")
    logger.info("=" * 60)
    
    settings = {
        "masking_engine": "pytorch",  # Primary choice
        "use_gpu": has_cuda,
        "torch_device": "cuda" if has_cuda else "cpu",
        "onnx_enabled": has_onnx,
        "batch_size": 16 if has_cuda else 4,
        "yolo_model_size": "small",  # Balanced for both GTX and RTX
    }
    
    if has_cuda:
        try:
            import torch
            # Detect GPU for batch size recommendations
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                if gpu_memory_gb >= 12:  # RTX 3060+, RTX 40 series, RTX 5070 Ti
                    settings["batch_size"] = 32
                    settings["yolo_model_size"] = "medium"
                    logger.info(f"Detected high-VRAM GPU ({gpu_memory_gb:.1f} GB)")
                    logger.info("  Recommended: batch_size=32, model=medium")
                elif gpu_memory_gb >= 4:  # GTX 1650, GTX 1050
                    settings["batch_size"] = 8
                    settings["yolo_model_size"] = "small"
                    logger.info(f"Detected mid-range GPU ({gpu_memory_gb:.1f} GB)")
                    logger.info("  Recommended: batch_size=8, model=small")
                else:
                    settings["batch_size"] = 4
                    settings["yolo_model_size"] = "nano"
                    logger.info(f"Detected low-VRAM GPU ({gpu_memory_gb:.1f} GB)")
                    logger.info("  Recommended: batch_size=4, model=nano")
        except Exception as e:
            logger.warning(f"Could not detect GPU memory: {e}")
    else:
        logger.info("GPU not available - using CPU settings")
        logger.info("  CPU mode: batch_size=4, model=small")
    
    return settings

def save_gpu_config(settings: dict):
    """Save GPU configuration to settings file."""
    config_path = Path(__file__).parent / "src" / "config" / "gpu_settings.json"
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info(f"\n✅ GPU settings saved to: {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save GPU settings: {e}")
        return False

def main():
    """Run complete GPU setup and diagnostics."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " 360toolkit GPU Environment Setup ".center(58) + "║")
    logger.info("║" + " Supports: GTX, RTX 30/40/50 series ".center(58) + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    # Run checks
    has_cuda = check_cuda_availability()
    has_onnx = check_onnx_runtime()
    all_deps_ok, missing = check_dependencies()
    
    # Generate recommendations
    settings = setup_recommended_settings(has_cuda, has_onnx)
    save_gpu_config(settings)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 60)
    
    if all_deps_ok:
        logger.info("✅ All dependencies installed")
    else:
        logger.warning(f"⚠️  Missing dependencies: {', '.join(missing)}")
        logger.info("   Install with: pip install -r requirements.txt")
    
    if has_cuda:
        logger.info("✅ GPU acceleration available (CUDA)")
    else:
        logger.warning("⚠️  GPU acceleration not available (CPU mode)")
    
    if has_onnx:
        logger.info("✅ ONNX Runtime GPU support available")
    
    logger.info("\n" + "=" * 60)
    logger.info("Ready to run: python run_app.py")
    logger.info("=" * 60 + "\n")
    
    return 0 if all_deps_ok and has_cuda else 1

if __name__ == '__main__':
    sys.exit(main())
