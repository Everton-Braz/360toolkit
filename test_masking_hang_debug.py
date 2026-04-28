#!/usr/bin/env python3
"""
Test script to debug why masking hangs after initialization.
Reproduces the exact flow from batch_orchestrator._execute_stage3()
"""
import sys
import logging
from pathlib import Path

# Setup logging with verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Import from src as a proper package
sys.path.insert(0, str(Path(__file__).parent))

from src.masking.sam3_external_masker import SAM3ExternalMasker
from src.config.settings import get_settings

logger.info("=" * 80)
logger.info("Testing SAM3 masking with accented path")
logger.info("=" * 80)

input_dir = r"D:\ARQUIVOS_TESTE_2\Pecem_8K\PÉCEM_FISHEYE\extracted_frames"
output_dir = r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\PÉCEM_FISHEYE_masks"

input_path = Path(input_dir)
output_path = Path(output_dir)

logger.info(f"Input directory exists: {input_path.exists()}")
logger.info(f"Input directory: {input_path}")

if input_path.exists():
    images = list(input_path.glob('*'))
    image_files = [p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}]
    logger.info(f"Total files in input: {len(images)}")
    logger.info(f"Image files found: {len(image_files)}")
    if image_files:
        logger.info(f"First image: {image_files[0].name}")

# Create masker
logger.info("\n" + "=" * 80)
logger.info("Creating SAM3ExternalMasker...")
logger.info("=" * 80)

try:
    settings = get_settings()
    
    exe_path = settings.get('sam3_segmenter_path')
    model_path = settings.get('sam3_model_path')
    
    logger.info(f"SAM3 exe: {exe_path}")
    logger.info(f"SAM3 model: {model_path}")
    
    masker = SAM3ExternalMasker(
        segment_persons_exe=str(exe_path),
        model_path=str(model_path),
        use_gpu=True,
        alpha_only=False,
    )
    logger.info("[OK] SAM3ExternalMasker created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create masker: {e}", exc_info=True)
    sys.exit(1)

# Run process_batch with detailed logging
logger.info("\n" + "=" * 80)
logger.info("Calling process_batch()...")
logger.info("=" * 80)

try:
    output_path.mkdir(parents=True, exist_ok=True)
    
    def progress_cb(current, total, msg):
        logger.info(f"[Progress] {current}/{total}: {msg}")
    
    def cancel_check():
        return False  # Never cancel
    
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    result = masker.process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        save_visualization=False,
        progress_callback=progress_cb,
        cancellation_check=cancel_check,
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("[OK] process_batch() completed!")
    logger.info("=" * 80)
    logger.info(f"Result: {result}")
    
except Exception as e:
    logger.error(f"❌ process_batch() FAILED: {e}", exc_info=True)
    sys.exit(1)

logger.info("\nTest completed successfully!")
