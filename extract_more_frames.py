"""
Extract more frames from INSV video for full Stage 4 test
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.extraction.sdk_extractor import SDKExtractor
from src.config.defaults import ExtractionSettings

def extract_frames():
    """Extract 30 frames at 1 FPS for testing"""
    
    input_file = Path(r"G:\.shortcut-targets-by-id\12X9Cn_caDGuRMIO-hF6196FMdQyGNUDA\PROJETOS - CHICO SOMBRA\VIDEOS 360\VID_20251215_170106_00_211.insv")
    output_dir = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\stage1_frames")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing frames
    for f in output_dir.glob("*.png"):
        f.unlink()
    
    logger.info("=" * 80)
    logger.info("EXTRACTING 30 FRAMES FOR STAGE 4 TEST")
    logger.info("=" * 80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Settings: 1 FPS, first 30 seconds, 8K resolution")
    logger.info("")
    
    # Configure extraction
    settings = ExtractionSettings(
        fps=1.0,
        start_time=0,
        end_time=30,  # 30 seconds = 30 frames at 1 FPS
        resolution="8K",
        output_format="png",
        use_gpu=True
    )
    
    # Extract frames
    extractor = SDKExtractor(settings)
    
    logger.info("Starting extraction...")
    frame_paths = extractor.extract_frames(
        input_file=input_file,
        output_dir=output_dir
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ Extracted {len(frame_paths)} frames")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next step: Run test_stage4_alignment.py")
    
    return frame_paths

if __name__ == "__main__":
    try:
        extract_frames()
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        sys.exit(1)
