#!/usr/bin/env python3
"""
Full Pipeline Test Script for 360ToolKit

Runs all 3 stages: Extract → Split → Mask
with timing and performance metrics.

Usage:
    python run_full_pipeline.py [input_video] [output_dir]
    
Examples:
    python run_full_pipeline.py
    python run_full_pipeline.py "G:\path\to\video.insv" "C:\output"
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.batch_orchestrator import PipelineWorker
from src.config.defaults import (
    DEFAULT_FPS, DEFAULT_H_FOV, DEFAULT_SPLIT_COUNT,
    DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_time(seconds):
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.0f}s"


def run_pipeline(input_file: str, output_dir: str, config_overrides: dict = None):
    """
    Run the full 3-stage pipeline.
    
    Args:
        input_file: Path to input .insv or .mp4 video
        output_dir: Output directory for all stages
        config_overrides: Optional dict to override default config
    """
    print("\n" + "="*70)
    print("  360ToolKit - Full Pipeline Execution")
    print("="*70)
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_dir}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Validate input
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build pipeline configuration
    config = {
        # Input/Output
        'input_file': str(input_path),
        'output_dir': str(output_path),
        
        # Stage enables
        'enable_stage1': True,
        'enable_stage2': True,
        'enable_stage3': True,
        'skip_transform': False,
        
        # Stage 1: Frame Extraction
        'fps': 2.0,  # 2 frames per second
        'extraction_method': 'sdk_stitching',
        'start_time': 0.0,
        'end_time': 15.0,  # First 15 seconds
        'sdk_quality': 'best',
        'sdk_resolution': '8k',
        'output_format': 'png',
        
        # Stage 2: Perspective Splitting
        'transform_type': 'perspective',
        'output_width': 1920,
        'output_height': 1920,
        'stage2_format': 'jpg',  # JPEG is faster
        'camera_config': {
            'cameras': generate_camera_positions(8, DEFAULT_H_FOV)
        },
        
        # Stage 3: Masking
        'masking_enabled': True,
        'mask_categories': ['person'],
        'yolo_model_size': 'medium',
        'confidence_threshold': 0.5,
    }
    
    # Apply overrides
    if config_overrides:
        config.update(config_overrides)
    
    print("Configuration:")
    print(f"  FPS: {config['fps']}")
    print(f"  Time Range: {config['start_time']}s - {config['end_time']}s")
    print(f"  SDK Quality: {config['sdk_quality']}")
    print(f"  Resolution: {config['sdk_resolution']}")
    print(f"  Cameras: {len(config['camera_config']['cameras'])}")
    print(f"  Output Size: {config['output_width']}x{config['output_height']}")
    print()
    
    # Track timing
    stage_times = {}
    total_start = time.perf_counter()
    
    # Create worker
    worker = PipelineWorker(config)
    
    # Progress callback
    def on_progress(current, total, message):
        pct = (current / total * 100) if total > 0 else 0
        print(f"\r  [{pct:5.1f}%] {message}", end="", flush=True)
    
    def on_stage_complete(stage_num, result):
        elapsed = time.perf_counter() - stage_start
        stage_times[stage_num] = elapsed
        success = result.get('success', False)
        status = "✓" if success else "✗"
        print(f"\n  Stage {stage_num} {status} - {format_time(elapsed)}")
        
        if stage_num == 1:
            count = result.get('count', len(result.get('frames', [])))
            print(f"    Extracted: {count} frames")
        elif stage_num == 2:
            count = result.get('perspective_count', 0)
            print(f"    Generated: {count} perspectives")
        elif stage_num == 3:
            count = result.get('mask_count', 0)
            skipped = result.get('skipped_count', 0)
            print(f"    Masks: {count} created, {skipped} skipped")
    
    def on_finished(result):
        pass  # Handle in main
    
    def on_error(error_msg):
        print(f"\n  ERROR: {error_msg}")
    
    # Connect signals
    worker.progress.connect(on_progress)
    worker.stage_complete.connect(on_stage_complete)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    
    # Run pipeline
    print("Starting pipeline...\n")
    
    print("=" * 50)
    print("STAGE 1: Frame Extraction (SDK Stitching)")
    print("=" * 50)
    stage_start = time.perf_counter()
    
    # Run synchronously (not in thread for this script)
    worker.run()
    
    total_elapsed = time.perf_counter() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    
    print("\nStage Timings:")
    for stage, elapsed in sorted(stage_times.items()):
        pct = (elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
        print(f"  Stage {stage}: {format_time(elapsed):>12} ({pct:5.1f}%)")
    
    print(f"\n  TOTAL: {format_time(total_elapsed)}")
    
    # Count output files
    print("\nOutput Files:")
    stage1_dir = output_path / 'stage1_frames'
    stage2_dir = output_path / 'stage2_perspectives'
    stage3_masks = list((output_path / 'stage2_perspectives').glob('*_mask.png')) if (output_path / 'stage2_perspectives').exists() else []
    
    if stage1_dir.exists():
        stage1_count = len(list(stage1_dir.glob('*.*')))
        print(f"  Stage 1 (Equirectangular): {stage1_count} files")
    
    if stage2_dir.exists():
        stage2_count = len([f for f in stage2_dir.glob('*.*') if '_mask' not in f.stem])
        print(f"  Stage 2 (Perspectives):    {stage2_count} files")
    
    if stage3_masks:
        print(f"  Stage 3 (Masks):           {len(stage3_masks)} files")
    
    print("\n" + "="*70)
    
    return True


def generate_camera_positions(count: int, fov: float) -> list:
    """Generate evenly-spaced camera positions around horizontal ring"""
    cameras = []
    yaw_step = 360.0 / count
    
    for i in range(count):
        cameras.append({
            'yaw': i * yaw_step - 180,  # Start from -180
            'pitch': 0,
            'roll': 0,
            'fov': fov
        })
    
    return cameras


def main():
    parser = argparse.ArgumentParser(
        description='Run 360ToolKit full pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py
  python run_full_pipeline.py "G:\\Videos\\my_video.insv" "C:\\Output"
  python run_full_pipeline.py --fps 1 --duration 30
        """
    )
    
    parser.add_argument('input', nargs='?', 
                       default=r"G:\.shortcut-targets-by-id\12X9Cn_caDGuRMIO-hF6196FMdQyGNUDA\PROJETOS - CHICO SOMBRA\VIDEOS 360\VID_20251215_170106_00_211.insv",
                       help='Input video file (.insv or .mp4)')
    
    parser.add_argument('output', nargs='?',
                       default=r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\pipeline_test",
                       help='Output directory')
    
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Frames per second to extract (default: 2)')
    
    parser.add_argument('--duration', type=float, default=15.0,
                       help='Duration in seconds to process (default: 15)')
    
    parser.add_argument('--cameras', type=int, default=8,
                       help='Number of camera perspectives (default: 8)')
    
    parser.add_argument('--fov', type=float, default=110.0,
                       help='Field of view in degrees (default: 110)')
    
    parser.add_argument('--quality', choices=['best', 'good', 'balanced', 'draft'],
                       default='best', help='SDK quality preset (default: best)')
    
    parser.add_argument('--resolution', choices=['8k', '6k', '4k', '2k'],
                       default='4k', help='Output resolution (default: 4k)')
    
    parser.add_argument('--no-mask', action='store_true',
                       help='Skip Stage 3 masking')
    
    parser.add_argument('--stage1-only', action='store_true',
                       help='Run only Stage 1 (extraction)')
    
    parser.add_argument('--stage2-only', action='store_true',
                       help='Run only Stage 2 (perspective split)')
    
    args = parser.parse_args()
    
    # Build config overrides
    overrides = {
        'fps': args.fps,
        'end_time': args.duration,
        'sdk_quality': args.quality,
        'sdk_resolution': args.resolution,
        'camera_config': {
            'cameras': generate_camera_positions(args.cameras, args.fov)
        },
    }
    
    if args.no_mask:
        overrides['enable_stage3'] = False
    
    if args.stage1_only:
        overrides['enable_stage2'] = False
        overrides['enable_stage3'] = False
    
    if args.stage2_only:
        overrides['enable_stage1'] = False
        overrides['enable_stage3'] = False
    
    # Run pipeline
    try:
        success = run_pipeline(args.input, args.output, overrides)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
