"""
360toolkit - Main Application Entry Point
Unified photogrammetry preprocessing pipeline.
"""

import sys
import os
import multiprocessing
from pathlib import Path

# CRITICAL: Must be called at the very start for PyInstaller on Windows
# This prevents child processes from spawning new GUI windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

import logging


def _bootstrap_windows_dlls() -> None:
    if os.name != "nt":
        return

    candidate_dirs = [
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
        Path(sys.prefix) / "Lib" / "site-packages" / "onnxruntime" / "capi",
    ]

    existing_path = os.environ.get("PATH", "")
    prepend: list[str] = []

    for dll_dir in candidate_dirs:
        if dll_dir.exists():
            dir_str = str(dll_dir)
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(dir_str)
            except Exception:
                pass
            if dir_str not in existing_path:
                prepend.append(dir_str)

    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [existing_path])


_bootstrap_windows_dlls()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('360toolkit.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_cli_mode():
    """Run pipeline in CLI mode (no GUI). Used for automated testing from .exe."""
    import argparse
    
    parser = argparse.ArgumentParser(description="360toolkit CLI mode")
    parser.add_argument("--cli", action="store_true", help="Enable CLI mode (no GUI)")
    parser.add_argument("--input", "-i", required=True, help="Input file (.insv, .mp4)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS")
    parser.add_argument("--split-count", type=int, default=8, help="Number of camera splits")
    parser.add_argument("--fov", type=int, default=110, help="Horizontal FOV")
    parser.add_argument("--stage", choices=["all", "extract", "split", "mask"], default="all",
                        help="Pipeline stage to run")
    parser.add_argument("--extraction-method", default="sdk_stitching", help="Extraction method")
    parser.add_argument("--sdk-quality", default="best", help="SDK quality setting")
    parser.add_argument("--output-format", default="jpg", help="Output image format")
    parser.add_argument("--end-time", type=float, default=0, help="End time in seconds (0=full)")
    parser.add_argument("--model-size", default="small", help="YOLO model size")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence")
    parser.add_argument("--categories", nargs="*", default=["persons", "personal_objects"],
                        help="Masking categories")
    parser.add_argument("--output-width", type=int, default=1920, help="Output width")
    parser.add_argument("--output-height", type=int, default=1920, help="Output height")
    parser.add_argument("--stage2-format", default="png", help="Split output format (png/jpg)")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Enable GPU")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--stage2-input-dir", default=None, help="Input dir for splitting (skip extraction)")
    parser.add_argument("--stage3-input-dir", default=None, help="Input dir for masking (skip splitting)")
    parser.add_argument("--mask-target", default="split", help="Mask target type")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("360toolkit v1.3.0 - CLI Mode")
    logger.info("=" * 60)
    
    # Build camera config
    cameras = [
        {"yaw": i * (360 // args.split_count), "pitch": 0, "roll": 0, "fov": args.fov}
        for i in range(args.split_count)
    ]
    
    # Determine stage flags
    enable_s1 = args.stage in ("all", "extract")
    enable_s2 = args.stage in ("all", "split")
    enable_s3 = args.stage in ("all", "mask")
    
    config = {
        "input_file": args.input,
        "output_dir": args.output,
        "fps": args.fps,
        "extraction_method": args.extraction_method,
        "sdk_quality": args.sdk_quality,
        "output_format": args.output_format,
        "end_time": args.end_time if args.end_time > 0 else None,
        "enable_stage1": enable_s1,
        "enable_stage2": enable_s2,
        "enable_stage3": enable_s3,
        "use_rig_sfm": False,
        "train_lighting": False,
        "skip_transform": False,
        "transform_type": "perspective",
        "output_width": args.output_width,
        "output_height": args.output_height,
        "stage2_format": args.stage2_format,
        "camera_config": {"cameras": cameras},
        "masking_engine": "yolo_pytorch",
        "model_size": args.model_size,
        "confidence_threshold": args.confidence,
        "masking_categories": {cat: True for cat in args.categories},
        "use_gpu": args.use_gpu,
    }
    
    # Add stage input dirs if provided
    if args.stage2_input_dir:
        config["stage2_input_dir"] = args.stage2_input_dir
    if args.stage3_input_dir:
        config["stage3_input_dir"] = args.stage3_input_dir
        config["mask_target"] = args.mask_target
    
    # Need QApplication for PipelineWorker (QThread)
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    
    from src.pipeline.batch_orchestrator import PipelineWorker
    
    results = {}
    
    def on_progress(current, total, message):
        print(f"[{current}/{total}] {message}")
    
    def on_stage_complete(stage, result):
        results[stage] = result
        success = result.get("success", False) if isinstance(result, dict) else False
        print(f"Stage {stage} complete: success={success}")
    
    def on_finished(result):
        results["final"] = result
        success = result.get("success", False) if isinstance(result, dict) else False
        print(f"Pipeline finished: success={success}")
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, dict) and "success" in v:
                    print(f"  {k}: success={v['success']}")
                    if "error" in v:
                        print(f"    error: {v['error']}")
        if success:
            print("CLI_PIPELINE_PASSED")
        else:
            print("CLI_PIPELINE_FAILED")
        app.quit()
    
    def on_error(msg):
        print(f"ERROR: {msg}")
    
    worker = PipelineWorker(config)
    worker.progress.connect(on_progress)
    worker.stage_complete.connect(on_stage_complete)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    
    worker.start()
    app.exec()
    
    final = results.get("final", {})
    success = final.get("success", False) if isinstance(final, dict) else False
    sys.exit(0 if success else 1)


def main():
    """Main application entry point"""
    
    # Check for CLI mode before launching GUI
    if '--cli' in sys.argv:
        return run_cli_mode()
    
    logger.info("=" * 60)
    logger.info("Starting 360toolkit v1.3.0")
    logger.info("=" * 60)
    
    try:
        # Enable High-DPI scaling BEFORE creating QApplication
        # This is essential for 4K monitors
        os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
        os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
        
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        from src.ui.main_window import MainWindow
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("360toolkit")
        app.setOrganizationName("360toolkit Development Team")
        
        # Set default font size for better readability on high-DPI
        font = app.font()
        font.setPointSize(10)  # Slightly larger default font
        app.setFont(font)
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        logger.info("Application window opened")
        
        # Run event loop
        sys.exit(app.exec())
    
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install dependencies: pip install -r requirements.txt")
        print("\n" + "="*60)
        print("ERROR: Missing Dependencies")
        print("="*60)
        print(f"\n{e}\n")
        print("Please install dependencies:")
        print("  pip install -r requirements.txt")
        print("\nFor GPU support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("="*60)
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\nERROR: {e}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
