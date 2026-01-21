import argparse
import sys
import logging
from pathlib import Path

# Import pipeline worker for CLI batch runs
from src.pipeline.batch_orchestrator import PipelineWorker

def main():
    parser = argparse.ArgumentParser(
        description="360FrameTools CLI - Batch photogrammetry pipeline (extract, split, mask)"
    )
    parser.add_argument("--input", "-i", required=True, help="Input file or folder (.insv, .mp4, .jpg, .png, .tiff)")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS (default: 1.0)")
    parser.add_argument("--split-count", type=int, default=8, help="Number of compass splits (default: 8)")
    parser.add_argument("--fov", type=int, default=110, help="Horizontal FOV for splits (default: 110)")
    parser.add_argument("--masking-engine", choices=["yolo", "onnx", "sam_vitb", "hybrid"], default="hybrid", help="Masking engine (default: hybrid)")
    parser.add_argument("--categories", nargs="*", default=["persons", "personal_objects"], help="Masking categories (default: persons personal_objects)")
    parser.add_argument("--model-size", choices=["nano", "small", "medium", "large", "xlarge"], default="small", help="YOLO model size (default: small)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--stage", choices=["all", "extract", "split", "mask"], default="all", help="Pipeline stage to run (default: all)")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration if available")
    parser.add_argument("--cleanup", action="store_true", help="Delete intermediate files after run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Prepare config dict for orchestrator
    config = {
        "input_file": args.input,
        "output_dir": args.output,
        "fps": args.fps,
        "split_count": args.split_count,
        "fov": args.fov,
        "masking_engine": args.masking_engine,
        "masking_categories": {cat: True for cat in args.categories},
        "model_size": args.model_size,
        "confidence": args.confidence,
        "use_gpu": args.use_gpu,
        "cleanup": args.cleanup,
        "stage": args.stage,
    }

    # Run pipeline

    worker = PipelineWorker(config)
    worker.run()
    print("\n✅ Pipeline run() completed. Check output folders for results.")

    # Print summary
    if result.get("success", False):
        print("\n✅ Pipeline completed successfully!")
        print(f"Output: {args.output}")
        print(f"Details: {result}")
    else:
        print("\n❌ Pipeline failed.")
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
