from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.batch_orchestrator import PipelineWorker


def main() -> None:
    run_dir = Path(r"C:\Users\User\Documents\ARQUIVOS_TESTE\reextract_sdk\run_20260213_115128")
    extracted = run_dir / "extracted_frames"

    imgs = sorted([p for p in extracted.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for path in imgs[10:]:
        path.unlink(missing_ok=True)

    config = {
        "output_dir": str(run_dir),
        "enable_stage1": False,
        "enable_stage2": True,
        "enable_stage3": True,
        "output_width": 1600,
        "output_height": 1600,
        "stage2_format": "jpg",
        "transform_type": "perspective",
        "masking_enabled": True,
        "mask_target": "split",
        "yolo_model_size": "small",
        "confidence_threshold": 0.5,
        "use_gpu": True,
    }

    worker = PipelineWorker(config)

    stage2 = worker._execute_stage2()
    stage3 = worker._execute_stage3()

    perspective_dir = run_dir / "perspective_views"
    masks_dir = run_dir / "masks"

    summary = {
        "run_dir": str(run_dir),
        "stage2_success": stage2.get("success"),
        "stage2_error": stage2.get("error"),
        "split_views": len(list(perspective_dir.glob("*.*"))) if perspective_dir.exists() else 0,
        "stage3_success": stage3.get("success"),
        "stage3_error": stage3.get("error"),
        "masks": len(list(masks_dir.glob("*.*"))) if masks_dir.exists() else 0,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
