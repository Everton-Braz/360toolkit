from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_orchestrator import PipelineWorker


def main() -> None:
    run_dir = Path(r"C:\Users\User\Documents\ARQUIVOS_TESTE\reextract_sdk\run_20260213_115128")

    config = {
        "output_dir": str(run_dir),
        "enable_stage1": False,
        "enable_stage2": False,
        "enable_stage3": False,
        "use_rig_sfm": False,
        "export_realityscan": True,
        "export_include_masks": True,
    }

    worker = PipelineWorker(config)
    result = worker._execute_realityscan_export_only()

    export_dir = run_dir / "realityscan_export"
    images_dir = export_dir / "images"

    image_files = []
    mask_files = []
    if images_dir.exists():
        for path in images_dir.glob("*.*"):
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                if path.name.endswith("_mask.png"):
                    mask_files.append(path)
                else:
                    image_files.append(path)

    summary = {
        "result": result,
        "export_dir_exists": export_dir.exists(),
        "images_dir_exists": images_dir.exists(),
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "sample_images": [p.name for p in sorted(image_files)[:5]],
        "sample_masks": [p.name for p in sorted(mask_files)[:5]],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
