from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.batch_orchestrator import PipelineWorker
from src.utils.colmap_paths import resolve_default_colmap_path


def _backup_existing_reconstruction(output_dir: Path) -> Path | None:
    reconstruction_dir = output_dir / "reconstruction"
    if not reconstruction_dir.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir / f"reconstruction_backup_{timestamp}"
    shutil.move(str(reconstruction_dir), str(backup_dir))
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 4 reconstruction only against an existing output folder")
    parser.add_argument("--output-dir", required=True, help="Existing pipeline output folder containing perspective_views")
    parser.add_argument("--colmap-bin", default=str(resolve_default_colmap_path(PROJECT_ROOT)), help="Path to colmap executable")
    parser.add_argument("--mapping-backend", choices=["glomap", "colmap"], default="glomap")
    parser.add_argument("--fresh", action="store_true", help="Move any existing reconstruction folder aside before rebuilding")
    parser.add_argument("--use-hloc", action="store_true", help="Use ALIKED + LightGlue via HLOC when available")
    parser.add_argument("--skip-masks", action="store_true", help="Ignore existing masks during reconstruction")
    parser.add_argument("--export-realityscan", action="store_true")
    parser.add_argument("--export-lichtfeld", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    perspective_views_dir = output_dir / "perspective_views"
    if not perspective_views_dir.exists():
        print(json.dumps({
            "success": False,
            "error": f"Missing perspective_views folder: {perspective_views_dir}",
        }, indent=2))
        return 2

    backup_dir = None
    if args.fresh:
        backup_dir = _backup_existing_reconstruction(output_dir)

    config = {
        "output_dir": str(output_dir),
        "stage2_enabled": True,
        "stage3_enabled": (output_dir / "masks").exists() and (not args.skip_masks),
        "stage4_enabled": True,
        "alignment_mode": "perspective_reconstruction",
        "mapping_backend": args.mapping_backend,
        "use_gpu_colmap": True,
        "use_gpu": True,
        "use_lightglue_aliked": bool(args.use_hloc),
        "enable_hloc_fallback": bool(args.use_hloc),
        "prefer_colmap_learned": False,
        "require_learned_pipeline": False,
        "reuse_colmap_database": not args.fresh,
        "export_realityscan": bool(args.export_realityscan),
        "export_lichtfeld": bool(args.export_lichtfeld),
        "export_sidecars": False,
        "colmap_path": str(Path(args.colmap_bin)),
    }

    worker = PipelineWorker(config)
    result = worker._execute_stage4()
    payload = {
        "backup_dir": str(backup_dir) if backup_dir else None,
        "result": result,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())