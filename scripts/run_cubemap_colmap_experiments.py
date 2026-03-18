from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.batch_orchestrator import PipelineWorker
from src.utils.colmap_paths import resolve_default_colmap_path
from src.utils.cubemap_dataset import copy_cubemap_subset, rotate_dataset_orientation_180


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    mapping_backend: str
    matching_method: str
    use_hloc: bool
    include_masks: bool


PRESETS: List[ExperimentPreset] = [
    ExperimentPreset("baseline_glomap_nomask", "glomap", "sequential", False, False),
    ExperimentPreset("baseline_glomap_masks", "glomap", "sequential", False, True),
    ExperimentPreset("hloc_glomap_nomask", "glomap", "sequential", True, False),
    ExperimentPreset("mapper_nomask", "colmap", "sequential", False, False),
]


def _write_overall_summary(
    output_root: Path,
    dataset_root: Path,
    colmap_bin: Path,
    orientation_result: Dict[str, object] | None,
    results: List[Dict[str, object]],
) -> Path:
    payload = {
        "success": True,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "colmap_bin": str(colmap_bin),
        "orientation": orientation_result,
        "results": results,
    }
    summary_path = output_root / "experiments_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return summary_path


def _validate_sparse_outputs(recon_dir: Path) -> Dict[str, object]:
    sparse0 = recon_dir / "sparse" / "0"
    if not sparse0.exists():
        return {"ok": False, "reason": f"Missing sparse model folder: {sparse0}", "registered_images": 0}

    has_cameras = (sparse0 / "cameras.bin").exists() or (sparse0 / "cameras.txt").exists()
    has_images = (sparse0 / "images.bin").exists() or (sparse0 / "images.txt").exists()
    has_points = (sparse0 / "points3D.bin").exists() or (sparse0 / "points3D.txt").exists()
    if not (has_cameras and has_images and has_points):
        return {
            "ok": False,
            "reason": "Sparse model is incomplete",
            "registered_images": 0,
        }

    registered_images = 0
    images_txt = sparse0 / "images.txt"
    if images_txt.exists():
        for line in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("# Number of images:"):
                try:
                    registered_images = int(line.split(":", 1)[1].split(",", 1)[0].strip())
                except Exception:
                    registered_images = 0
                break
    return {"ok": True, "reason": "Sparse model files present", "registered_images": registered_images}


def _run_preset(
    dataset_root: Path,
    output_root: Path,
    colmap_bin: Path,
    preset: ExperimentPreset,
    max_frames: int | None,
) -> Dict[str, object]:
    run_dir = output_root / preset.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    subset_info = copy_cubemap_subset(
        source_dataset_root=dataset_root,
        target_dataset_root=run_dir,
        max_frames=max_frames,
        include_masks=preset.include_masks,
    )

    config = {
        "output_dir": str(run_dir),
        "stage2_enabled": True,
        "stage3_enabled": bool(preset.include_masks),
        "stage4_enabled": True,
        "alignment_mode": "perspective_reconstruction",
        "mapping_backend": preset.mapping_backend,
        "matching_method": preset.matching_method,
        "use_gpu_colmap": True,
        "use_gpu": True,
        "use_lightglue_aliked": bool(preset.use_hloc),
        "enable_hloc_fallback": bool(preset.use_hloc),
        "prefer_colmap_learned": False,
        "require_learned_pipeline": False,
        "reuse_colmap_database": False,
        "export_realityscan": False,
        "export_lichtfeld": False,
        "export_sidecars": False,
        "colmap_path": str(colmap_bin),
        "sphere_alignment_path": str(colmap_bin),
        "camera_grouping": "per_folder",
    }

    worker = PipelineWorker(config)
    try:
        result = worker._execute_stage4()
    except Exception as exc:
        result = {
            "success": False,
            "error": str(exc),
            "colmap_output": None,
            "num_aligned": 0,
        }

    validation = _validate_sparse_outputs(run_dir / "reconstruction")
    if int(validation.get("registered_images", 0) or 0) <= 0:
        validation["registered_images"] = int(result.get("num_aligned", 0) or 0)

    payload = {
        "preset": asdict(preset),
        "subset": subset_info,
        "result": result,
        "validation": validation,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated COLMAP cubemap experiments on a dataset")
    parser.add_argument(
        "--dataset-root",
        default=r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\mandarabyyoo",
        help="Dataset root containing perspective_views and masks",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory to write experiment runs (default: <dataset-root>/colmap_experiments)",
    )
    parser.add_argument("--max-frames", type=int, default=24, help="How many frame groups to include per preset")
    parser.add_argument(
        "--colmap-bin",
        default=str(resolve_default_colmap_path(PROJECT_ROOT)),
        help="Path to COLMAP executable or batch wrapper",
    )
    parser.add_argument("--fix-orientation", action="store_true", help="Rotate dataset images and masks 180 degrees before running")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help="Optional preset names to run. Defaults to all presets.",
    )
    parser.add_argument(
        "--skip-hloc",
        action="store_true",
        help="Skip presets that require HLOC/torch learned features.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root) if args.output_root else dataset_root / "colmap_experiments"
    output_root = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    orientation_result = None
    if args.fix_orientation:
        orientation_result = rotate_dataset_orientation_180(dataset_root)
        if not orientation_result.get("success"):
            print(json.dumps({"success": False, "orientation": orientation_result}, indent=2))
            return 1

    selected_presets = PRESETS
    if args.presets:
        wanted = {name.strip() for name in args.presets if name.strip()}
        selected_presets = [preset for preset in PRESETS if preset.name in wanted]
    if args.skip_hloc:
        selected_presets = [preset for preset in selected_presets if not preset.use_hloc]
    if not selected_presets:
        print(json.dumps({"success": False, "error": "No presets selected"}, indent=2))
        return 2

    results: List[Dict[str, object]] = []
    colmap_bin = Path(args.colmap_bin)
    for preset in selected_presets:
        result = _run_preset(
            dataset_root=dataset_root,
            output_root=output_root,
            colmap_bin=colmap_bin,
            preset=preset,
            max_frames=args.max_frames,
        )
        results.append(result)
        _write_overall_summary(
            output_root=output_root,
            dataset_root=dataset_root,
            colmap_bin=colmap_bin,
            orientation_result=orientation_result,
            results=results,
        )

    payload = {
        "success": True,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "colmap_bin": str(colmap_bin),
        "orientation": orientation_result,
        "results": results,
    }
    summary_path = _write_overall_summary(
        output_root=output_root,
        dataset_root=dataset_root,
        colmap_bin=colmap_bin,
        orientation_result=orientation_result,
        results=results,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())