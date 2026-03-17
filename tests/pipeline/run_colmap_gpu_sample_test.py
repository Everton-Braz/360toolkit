#!/usr/bin/env python3
"""Run reconstruction-only COLMAP GPU test using local sample frames."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_orchestrator import PipelineWorker
from src.utils.colmap_paths import resolve_default_colmap_path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _validate_sparse_outputs(recon_dir: Path) -> tuple[bool, str]:
    sparse0 = recon_dir / "sparse" / "0"
    if not sparse0.exists():
        return False, f"Missing sparse model folder: {sparse0}"

    has_cameras = (sparse0 / "cameras.bin").exists() or (sparse0 / "cameras.txt").exists()
    has_images = (sparse0 / "images.bin").exists() or (sparse0 / "images.txt").exists()
    has_points = (sparse0 / "points3D.bin").exists() or (sparse0 / "points3D.txt").exists()

    if not has_cameras:
        return False, "Missing cameras model"
    if not has_images:
        return False, "Missing images model"
    if not has_points:
        return False, "Missing points3D model"

    return True, "Sparse model files present"


def main() -> int:
    parser = argparse.ArgumentParser(description="COLMAP GPU reconstruction test with sample frames")
    parser.add_argument(
        "--sample-dir",
        default=r"c:\Users\User\Documents\APLICATIVOS\360ToolKit\test_export\benchmark_test\input",
        help="Directory containing sample input frames",
    )
    parser.add_argument(
        "--output-root",
        default=r"c:\Users\User\Documents\APLICATIVOS\360ToolKit\test_export\reconstruction_gpu_test",
        help="Output root for test run",
    )
    parser.add_argument("--max-frames", type=int, default=5, help="How many sample frames to use")
    parser.add_argument(
        "--colmap-bin",
        default=str(resolve_default_colmap_path(PROJECT_ROOT)),
        help="Optional external COLMAP binary path (uses CLI path and GPU flags)",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        print(f"[FAIL] Sample dir not found: {sample_dir}")
        return 2

    images = sorted([p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if len(images) < 2:
        print(f"[FAIL] Need at least 2 sample frames, found {len(images)} in {sample_dir}")
        return 3

    selected = images[: max(2, args.max_frames)]

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"run_{run_stamp}"
    extracted_dir = run_dir / "extracted_frames"
    recon_dir = run_dir / "reconstruction"

    if run_dir.exists():
        shutil.rmtree(run_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(selected):
        dst = extracted_dir / f"frame_{idx:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    config = {
        "output_dir": str(run_dir),
        "enable_stage1": False,
        "enable_stage2": False,
        "enable_stage3": False,
        "use_rig_sfm": True,
        "alignment_mode": "perspective_reconstruction",
        "use_gpu": True,
        "use_gpu_colmap": True,
        "export_lichtfeld": False,
        "export_realityscan": False,
        "export_sidecars": False,
    }
    if args.colmap_bin:
        config["sphere_alignment_path"] = str(Path(args.colmap_bin))
        config["colmap_path"] = str(Path(args.colmap_bin))

    print(f"[INFO] Using {len(selected)} sample frames from: {sample_dir}")
    print(f"[INFO] Run directory: {run_dir}")
    print("[INFO] Running reconstruction (COLMAP GPU requested)...")

    worker = PipelineWorker(config)
    result = worker._execute_stage4()

    summary = {
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "colmap_output": str(result.get("colmap_output")) if result.get("colmap_output") else None,
        "run_dir": str(run_dir),
    }

    ok, validation = _validate_sparse_outputs(recon_dir)
    summary["validation_ok"] = ok
    summary["validation_message"] = validation

    print(json.dumps(summary, indent=2))

    if not result.get("success"):
        print("[FAIL] Reconstruction failed")
        return 4
    if not ok:
        print(f"[FAIL] Output validation failed: {validation}")
        return 5

    print("[PASS] COLMAP GPU reconstruction test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
