#!/usr/bin/env python3
"""
Small-database integration test for Stage 4 Mode C (SphereSfM + Pose Transfer).

Workflow:
1) Stage 1: Extract a few equirectangular frames from input video
2) Stage 4: Run Mode C alignment (SphereSfM/COLMAP + pose transfer)
3) Validate expected outputs (sparse model + perspective images)

Usage:
    python tests/pipeline/run_mode_c_small_db_test.py
    python tests/pipeline/run_mode_c_small_db_test.py --input "C:/path/video.insv" --duration 6 --fps 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_orchestrator import PipelineWorker
from src.premium.sphere_sfm_integration import verify_spheresfm_installation


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _validate_stage4_outputs(stage4_dir: Path) -> tuple[bool, str]:
    sparse0 = stage4_dir / "sparse" / "0"
    images_dir = stage4_dir / "images"

    if not sparse0.exists():
        return False, f"Missing sparse model folder: {sparse0}"

    has_cameras = (sparse0 / "cameras.bin").exists() or (sparse0 / "cameras.txt").exists()
    has_images = (sparse0 / "images.bin").exists() or (sparse0 / "images.txt").exists()
    has_points = (sparse0 / "points3D.bin").exists() or (sparse0 / "points3D.txt").exists()

    if not has_cameras:
        return False, f"Missing cameras model in {sparse0}"
    if not has_images:
        return False, f"Missing images model in {sparse0}"
    if not has_points:
        return False, f"Missing points3D model in {sparse0}"

    num_perspectives = _count_images(images_dir)
    if num_perspectives == 0:
        return False, f"No perspective images generated in {images_dir}"

    return True, f"OK: sparse model + {num_perspectives} perspective images"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mode C small-db test (Stage 1 + Stage 4)")
    parser.add_argument(
        "--input",
        default=r"C:\Users\User\Documents\APLICATIVOS\Arquivos_Teste\VID_20251215_165650_00_210.insv",
        help="Input .insv/.mp4 path",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\User\Documents\APLICATIVOS\Arquivos_Teste\mode_c_small_db_test",
        help="Output root directory",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Extraction FPS")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration in seconds from start")
    parser.add_argument("--extraction-method", default="sdk_stitching", help="Stage 1 extraction method")
    parser.add_argument("--sphere-bin", default=None, help="Optional explicit SphereSfM/COLMAP binary path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[FAIL] Input not found: {input_path}")
        return 2

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output) / f"run_{run_stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    status = verify_spheresfm_installation()
    print(f"[INFO] SphereSfM/COLMAP path: {status.get('path')}")
    print(f"[INFO] Supports Mapper.sphere_camera: {status.get('supports_sphere_camera')}")
    if not status.get("installed", False):
        print(f"[FAIL] SphereSfM/COLMAP not available: {status.get('error')}")
        return 3

    config = {
        "input_file": str(input_path),
        "output_dir": str(output_root),
        "enable_stage1": True,
        "enable_stage2": False,
        "enable_stage3": False,
        "use_rig_sfm": True,
        "train_lighting": False,
        "alignment_mode": "pose_transfer",
        "fps": float(args.fps),
        "start_time": 0.0,
        "end_time": float(args.duration),
        "extraction_method": args.extraction_method,
        "allow_fallback": True,
        "sdk_quality": "good",
        "sdk_resolution": "4k",
        "output_format": "png",
        "use_gpu": True,
    }
    if args.sphere_bin:
        config["sphere_alignment_path"] = args.sphere_bin

    worker = PipelineWorker(config)

    print("[INFO] Running Stage 1 extraction...")
    stage1_result = worker._execute_stage1()
    if not stage1_result.get("success"):
        print(f"[FAIL] Stage 1 failed: {stage1_result.get('error')}")
        return 4

    stage1_dir = output_root / "stage1_frames"
    frame_count = _count_images(stage1_dir)
    print(f"[INFO] Stage 1 extracted frames: {frame_count}")
    if frame_count < 2:
        print(f"[FAIL] Too few frames extracted ({frame_count}). Need at least 2.")
        return 5

    print("[INFO] Running Stage 4 Mode C alignment...")
    stage4_result = worker._execute_stage4()
    if not stage4_result.get("success"):
        print(f"[FAIL] Stage 4 failed: {stage4_result.get('error')}")
        return 6

    stage4_dir = output_root / "stage4_alignment"
    ok, message = _validate_stage4_outputs(stage4_dir)
    if not ok:
        print(f"[FAIL] Validation failed: {message}")
        return 7

    print("[PASS] Mode C small-db test succeeded")
    print(f"[INFO] Output: {output_root}")
    print(f"[INFO] {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
