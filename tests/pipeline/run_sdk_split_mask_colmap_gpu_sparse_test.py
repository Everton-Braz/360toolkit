#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_orchestrator import PipelineWorker

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _trim_to_n_images(folder: Path, keep_n: int) -> int:
    images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    for p in images[keep_n:]:
        p.unlink(missing_ok=True)
    return min(len(images), keep_n)


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _validate_sparse_model(sparse_root: Path) -> tuple[bool, str]:
    candidate_dirs = []
    if (sparse_root / "0").exists():
        candidate_dirs.append(sparse_root / "0")
    if sparse_root.exists():
        candidate_dirs.extend([d for d in sparse_root.iterdir() if d.is_dir()])

    for d in candidate_dirs:
        has_cameras = (d / "cameras.bin").exists() or (d / "cameras.txt").exists()
        has_images = (d / "images.bin").exists() or (d / "images.txt").exists()
        has_points = (d / "points3D.bin").exists() or (d / "points3D.txt").exists()
        if has_cameras and has_images and has_points:
            return True, f"Sparse model OK in {d}"

    return False, f"No valid sparse model in {sparse_root}"


def main() -> int:
    parser = argparse.ArgumentParser(description="SDK extract -> split 10 -> mask -> COLMAP GPU sparse test")
    parser.add_argument(
        "--input",
        default=r"G:\.shortcut-targets-by-id\12X9Cn_caDGuRMIO-hF6196FMdQyGNUDA\PROJETOS - CHICO SOMBRA\VIDEOS 360\VID_20251215_170106_00_211.insv",
        help="Input INSV/MP4",
    )
    parser.add_argument(
        "--output-root",
        default=r"C:\Users\User\Documents\ARQUIVOS_TESTE\colmap_gpu_sparse_test",
        help="Output root folder",
    )
    parser.add_argument(
        "--colmap-bin",
        default=r"C:\Users\User\Documents\APLICATIVOS\360ToolKit\bin\colmap\colmap.exe",
        help="COLMAP GPU binary path",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Extraction FPS")
    parser.add_argument("--duration", type=float, default=12.0, help="Extraction duration seconds")
    parser.add_argument("--split-width", type=int, default=1600)
    parser.add_argument("--split-height", type=int, default=1600)
    args = parser.parse_args()

    input_path = Path(args.input)
    colmap_bin = Path(args.colmap_bin)
    if not input_path.exists():
        print(f"[FAIL] Input not found: {input_path}")
        return 2
    if not colmap_bin.exists():
        print(f"[FAIL] COLMAP binary not found: {colmap_bin}")
        return 3

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"run_{run_stamp}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input_file": str(input_path),
        "output_dir": str(run_dir),
        "enable_stage1": True,
        "enable_stage2": True,
        "enable_stage3": True,
        "use_rig_sfm": False,
        "train_lighting": False,
        "fps": float(args.fps),
        "start_time": 0.0,
        "end_time": float(args.duration),
        "extraction_method": "sdk_stitching",
        "sdk_quality": "best",
        "sdk_resolution": "4k",
        "output_format": "jpg",
        "transform_type": "perspective",
        "stage2_format": "jpg",
        "output_width": int(args.split_width),
        "output_height": int(args.split_height),
        "masking_enabled": True,
        "mask_target": "split",
        "yolo_model_size": "small",
        "confidence_threshold": 0.5,
        "use_gpu": True,
    }

    worker = PipelineWorker(config)

    print("[INFO] Stage 1 - SDK extraction")
    stage1 = worker._execute_stage1()
    if not stage1.get("success"):
        print(f"[FAIL] Extraction failed: {stage1.get('error')}")
        return 4

    extracted_dir = run_dir / "extracted_frames"
    extracted_count = _count_images(extracted_dir)
    print(f"[INFO] Extracted frames: {extracted_count}")
    if extracted_count < 2:
        print("[FAIL] Too few extracted frames")
        return 5

    kept = _trim_to_n_images(extracted_dir, 10)
    print(f"[INFO] Kept for split: {kept} frames")
    if kept < 2:
        print("[FAIL] Not enough frames after trim")
        return 6

    print("[INFO] Stage 2 - perspective split")
    stage2 = worker._execute_stage2()
    if not stage2.get("success"):
        print(f"[FAIL] Split failed: {stage2.get('error')}")
        return 7

    perspectives_dir = run_dir / "perspective_views"
    split_count = _count_images(perspectives_dir)
    print(f"[INFO] Split views: {split_count}")
    if split_count < 10:
        print("[FAIL] Too few split images")
        return 8

    print("[INFO] Stage 3 - masking")
    stage3 = worker._execute_stage3()
    if not stage3.get("success"):
        print(f"[FAIL] Masking failed: {stage3.get('error')}")
        return 9

    masks_dir = run_dir / "masks"
    mask_count = _count_images(masks_dir)
    print(f"[INFO] Masks: {mask_count}")

    recon_dir = run_dir / "reconstruction_cli_gpu"
    recon_dir.mkdir(parents=True, exist_ok=True)
    db_path = recon_dir / "database.db"
    sparse_path = recon_dir / "sparse"

    print("[INFO] COLMAP feature_extractor (GPU)")
    rc, out, err = _run_cmd([
        str(colmap_bin),
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(perspectives_dir),
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.gpu_index", "0",
        "--ImageReader.single_camera", "1",
    ], cwd=PROJECT_ROOT)
    if rc != 0:
        print("[FAIL] feature_extractor failed")
        print(err[-2000:])
        return 10

    print("[INFO] COLMAP sequential_matcher (GPU)")
    rc, out, err = _run_cmd([
        str(colmap_bin),
        "sequential_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.gpu_index", "0",
        "--SequentialMatching.overlap", "8",
    ], cwd=PROJECT_ROOT)
    if rc != 0:
        print("[FAIL] sequential_matcher failed")
        print(err[-2000:])
        return 11

    print("[INFO] COLMAP mapper (sparse model)")
    rc, out, err = _run_cmd([
        str(colmap_bin),
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(perspectives_dir),
        "--output_path", str(sparse_path),
    ], cwd=PROJECT_ROOT)
    if rc != 0:
        print("[FAIL] mapper failed")
        print(err[-2000:])
        return 12

    ok, message = _validate_sparse_model(sparse_path)
    summary = {
        "success": ok,
        "input": str(input_path),
        "run_dir": str(run_dir),
        "extracted_frames": kept,
        "split_views": split_count,
        "masks": mask_count,
        "sparse_path": str(sparse_path),
        "validation": message,
    }
    print(json.dumps(summary, indent=2))

    if not ok:
        print(f"[FAIL] {message}")
        return 13

    print("[PASS] SDK -> Split(10) -> Mask -> COLMAP GPU sparse test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
