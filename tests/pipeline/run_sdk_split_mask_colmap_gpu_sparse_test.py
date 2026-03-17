#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_orchestrator import PipelineWorker
from src.utils.colmap_paths import get_colmap_runtime_dirs, resolve_default_colmap_path

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
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=_build_colmap_env())
    return proc.returncode, proc.stdout, proc.stderr


def _build_colmap_env() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths: list[str] = []

    def _add(path: Path) -> None:
        if path.exists() and path.is_dir():
            p = str(path)
            if p not in extra_paths:
                extra_paths.append(p)

    for runtime_dir in get_colmap_runtime_dirs(resolve_default_colmap_path(PROJECT_ROOT)):
        _add(runtime_dir)

    # CUDA from current conda env (if any)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        _add(Path(conda_prefix) / "Library" / "bin")
        _add(Path(conda_prefix) / "Lib" / "site-packages" / "torch" / "lib")

    # Common local Python Torch install
    user_home = Path.home()
    _add(user_home / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "Lib" / "site-packages" / "torch" / "lib")

    # Discover torch/cuda libs in local miniconda envs
    miniconda_root = user_home / "miniconda3" / "envs"
    if miniconda_root.exists():
        for env_dir in miniconda_root.iterdir():
            if not env_dir.is_dir():
                continue
            _add(env_dir / "Library" / "bin")
            _add(env_dir / "Lib" / "site-packages" / "torch" / "lib")

    # CUDA toolkit installation
    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.exists():
        for version_dir in cuda_root.iterdir():
            if version_dir.is_dir():
                _add(version_dir / "bin")
                _add(version_dir / "bin" / "x64")

    if extra_paths:
        env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])

    return env


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


def _db_count(db_path: Path, table: str) -> int:
    if not db_path.exists():
        return 0
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        value = int(cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        con.close()
        return value
    except Exception:
        return 0


def _db_sum_rows(db_path: Path, table: str) -> int:
    if not db_path.exists():
        return 0
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        value = cur.execute(f"SELECT COALESCE(SUM(rows), 0) FROM {table}").fetchone()[0]
        con.close()
        return int(value or 0)
    except Exception:
        return 0


def _clear_match_tables(db_path: Path) -> None:
    if not db_path.exists():
        return
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("DELETE FROM matches")
        cur.execute("DELETE FROM two_view_geometries")
        con.commit()
        con.close()
    except Exception:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="SDK extract -> split -> mask -> COLMAP GPU sparse test")
    parser.add_argument(
        "--input",
        default=r"G:\.shortcut-targets-by-id\12X9Cn_caDGuRMIO-hF6196FMdQyGNUDA\PROJETOS - CHICO SOMBRA\VIDEOS 360\VID_20251215_170106_00_211.insv",
        help="Input INSV/MP4",
    )
    parser.add_argument(
        "--output-root",
        default=r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\colmap_gpu_sparse_test",
        help="Output root folder",
    )
    parser.add_argument(
        "--colmap-bin",
        default=str(resolve_default_colmap_path(PROJECT_ROOT)),
        help="COLMAP GPU binary path",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Extraction FPS")
    parser.add_argument("--duration", type=float, default=12.0, help="Extraction duration seconds")
    parser.add_argument("--keep-frames", type=int, default=20, help="How many extracted frames to keep for split/mask/COLMAP")
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

    kept = _trim_to_n_images(extracted_dir, max(2, int(args.keep_frames)))
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

    recon_input_dir = extracted_dir
    print(f"[INFO] COLMAP reconstruction input: {recon_input_dir}")

    recon_dir = run_dir / "reconstruction_cli_gpu"
    recon_dir.mkdir(parents=True, exist_ok=True)
    db_path = recon_dir / "database.db"
    sparse_path = recon_dir / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] COLMAP feature_extractor (GPU)")
    feature_gpu = True
    rc, out, err = _run_cmd([
        str(colmap_bin),
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(recon_input_dir),
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
        "--ImageReader.single_camera_per_folder", "1",
    ], cwd=PROJECT_ROOT)
    images_in_db = _db_count(db_path, "images")
    if rc != 0 or images_in_db == 0:
        print("[WARN] GPU feature extraction failed or produced 0 images; retrying on CPU")
        feature_gpu = False
        rc, out, err = _run_cmd([
            str(colmap_bin),
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(recon_input_dir),
            "--FeatureExtraction.use_gpu", "0",
            "--ImageReader.single_camera_per_folder", "1",
        ], cwd=PROJECT_ROOT)
        images_in_db = _db_count(db_path, "images")
        if rc != 0 or images_in_db == 0:
            print("[FAIL] feature_extractor failed")
            print(err[-2000:])
            return 10

    print("[INFO] COLMAP exhaustive_matcher (GPU)")
    matching_gpu = True
    rc, out, err = _run_cmd([
        str(colmap_bin),
        "exhaustive_matcher",
        "--database_path", str(db_path),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
    ], cwd=PROJECT_ROOT)
    tvg_rows = _db_sum_rows(db_path, "two_view_geometries")
    if rc != 0 or tvg_rows == 0:
        print("[WARN] GPU matching failed or produced 0 verified pairs; retrying on CPU")
        matching_gpu = False
        _clear_match_tables(db_path)
        rc, out, err = _run_cmd([
            str(colmap_bin),
            "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "0",
        ], cwd=PROJECT_ROOT)
        tvg_rows = _db_sum_rows(db_path, "two_view_geometries")
        if rc != 0 or tvg_rows == 0:
            print("[FAIL] exhaustive_matcher failed")
            print(err[-2000:])
            return 11

    print("[INFO] COLMAP mapper (sparse model)")
    rc, out, err = _run_cmd([
        str(colmap_bin),
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(recon_input_dir),
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
        "feature_gpu": feature_gpu,
        "matching_gpu": matching_gpu,
        "colmap_images_in_db": _db_count(db_path, "images"),
        "colmap_pairs_in_db": _db_count(db_path, "two_view_geometries"),
        "colmap_verified_matches": _db_sum_rows(db_path, "two_view_geometries"),
        "colmap_input": str(recon_input_dir),
        "sparse_path": str(sparse_path),
        "validation": message,
    }
    print(json.dumps(summary, indent=2))

    if not ok:
        print(f"[FAIL] {message}")
        return 13

    print(f"[PASS] SDK -> Split({kept}) -> Mask -> COLMAP GPU sparse test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
