#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pycolmap


IMAGE_DIR = Path(r"C:\Users\Everton-PC\Documents\PROJETOS\Salao do Reino\extracted_frames_subset_300")
RUN_ROOT = Path(r"C:\Users\Everton-PC\Documents\PROJETOS\Salao do Reino\pycolmap_aliked_incremental_300")
SOURCE_DB = Path(r"C:\Users\Everton-PC\Documents\PROJETOS\Salao do Reino\colmap_subset_aliked_lightglue_gpu_300\database.db")


def _reset_run_root() -> tuple[Path, Path]:
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    sparse_dir = RUN_ROOT / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    return RUN_ROOT / "database.db", sparse_dir


def _build_incremental_options() -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = False
    options.max_num_models = 1
    options.min_model_size = 50
    options.min_num_matches = 15
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.ba_use_gpu = True
    options.ba_gpu_index = "0"
    options.mapper.init_min_num_inliers = 60
    options.mapper.init_max_error = 4.0
    options.mapper.init_min_tri_angle = 8.0
    options.mapper.abs_pose_min_num_inliers = 24
    options.mapper.abs_pose_max_error = 10.0
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    options.mapper.filter_max_reproj_error = 3.0
    options.mapper.filter_min_tri_angle = 1.5
    options.mapper.max_reg_trials = 4
    return options


def _summarize(best_model: pycolmap.Reconstruction, model_dir: Path) -> dict[str, float | int | str]:
    summary = {
        "model_dir": str(model_dir),
        "num_cameras": int(best_model.num_cameras()),
        "num_frames": int(best_model.num_frames()),
        "num_images": int(best_model.num_images()),
        "num_reg_frames": int(best_model.num_reg_frames()),
        "num_reg_images": int(best_model.num_reg_images()),
        "num_points3D": int(best_model.num_points3D()),
        "num_observations": int(best_model.compute_num_observations()),
        "mean_observations_per_reg_image": float(best_model.compute_mean_observations_per_reg_image()),
    }
    return summary


def main() -> int:
    if not IMAGE_DIR.exists():
        print(f"[FAIL] Image directory not found: {IMAGE_DIR}")
        return 2
    if not SOURCE_DB.exists():
        print(f"[FAIL] Source database not found: {SOURCE_DB}")
        return 2

    db_path, sparse_dir = _reset_run_root()
    shutil.copy2(SOURCE_DB, db_path)
    pycolmap.set_random_seed(0)

    print("[1/1] pycolmap.incremental_mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=IMAGE_DIR,
        output_path=sparse_dir,
        options=_build_incremental_options(),
    )
    if not reconstructions:
        print("[FAIL] pycolmap.incremental_mapping returned no models")
        return 3

    best_model_id, best_model = max(
        reconstructions.items(),
        key=lambda item: int(item[1].num_reg_images()),
    )
    model_dir = sparse_dir / str(best_model_id)
    summary = _summarize(best_model, model_dir)
    summary_path = RUN_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"SUMMARY_PATH={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())