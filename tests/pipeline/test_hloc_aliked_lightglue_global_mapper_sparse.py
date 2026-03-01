from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from src.pipeline.batch_orchestrator import PipelineWorker

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

INPUT_ROOT = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\VIDEO_MP4")
INPUT_FRAMES = INPUT_ROOT / "extracted_frames"
OUTPUT_ROOT = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\hloc_aliked_lightglue_global_mapper_test")
COLMAP_BIN = Path(
    r"C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit\bin\COLMAP-windows-latest-CUDA-cuDSS-GUI\COLMAP.bat"
)


def _image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


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


def _run_hloc_smoke(image_dir: Path, output_dir: Path) -> tuple[bool, str]:
    try:
        from hloc import extract_features, pairs_from_retrieval  # type: ignore
    except Exception as exc:
        return False, f"HLOC import failed: {exc}"

    # Keep this step lightweight: retrieval descriptors + small pairs file only.
    try:
        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = output_dir / "hloc_retrieval.h5"
        extract_features.main(retrieval_conf, image_dir, output_dir, feature_path=retrieval_path)

        pairs_path = output_dir / "hloc_pairs_netvlad.txt"
        pairs_from_retrieval.main(retrieval_path, pairs_path, num_matched=20)

        if not pairs_path.exists() or pairs_path.stat().st_size == 0:
            return False, "HLOC pairs file missing/empty"
        return True, "HLOC retrieval+pairs OK"
    except Exception as exc:
        return False, f"HLOC retrieval/pairs failed: {exc}"


def test_hloc_aliked_lightglue_global_mapper_sparse() -> None:
    if not INPUT_ROOT.exists():
        pytest.skip(f"Input root not found: {INPUT_ROOT}")
    if not INPUT_FRAMES.exists():
        pytest.skip(f"Extracted frames folder not found: {INPUT_FRAMES}")
    if not COLMAP_BIN.exists():
        pytest.skip(f"COLMAP binary not found: {COLMAP_BIN}")

    images = _image_files(INPUT_FRAMES)
    if len(images) < 4:
        pytest.skip(f"Need at least 4 images in {INPUT_FRAMES}, found {len(images)}")

    # Use all frames by default; optional cap for quick local debugging.
    max_frames_env = os.environ.get("HLOC_COLMAP_TEST_MAX_FRAMES", "").strip()
    if max_frames_env:
        try:
            max_frames = max(4, int(max_frames_env))
            selected = images[:max_frames]
        except Exception:
            selected = images
    else:
        selected = images

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{run_stamp}"
    extracted_dir = run_dir / "extracted_frames"

    if run_dir.exists():
        shutil.rmtree(run_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(selected):
        dst = extracted_dir / f"frame_{idx:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    hloc_ok, hloc_msg = _run_hloc_smoke(extracted_dir, run_dir)
    if not hloc_ok:
        pytest.skip(f"HLOC step unavailable in environment: {hloc_msg}")

    config = {
        "output_dir": str(run_dir),
        "enable_stage1": False,
        "enable_stage2": False,
        "enable_stage3": False,
        "use_rig_sfm": True,
        "alignment_mode": "perspective_reconstruction",
        "mapping_backend": "glomap",
        "matching_method": "sequential",
        "use_lightglue_aliked": True,
        "camera_grouping": "per_folder",
        "require_learned_pipeline": True,
        "enable_hloc_fallback": True,
        "use_gpu": True,
        "use_gpu_colmap": True,
        "strict_gpu_only": True,
        "export_lichtfeld": False,
        "export_realityscan": False,
        "export_sidecars": False,
        "colmap_path": str(COLMAP_BIN),
    }

    worker = PipelineWorker(config)
    result = worker._execute_stage4()

    assert result.get("success"), f"Stage4 reconstruction failed: {result.get('error')}"

    recon_dir = run_dir / "reconstruction"
    ok, msg = _validate_sparse_outputs(recon_dir)
    assert ok, msg

    # Persist quick summary for debugging and regressions.
    summary_file = run_dir / "test_summary.txt"
    summary_file.write_text(
        "\n".join([
            f"hloc: {hloc_msg}",
            f"success: {result.get('success')}",
            f"colmap_output: {result.get('colmap_output')}",
            f"validation: {msg}",
            f"images_used: {len(selected)}",
        ]),
        encoding="utf-8",
    )
