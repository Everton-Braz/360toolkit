"""Run SphereSfM alignment on a subset of the mandarabyyoo dataset.

This script builds a subset of frames and masks, then runs the production
SphereSfM integration on that subset. It prefers GPU for feature extraction
and matching when requested, and falls back to CPU if the GPU attempt fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_spheresfm_mandarabyyoo import build_subset_dir, list_pngs
from src.pipeline.colmap_stage import ALIGNMENT_MODE_PANORAMA_SFM, ColmapSettings
from src.premium.sphere_sfm_integration import SphereSfMIntegrator


DATASET_ROOT = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\mandarabyyoo")
FRAMES_DIR = DATASET_ROOT / "extracted_frames"
MASKS_DIR = DATASET_ROOT / "masks"
OUTPUT_ROOT = DATASET_ROOT / "spheresfm_alignment_runs"


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SphereSfM alignment on mandarabyyoo subset")
    parser.add_argument("--subset-count", type=int, default=100, help="Number of frames and masks to use")
    parser.add_argument("--gpu", action="store_true", help="Try GPU for feature extraction and matching first")
    parser.add_argument("--max-image-size", type=int, default=3200, help="SphereSfM max image size")
    parser.add_argument("--max-num-features", type=int, default=8192, help="SphereSfM max feature count")
    parser.add_argument("--sequential-overlap", type=int, default=10, help="Sequential matcher overlap")
    parser.add_argument("--min-num-matches", type=int, default=15, help="Minimum verified matches for mapper")
    return parser.parse_args()


def count_registered_images(model_dir: Path) -> int:
    images_txt = model_dir / "images.txt"
    if not images_txt.exists():
        return 0
    with images_txt.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    return len(lines) // 2


def count_points(model_dir: Path) -> int:
    points_txt = model_dir / "points3D.txt"
    if not points_txt.exists():
        return 0
    with points_txt.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def build_settings(use_gpu: bool, args: argparse.Namespace) -> ColmapSettings:
    return ColmapSettings(
        alignment_mode=ALIGNMENT_MODE_PANORAMA_SFM,
        spheresfm_camera_model="SPHERE",
        spheresfm_use_gpu=use_gpu,
        spheresfm_matching_method="sequential",
        spheresfm_max_image_size=args.max_image_size,
        spheresfm_max_num_features=args.max_num_features,
        spheresfm_sequential_overlap=args.sequential_overlap,
        spheresfm_min_num_matches=args.min_num_matches,
    )


def run_once(frames_subset: Path, masks_subset: Path, output_dir: Path, use_gpu: bool, args: argparse.Namespace) -> dict:
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = build_settings(use_gpu=use_gpu, args=args)
    integrator = SphereSfMIntegrator(settings)

    progress_messages: list[str] = []

    def progress(message: str) -> None:
        progress_messages.append(message)
        print(f"[PROGRESS] {message}")

    result = integrator.run_alignment_mode_a(
        frames_dir=frames_subset,
        masks_dir=masks_subset,
        output_dir=output_dir,
        progress_callback=progress,
    )
    result["progress_messages"] = progress_messages
    result["used_gpu"] = use_gpu
    return result


def main() -> int:
    args = parse_args()

    if not FRAMES_DIR.exists():
        print(json.dumps({"success": False, "error": f"Frames directory not found: {FRAMES_DIR}"}, indent=2))
        return 1

    frames = list_pngs(FRAMES_DIR)
    masks = list_pngs(MASKS_DIR) if MASKS_DIR.exists() else []
    frames_subset, masks_subset = build_subset_dir(f"subset{args.subset_count}", frames, masks, args.subset_count)

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "frames_subset": str(frames_subset),
        "masks_subset": str(masks_subset),
        "subset_count_requested": args.subset_count,
        "subset_frame_count": len(list(frames_subset.glob("*.png"))),
        "subset_mask_count": len(list(masks_subset.glob("*.png"))),
        "gpu_requested": args.gpu,
        "max_image_size": args.max_image_size,
        "max_num_features": args.max_num_features,
        "sequential_overlap": args.sequential_overlap,
        "min_num_matches": args.min_num_matches,
    }
    print(json.dumps(summary, indent=2))

    gpu_output_dir = OUTPUT_ROOT / f"subset_{args.subset_count}_gpu"
    cpu_output_dir = OUTPUT_ROOT / f"subset_{args.subset_count}_cpu"

    attempts: list[dict] = []
    if args.gpu:
        gpu_result = run_once(frames_subset, masks_subset, gpu_output_dir, use_gpu=True, args=args)
        attempts.append({"mode": "gpu", "output_dir": str(gpu_output_dir), **gpu_result})
        if gpu_result.get("success"):
            result_payload = attempts[-1]
        else:
            cpu_result = run_once(frames_subset, masks_subset, cpu_output_dir, use_gpu=False, args=args)
            attempts.append({"mode": "cpu", "output_dir": str(cpu_output_dir), **cpu_result})
            result_payload = attempts[-1]
    else:
        cpu_result = run_once(frames_subset, masks_subset, cpu_output_dir, use_gpu=False, args=args)
        attempts.append({"mode": "cpu", "output_dir": str(cpu_output_dir), **cpu_result})
        result_payload = attempts[-1]

    model_dir = Path(result_payload["colmap_output"]) if result_payload.get("success") else None
    final_payload = {
        "summary": summary,
        "attempts": attempts,
        "success": bool(result_payload.get("success")),
        "selected_attempt": result_payload.get("mode"),
    }

    if model_dir and model_dir.exists():
        final_payload["registered_images"] = count_registered_images(model_dir)
        final_payload["points3D"] = count_points(model_dir)
        final_payload["model_dir"] = str(model_dir)

    result_json = OUTPUT_ROOT / f"subset_{args.subset_count}_result.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = make_json_safe(final_payload)
    result_json.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    print(json.dumps(safe_payload, indent=2))
    return 0 if final_payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())