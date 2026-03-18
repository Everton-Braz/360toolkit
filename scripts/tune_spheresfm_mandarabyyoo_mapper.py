"""Tune CPU-only SphereSfM mapper profiles on the existing subset_100 database."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_spheresfm_mandarabyyoo import DATASET_ROOT, FRAMES_DIR, MASKS_DIR, build_subset_dir, list_pngs
from src.premium.sphere_sfm_integration import (
    DEFAULT_SPHERESFM_MAPPER_FLAGS,
    DEFAULT_SPHERESFM_RELEASE_PATH,
    _compose_binary_command,
    _override_cli_flag_values,
)


DB_PATH = DATASET_ROOT / "spheresfm_alignment_runs" / "subset_100_cpu" / "database.db"
SUBSET_ROOT = DATASET_ROOT / "spheresfm_diagnostics" / "subsets" / "subset100"
SUBSET_FRAMES = SUBSET_ROOT / "frames"
OUTPUT_ROOT = DATASET_ROOT / "spheresfm_alignment_runs" / "mapper_tuning"
RESULTS_PATH = OUTPUT_ROOT / "subset_100_mapper_tuning_results.json"
SPHERESFM = DEFAULT_SPHERESFM_RELEASE_PATH


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


def ensure_subset_frames() -> Path:
    if SUBSET_FRAMES.exists() and len(list(SUBSET_FRAMES.glob("*.png"))) >= 100:
        return SUBSET_FRAMES

    frames = list_pngs(FRAMES_DIR)
    masks = list_pngs(MASKS_DIR) if MASKS_DIR.exists() else []
    frames_subset, _ = build_subset_dir("subset100", frames, masks, 100)
    return frames_subset


def write_log(log_path: Path, cmd: list[str], result: subprocess.CompletedProcess) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(cmd)
        + "\n\nSTDOUT:\n"
        + (result.stdout or "")
        + "\n\nSTDERR:\n"
        + (result.stderr or ""),
        encoding="utf-8",
    )


def count_model_stats(model_dir: Path) -> tuple[int, int]:
    images_txt = model_dir / "images.txt"
    points_txt = model_dir / "points3D.txt"

    num_aligned = 0
    if images_txt.exists():
        with images_txt.open("r", encoding="utf-8", errors="replace") as handle:
            lines = [line for line in handle if line.strip() and not line.startswith("#")]
        num_aligned = len(lines) // 2

    num_points = 0
    if points_txt.exists():
        with points_txt.open("r", encoding="utf-8", errors="replace") as handle:
            num_points = sum(1 for line in handle if line.strip() and not line.startswith("#"))

    return num_aligned, num_points


def run_command(args: list[str]) -> subprocess.CompletedProcess:
    env = dict(**__import__("os").environ)
    env["PATH"] = str(SPHERESFM.parent) + ";" + env.get("PATH", "")
    cmd = _compose_binary_command(SPHERESFM, args)
    return subprocess.run(
        cmd,
        cwd=str(SPHERESFM.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        stdin=subprocess.DEVNULL,
    )


def build_profiles() -> list[dict]:
    return [
        {
            "name": "strict_baseline",
            "flags": DEFAULT_SPHERESFM_MAPPER_FLAGS,
            "min_num_matches": 15,
        },
        {
            "name": "lenient_init",
            "flags": _override_cli_flag_values(
                DEFAULT_SPHERESFM_MAPPER_FLAGS,
                {
                    "--Mapper.init_min_num_inliers": "50",
                    "--Mapper.init_num_trials": "500",
                    "--Mapper.init_min_tri_angle": "8",
                    "--Mapper.abs_pose_min_num_inliers": "15",
                    "--Mapper.abs_pose_max_error": "12",
                },
            ),
            "min_num_matches": 15,
        },
        {
            "name": "ultra_lenient",
            "flags": _override_cli_flag_values(
                DEFAULT_SPHERESFM_MAPPER_FLAGS,
                {
                    "--Mapper.init_min_num_inliers": "30",
                    "--Mapper.init_num_trials": "1000",
                    "--Mapper.init_max_error": "8",
                    "--Mapper.init_max_forward_motion": "0.99",
                    "--Mapper.init_min_tri_angle": "4",
                    "--Mapper.abs_pose_min_num_inliers": "10",
                    "--Mapper.abs_pose_max_error": "16",
                    "--Mapper.abs_pose_min_inlier_ratio": "0.1",
                    "--Mapper.max_reg_trials": "5",
                    "--Mapper.tri_min_angle": "0.5",
                    "--Mapper.tri_max_transitivity": "2",
                    "--Mapper.tri_ignore_two_view_tracks": "0",
                    "--Mapper.filter_max_reproj_error": "8",
                    "--Mapper.filter_min_tri_angle": "0.5",
                },
            ),
            "min_num_matches": 10,
        },
        {
            "name": "single_model_lenient",
            "flags": _override_cli_flag_values(
                DEFAULT_SPHERESFM_MAPPER_FLAGS,
                {
                    "--Mapper.init_min_num_inliers": "50",
                    "--Mapper.init_num_trials": "500",
                    "--Mapper.init_min_tri_angle": "8",
                    "--Mapper.abs_pose_min_num_inliers": "15",
                    "--Mapper.abs_pose_max_error": "12",
                    "--Mapper.max_reg_trials": "6",
                    "--Mapper.tri_ignore_two_view_tracks": "0",
                    "--Mapper.filter_max_reproj_error": "6",
                    "--Mapper.multiple_models": "0",
                },
            ),
            "min_num_matches": 12,
        },
        {
            "name": "aggressive_registration",
            "flags": _override_cli_flag_values(
                DEFAULT_SPHERESFM_MAPPER_FLAGS,
                {
                    "--Mapper.init_min_num_inliers": "40",
                    "--Mapper.init_num_trials": "800",
                    "--Mapper.init_max_forward_motion": "0.99",
                    "--Mapper.init_min_tri_angle": "6",
                    "--Mapper.abs_pose_min_num_inliers": "12",
                    "--Mapper.abs_pose_max_error": "16",
                    "--Mapper.abs_pose_min_inlier_ratio": "0.15",
                    "--Mapper.max_reg_trials": "8",
                    "--Mapper.tri_min_angle": "0.75",
                    "--Mapper.tri_max_transitivity": "2",
                    "--Mapper.tri_ignore_two_view_tracks": "0",
                    "--Mapper.filter_max_reproj_error": "8",
                    "--Mapper.filter_min_tri_angle": "0.75",
                    "--Mapper.multiple_models": "0",
                },
            ),
            "min_num_matches": 10,
        },
    ]


def main() -> int:
    if not DB_PATH.exists():
        payload = {
            "success": False,
            "error": "Subset database not found. Run the 100-frame alignment script first.",
            "database_path": str(DB_PATH),
        }
        print(json.dumps(payload, indent=2))
        return 1

    if not SPHERESFM.exists():
        payload = {
            "success": False,
            "error": "SphereSfM binary not found.",
            "spheresfm_path": str(SPHERESFM),
        }
        print(json.dumps(payload, indent=2))
        return 1

    frames_dir = ensure_subset_frames()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    profile_results: list[dict] = []
    for profile in build_profiles():
        profile_root = OUTPUT_ROOT / profile["name"]
        sparse_root = profile_root / "sparse"
        shutil.rmtree(profile_root, ignore_errors=True)
        sparse_root.mkdir(parents=True, exist_ok=True)

        mapper_args = [
            "mapper",
            "--database_path", str(DB_PATH),
            "--image_path", str(frames_dir),
            "--output_path", str(sparse_root),
            "--Mapper.min_num_matches", str(profile["min_num_matches"]),
            "--Mapper.sphere_camera", "1",
        ] + profile["flags"].split()
        mapper_result = run_command(mapper_args)
        write_log(profile_root / "mapper.log", _compose_binary_command(SPHERESFM, mapper_args), mapper_result)

        best_model = None
        best_aligned = -1
        best_points = -1

        for model_dir in sorted(path for path in sparse_root.iterdir() if path.is_dir()):
            converter_args = [
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(model_dir),
                "--output_type", "TXT",
            ]
            converter_result = run_command(converter_args)
            write_log(profile_root / "model_converter.log", _compose_binary_command(SPHERESFM, converter_args), converter_result)
            num_aligned, num_points = count_model_stats(model_dir)
            if num_aligned > best_aligned or (num_aligned == best_aligned and num_points > best_points):
                best_model = model_dir
                best_aligned = num_aligned
                best_points = num_points

        profile_results.append(
            {
                "profile": profile["name"],
                "success": best_model is not None and best_aligned > 0,
                "returncode": mapper_result.returncode,
                "num_aligned": max(best_aligned, 0),
                "num_points": max(best_points, 0),
                "best_model": str(best_model) if best_model else None,
                "mapper_log": str(profile_root / "mapper.log"),
                "model_converter_log": str(profile_root / "model_converter.log"),
            }
        )

    ranked_results = sorted(profile_results, key=lambda item: (-item["num_aligned"], -item["num_points"]))
    payload = {
        "database_path": str(DB_PATH),
        "frames_dir": str(frames_dir),
        "profiles_tested": len(profile_results),
        "profiles_succeeded": sum(1 for item in ranked_results if item["success"]),
        "results": ranked_results,
    }
    RESULTS_PATH.write_text(json.dumps(make_json_safe(payload), indent=2), encoding="utf-8")
    print(json.dumps(make_json_safe(payload), indent=2))
    return 0 if any(item["success"] for item in ranked_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())