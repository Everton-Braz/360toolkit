#!/usr/bin/env python3
"""Automated validation runner for the packaged 360toolkit binary (.exe).

This script runs the built executable in CLI mode against real input files,
collects stage outputs, and writes summary reports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_X4 = Path(r"D:\ARQUIVOS_TESTE_2\Pecem_8K\VID_20260415_112943_00_245.insv")
DEFAULT_INPUT_A1 = Path(r"D:\ARQUIVOS_TESTE_2\VID_20260327_162728_005_Antigravity_A1_Sample.insv")
DEFAULT_INPUT_MP4 = Path(r"D:\ARQUIVOS_TESTE_2\PECEM_MP4\VID_20260415_112948.mp4")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\automated_tests_binary")


@dataclass(frozen=True)
class Profile:
    name: str
    input_path: Path
    fps: float
    split_count: int
    fov: int
    end_time: float
    model_size: str


def _discover_binary(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Binary not found: {candidate}")

    candidates = [
        Path("dist/360ToolkitGS-Simple/360ToolkitGS-Simple.exe"),
        Path("dist/360ToolkitGS-Simple.exe"),
        Path("build/360ToolkitGS/360ToolkitGS-Simple.exe"),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Could not auto-discover binary. Pass --binary with a .exe path."
    )


def _count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    total = 0
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
        total += sum(1 for _ in folder.rglob(ext))
    return total


def _count_masks(output_dir: Path) -> int:
    candidates = [
        output_dir / "masks_perspective",
        output_dir / "masks_equirect",
        output_dir / "masks_fisheye",
        output_dir / "masks_custom",
        output_dir,
    ]
    total = 0
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            total += sum(1 for _ in folder.rglob("*_mask.png"))
    return total


def _validate_outputs(output_dir: Path, stdout_text: str, return_code: int) -> dict[str, Any]:
    extracted = _count_images(output_dir / "extracted_frames")
    split = _count_images(output_dir / "perspective_views")
    masks = _count_masks(output_dir)

    stage1_ok = extracted > 0
    stage2_ok = split > 0
    # Stage 3 may produce zero masks if nothing was detected; use execution marker.
    stage3_ok = (
        "Stage 3 complete: success=True" in stdout_text
        or "CLI_PIPELINE_PASSED" in stdout_text
    )

    passed = return_code == 0 and stage1_ok and stage2_ok and stage3_ok

    return {
        "passed": passed,
        "checks": {
            "exit_code_zero": return_code == 0,
            "stage1_outputs": stage1_ok,
            "stage2_outputs": stage2_ok,
            "stage3_executed": stage3_ok,
        },
        "metrics": {
            "extracted_images": extracted,
            "perspective_images": split,
            "mask_files": masks,
        },
    }


def _run_profile(binary: Path, profile: Profile, run_dir: Path, timeout_min: int) -> dict[str, Any]:
    profile_dir = run_dir / profile.name
    profile_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(binary),
        "--cli",
        "--input",
        str(profile.input_path),
        "--output",
        str(profile_dir),
        "--fps",
        str(profile.fps),
        "--split-count",
        str(profile.split_count),
        "--fov",
        str(profile.fov),
        "--stage",
        "all",
        "--extraction-method",
        "sdk_stitching" if profile.input_path.suffix.lower() == ".insv" else "ffmpeg_stitched",
        "--sdk-quality",
        "best",
        "--output-format",
        "png",
        "--end-time",
        str(profile.end_time),
        "--model-size",
        profile.model_size,
        "--masking-engine",
        "sam3_cpp",
        "--confidence",
        "0.5",
        "--categories",
        "persons",
        "personal_objects",
        "--output-width",
        "1024",
        "--output-height",
        "1024",
        "--stage2-format",
        "png",
        "--stage2-numbering",
        "preserve_source",
        "--stage2-layout",
        "flat",
        "--use-gpu",
    ]

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_min) * 60,
        )
        return_code = proc.returncode
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout_text = exc.stdout or ""
        stderr_text = (exc.stderr or "") + "\nTIMEOUT"

    duration_s = round(time.time() - start, 2)

    (profile_dir / "stdout.log").write_text(stdout_text, encoding="utf-8", errors="replace")
    (profile_dir / "stderr.log").write_text(stderr_text, encoding="utf-8", errors="replace")

    validation = _validate_outputs(profile_dir, stdout_text, return_code)

    status = "pass" if validation["passed"] else ("partial" if timed_out else "fail")

    return {
        "name": profile.name,
        "input_path": str(profile.input_path),
        "output_dir": str(profile_dir),
        "command": cmd,
        "duration_s": duration_s,
        "timed_out": timed_out,
        "return_code": return_code,
        "status": status,
        "validation": validation,
    }


def _profiles_for(mode: str, full_target: str) -> list[Profile]:
    smoke = [
        Profile("smoke_insv_x4", DEFAULT_INPUT_X4, fps=0.5, split_count=2, fov=110, end_time=2.0, model_size="nano"),
        Profile("smoke_insv_a1", DEFAULT_INPUT_A1, fps=0.5, split_count=2, fov=110, end_time=2.0, model_size="nano"),
        Profile("smoke_mp4", DEFAULT_INPUT_MP4, fps=0.5, split_count=2, fov=110, end_time=2.0, model_size="nano"),
    ]

    full_x4 = Profile("full_insv_x4", DEFAULT_INPUT_X4, fps=1.0, split_count=8, fov=110, end_time=10.0, model_size="small")
    full_a1 = Profile("full_insv_a1", DEFAULT_INPUT_A1, fps=1.0, split_count=8, fov=110, end_time=10.0, model_size="small")
    full = full_x4 if full_target == "x4" else full_a1

    if mode == "smoke":
        return smoke
    if mode == "full":
        return [full]
    return smoke + [full]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated post-build binary tests for 360toolkit")
    parser.add_argument("--binary", default=None, help="Path to built .exe (optional auto-discovery)")
    parser.add_argument("--mode", choices=["smoke", "full", "both"], default="both")
    parser.add_argument("--full-target", choices=["x4", "a1"], default="x4")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-minutes-per-profile", type=int, default=30)
    args = parser.parse_args()

    binary = _discover_binary(args.binary)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"binary_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    profiles = _profiles_for(args.mode, args.full_target)

    preflight = {
        "binary": str(binary),
        "binary_exists": binary.exists(),
        "profiles": [p.name for p in profiles],
        "missing_inputs": [str(p.input_path) for p in profiles if not p.input_path.exists()],
    }

    if preflight["missing_inputs"]:
        summary = {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "mode": args.mode,
            "preflight": preflight,
            "profiles": [],
            "totals": {"pass": 0, "fail": len(preflight["missing_inputs"]), "partial": 0},
            "exit_code": 1,
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Missing input files:")
        for path in preflight["missing_inputs"]:
            print(f"  - {path}")
        return 1

    started_at = datetime.now().isoformat()
    results: list[dict[str, Any]] = []
    for profile in profiles:
        print(f"Running {profile.name}...")
        result = _run_profile(binary, profile, run_dir, args.max_minutes_per_profile)
        results.append(result)
        print(
            f"  status={result['status']} return={result['return_code']} "
            f"duration={result['duration_s']}s"
        )

    pass_count = sum(1 for r in results if r["status"] == "pass")
    partial_count = sum(1 for r in results if r["status"] == "partial")
    fail_count = sum(1 for r in results if r["status"] == "fail")

    if fail_count > 0:
        exit_code = 1
    elif partial_count > 0:
        exit_code = 2
    else:
        exit_code = 0

    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(),
        "mode": args.mode,
        "preflight": preflight,
        "profiles": results,
        "totals": {
            "pass": pass_count,
            "partial": partial_count,
            "fail": fail_count,
        },
        "exit_code": exit_code,
    }

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    text_lines = [
        "360toolkit Binary Test Summary",
        f"run_id: {run_id}",
        f"mode: {args.mode}",
        f"binary: {binary}",
        f"pass={pass_count} partial={partial_count} fail={fail_count}",
        "",
    ]
    for result in results:
        metrics = result["validation"]["metrics"]
        text_lines.append(
            f"- {result['name']}: {result['status']} rc={result['return_code']} "
            f"extracted={metrics['extracted_images']} split={metrics['perspective_images']} "
            f"masks={metrics['mask_files']}"
        )

    (run_dir / "run_summary.txt").write_text("\n".join(text_lines), encoding="utf-8")

    print(f"Summary written to: {run_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
