"""
test_a1_orientation_sweep.py — Systematic orientation sweep for Antigravity A1

The A1 drone has two lenses looking UP and DOWN (not front/back like a normal Insta360).
Stream 0 and Stream 1 are both 3840×3840 fisheye.

We enumerate the key combinations:
    - Stream order: [0,1] vs [1,0]
    - Per-stream pre-flip: none | transpose | hflip | vflip | rotate180
    - v360 parameters: ih_fov, pitch/yaw/roll, in_trans, ih_flip/iv_flip
    - Input layout: side-by-side vs top-bottom

For simplicity we test the most likely candidates first based on the drone
geometry (up/down lenses), then fall back to a brute-force grid.

Usage:
    python test_a1_orientation_sweep.py            # fast targeted sweep (14 variants)
    python test_a1_orientation_sweep.py --full     # full grid (~100 variants)

Output: C:\\Users\\Everton-PC\\Documents\\ARQUIVOS_TESTE\\a1_orient_sweep_v2\\
        Each variant produces one JPEG thumbnail (960x480) for quick visual review.
"""

import argparse
import os
import subprocess
import sys

# ─── Paths ────────────────────────────────────────────────────────────────────
A1_INSV = r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\VID_20260327_162728_005_Antigravity_A1_Sample.insv"
OUT_BASE = r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\a1_orient_sweep_v2"
FFMPEG   = "ffmpeg"  # assumes system PATH; adjust if needed

THUMB_W  = 960
THUMB_H  = 480

# ─── Helper ───────────────────────────────────────────────────────────────────

def run_variant(tag: str, filter_complex: str, out_dir: str) -> bool:
    """
    Extract frame 0 (at t=0) using the given filter_complex.
    Returns True if a JPEG was produced.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "frame.jpg")

    # We grab exactly 1 frame at the very start (-t 0.5 + fps=1 → 1 frame)
    cmd = [
        FFMPEG,
        "-y",
        "-i", A1_INSV,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-frames:v", "1",   # exactly one frame
        "-q:v", "3",
        out_file,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            print(f"  [OK]  {tag}")
            return True
        else:
            err_tail = (r.stderr or "")[-300:].strip().replace("\n", " | ")
            print(f"  [FAIL] {tag}  →  {err_tail}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {tag}")
        return False
    except Exception as e:
        print(f"  [ERROR] {tag}: {e}")
        return False


def make_filter(stream_a: int, stream_b: int,
                pre_a: str, pre_b: str,
                ih_fov: int, pitch: int, yaw: int, roll: int,
                stack_mode: str = "hstack",
                in_stereo: str | None = None,
                in_trans: bool = False,
                ih_flip: bool = False,
                iv_flip: bool = False,
                h_flip: bool = False,
                v_flip: bool = False) -> str:
    """
    Build an FFmpeg filter_complex string:
      - pre_a / pre_b: optional filter chain for each stream before stacking
        e.g. "" | "transpose=1" | "hflip" | "vflip" | "transpose=2,hflip"
            - streams are stacked then passed through v360 dfisheye→equirect
      - output is scaled to THUMB_W × THUMB_H for quick review
    """
    # Stream labels
    parts = []
    # Stream A processing
    if pre_a:
        parts.append(f"[0:v:{stream_a}]{pre_a}[sa]")
        label_a = "[sa]"
    else:
        label_a = f"[0:v:{stream_a}]"

    # Stream B processing
    if pre_b:
        parts.append(f"[0:v:{stream_b}]{pre_b}[sb]")
        label_b = "[sb]"
    else:
        label_b = f"[0:v:{stream_b}]"

    # stack the two lenses into a single frame for v360
    parts.append(f"{label_a}{label_b}{stack_mode}=inputs=2[stacked]")

    # v360 dfisheye → equirect
    v360_args = [
        "dfisheye",
        "equirect",
        f"ih_fov={ih_fov}",
        f"iv_fov={ih_fov}",
        f"yaw={yaw}",
        f"pitch={pitch}",
        f"roll={roll}",
    ]
    if in_stereo:
        v360_args.append(f"in_stereo={in_stereo}")
    if in_trans:
        v360_args.append("in_trans=1")
    if ih_flip:
        v360_args.append("ih_flip=1")
    if iv_flip:
        v360_args.append("iv_flip=1")
    if h_flip:
        v360_args.append("h_flip=1")
    if v_flip:
        v360_args.append("v_flip=1")

    v360 = f"[stacked]v360={':'.join(v360_args)},scale={THUMB_W}:{THUMB_H}[out]"
    parts.append(v360)

    # Return filter_complex string only (no -map, that's a separate cmd arg)
    return ";".join(parts)


# ─── Variant definitions ───────────────────────────────────────────────────────

def build_targeted_variants():
    """
    Targeted variants for the A1 up/down geometry.

    Reasoning:
      Normal Insta360 (front/back): hstack([front, back]), pitch=0
      A1 (up/down):  we need to figure out which stream is UP vs DOWN
                     and what rotation gets us to a standard landscape equirect.

    The v360 dfisheye filter treats the LEFT half as "front" (pitch=0, yaw=0)
    and the RIGHT half as "back" (pitch=0, yaw=180).

    For an UP/DOWN drone arrangement we expect one of:
      a) stream0=UP, stream1=DOWN  → pitch=-90 (tilt down 90° so UP becomes front)
      b) stream0=DOWN, stream1=UP  → pitch=+90 (tilt up 90°)
      c) Same but with yaw=90 or yaw=180 depending on drone mounting
      d) Lenses might need a 90° rotation each to match dfisheye pole convention
    """
    variants = []

    # The previous winner has the right gross orientation but a hard equator seam.
    # Refine around that candidate using the v360 axes we were not testing before.

    # ── Group 1: Refine the closest side-by-side candidate ─────────────────────
    for order in [(0, 1)]:
        for fov in [185, 190, 200, 220]:
            for roll in [0, 90, -90]:
                for flags in [
                    {},
                    {"in_trans": True},
                    {"ih_flip": True},
                    {"iv_flip": True},
                    {"ih_flip": True, "iv_flip": True},
                ]:
                    a, b = order
                    flag_tag = "_".join(
                        [name for name, enabled in flags.items() if enabled]
                    ) or "plain"
                    tag = (
                        f"ord{a}{b}_sbs_refine_fov{fov}_p-90_y0_r{roll:+d}_{flag_tag}"
                    )
                    fc = make_filter(
                        a, b, "", "", fov, -90, 0, roll,
                        stack_mode="hstack",
                        **flags,
                    )
                    variants.append((tag, fc))

    # ── Group 2: Per-stream rotations near the closest candidate ───────────────
    for order in [(0, 1)]:
        for fov in [185, 200, 220]:
            for roll in [0, 90, -90]:
                for pre in [
                    ("transpose=1", "transpose=1"),
                    ("transpose=2", "transpose=2"),
                    ("transpose=1", "transpose=2"),
                    ("transpose=2", "transpose=1"),
                ]:
                    a, b = order
                    prea, preb = pre
                    pre_tag = f"A_{prea}_B_{preb}"
                    tag = f"ord{a}{b}_{pre_tag}_fov{fov}_p-90_r{roll:+d}"
                    fc = make_filter(a, b, prea, preb, fov, -90, 0, roll)
                    variants.append((tag, fc))

    # ── Group 3: Try top-bottom packing for v360 input ─────────────────────────
    for order in [(0, 1)]:
        for fov in [185, 220]:
            for pitch in [-90, 90]:
                for roll in [0, 90, -90]:
                    for flags in [
                        {},
                        {"in_trans": True},
                    ]:
                        a, b = order
                        flag_tag = "_".join(
                            [name for name, enabled in flags.items() if enabled]
                        ) or "plain"
                        tag = (
                            f"ord{a}{b}_tb_fov{fov}_p{pitch:+d}_y0_r{roll:+d}_{flag_tag}"
                        )
                        fc = make_filter(
                            a, b, "", "", fov, pitch, 0, roll,
                            stack_mode="vstack",
                            in_stereo="tb",
                            **flags,
                        )
                        variants.append((tag, fc))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        if v[0] not in seen:
            seen.add(v[0])
            unique.append(v)
    return unique


def build_full_variants():
    """Full grid sweep — more exhaustive."""
    variants = []
    for order in [(0, 1), (1, 0)]:
        for fov in [180, 185, 190, 195, 200, 210, 220, 230]:
            for pitch in range(-90, 91, 30):
                for yaw in [0, 90, 180, -90]:
                    for roll in [0, 90, -90, 180]:
                        a, b = order
                        tag = f"ord{a}{b}_fov{fov}_p{pitch:+d}_y{yaw:+d}_r{roll:+d}"
                        fc = make_filter(a, b, "", "", fov, pitch, yaw, roll)
                        variants.append((tag, fc))
    # Also add per-stream pre-processing for pitch=0 and -90
    for order in [(0, 1), (1, 0)]:
        for pre in [
            ("transpose=1", "transpose=1"),
            ("transpose=2", "transpose=2"),
            ("hflip", "hflip"),
            ("vflip", "vflip"),
        ]:
            for pitch in [0, -90, 90]:
                for roll in [0, 90, -90]:
                    a, b = order
                    prea, preb = pre
                    pre_tag = f"A_{prea}_B_{preb}"
                    tag = f"ord{a}{b}_{pre_tag}_fov190_p{pitch:+d}_r{roll:+d}"
                    fc = make_filter(a, b, prea, preb, 190, pitch, 0, roll)
                    variants.append((tag, fc))
    for order in [(0, 1), (1, 0)]:
        for fov in [185, 200, 220]:
            for pitch in [-90, 90]:
                for roll in [0, 90, -90, 180]:
                    a, b = order
                    tag = f"ord{a}{b}_tb_fov{fov}_p{pitch:+d}_r{roll:+d}"
                    fc = make_filter(
                        a, b, "", "", fov, pitch, 0, roll,
                        stack_mode="vstack",
                        in_stereo="tb",
                    )
                    variants.append((tag, fc))
    seen = set()
    unique = []
    for v in variants:
        if v[0] not in seen:
            seen.add(v[0])
            unique.append(v)
    return unique


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the full grid sweep")
    args = parser.parse_args()

    if not os.path.exists(A1_INSV):
        print(f"FATAL: A1 INSV not found: {A1_INSV}")
        sys.exit(1)

    os.makedirs(OUT_BASE, exist_ok=True)

    variants = build_full_variants() if args.full else build_targeted_variants()
    print(f"Running {len(variants)} orientation variants -> {OUT_BASE}")
    print()

    successes = []
    for tag, fc in variants:
        out_dir = os.path.join(OUT_BASE, tag)
        ok = run_variant(tag, fc, out_dir)
        if ok:
            successes.append((tag, os.path.join(out_dir, "frame.jpg")))

    print()
    print("=" * 70)
    print(f"Done! {len(successes)}/{len(variants)} variants produced output.")
    if successes:
        print("\nProduced frames (check these visually):")
        for tag, path in successes:
            print(f"  {path}")
    else:
        print("No variants produced output — check ffmpeg PATH or filter syntax.")

    # Write a simple HTML gallery for easy visual review
    html_path = os.path.join(OUT_BASE, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>A1 Orientation Sweep</title></head><body>\n")
        f.write(f"<h1>A1 Orientation Sweep — {len(successes)} results</h1>\n")
        f.write("<p>Looking for: standard equirectangular (horizon is midline, sky on top, ground on bottom)</p>\n")
        for tag, img_path in successes:
            # Use relative path from html file
            rel = os.path.relpath(img_path, OUT_BASE)
            rel_fwd = rel.replace("\\", "/")
            f.write(f'<div style="display:inline-block;margin:8px;text-align:center">\n')
            f.write(f'  <img src="{rel_fwd}" width="{THUMB_W}" height="{THUMB_H}" style="border:2px solid #ccc"/><br/>\n')
            f.write(f'  <small>{tag}</small>\n')
            f.write(f'</div>\n')
        f.write("</body></html>\n")
    print(f"\nHTML gallery: {html_path}")
    print("Open it in a browser to visually compare all results.")


if __name__ == "__main__":
    main()
