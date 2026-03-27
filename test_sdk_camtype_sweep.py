"""
test_sdk_camtype_sweep.py — Phase 2 of SDK bypass

Sweeps camtype integers by patching the A1 INSV calibration string
and testing if the SDK produces output. Tries all values 0-200.

Usage:
    python test_sdk_camtype_sweep.py [start] [end]
    python test_sdk_camtype_sweep.py 0 50
"""

import sys, os, subprocess, tempfile, shutil

# ─── Configuration ────────────────────────────────────────────────────────────
A1_INSV  = r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\VID_20260327_162728_005_Antigravity_A1_Sample.insv"
SDK_EXE  = r"C:\Users\Everton-PC\Documents\Windows_CameraSDK-2.1.1_MediaSDK-3.1.0\MediaSDK-3.1.0.0-20250904-win64\MediaSDK\bin\MediaSDKTest.exe"
SDK_MODELS = r"C:\Users\Everton-PC\Documents\Windows_CameraSDK-2.1.1_MediaSDK-3.1.0\MediaSDK-3.1.0.0-20250904-win64\MediaSDK\models"
WORK_DIR = r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\_cam_sweep"

# The two camtype tokens currently in the A1 INSV trailer
OLD_TYPES = [b"_112_", b"_155_"]

SWEEP_START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SWEEP_END   = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def patch_data(data: bytes, new_type: int) -> bytes:
    """Replace both camtype tokens in the INSV data with the new value."""
    replacement = ("_%03d_" % new_type).encode()
    for old in OLD_TYPES:
        # Only replace if the replacement length matches (it always does: _NNN_)
        data = data.replace(old, replacement)
    return data


def tail_patch(src_path: str, dst_path: str, new_type: int, tail_size: int = 300 * 1024):
    """
    Read only the last tail_size bytes of src, patch them, and write
    head + patched_tail to dst. This avoids reading the full 190 MB file for
    every sweep iteration.
    """
    file_size = os.path.getsize(src_path)
    head_size = max(0, file_size - tail_size)

    with open(src_path, "rb") as f:
        f.seek(head_size)
        tail = f.read()

    patched_tail = patch_data(tail, new_type)

    # Write head (copy verbatim) then patched tail
    with open(src_path, "rb") as fsrc, open(dst_path, "wb") as fdst:
        bytes_remaining = head_size
        buf_size = 4 * 1024 * 1024
        while bytes_remaining > 0:
            chunk = fsrc.read(min(buf_size, bytes_remaining))
            if not chunk:
                break
            fdst.write(chunk)
            bytes_remaining -= len(chunk)
        fdst.write(patched_tail)


def run_sdk(insv_path: str, out_dir: str, timeout: int = 15) -> tuple:
    """
    Run the SDK to extract the very first frame.
    Returns (returncode, stdout+stderr text, num_frames).
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        SDK_EXE,
        "-inputs", insv_path,
        "-stitch_type", "template",   # fastest mode
        "-output_size", "3840x1920",
        "-image_sequence_dir", out_dir,
        "-export_frame_index", "0-0-1",  # frame 0, step 1 → only 1 frame
        "-model_root_dir", SDK_MODELS,
        "-disable_cuda", "true",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = result.stdout + result.stderr
        frames = len([f for f in os.listdir(out_dir) if f.endswith((".jpg", ".png"))])
        return result.returncode, log, frames
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT", 0
    except Exception as e:
        return -2, str(e), 0


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    # Check prerequisites
    for path, name in [(A1_INSV, "A1 INSV"), (SDK_EXE, "SDK exe")]:
        if not os.path.exists(path):
            print(f"FATAL: {name} not found: {path}")
            sys.exit(1)

    print(f"Sweeping camtype {SWEEP_START}..{SWEEP_END}")
    print(f"A1 INSV size: {os.path.getsize(A1_INSV):,} bytes")
    print(f"Old tokens: {OLD_TYPES}")
    print(f"Work dir:   {WORK_DIR}")
    print()

    successes = []
    patched_insv = os.path.join(WORK_DIR, "patched.insv")

    for t in range(SWEEP_START, SWEEP_END + 1):
        token = ("_%03d_" % t).encode()
        print(f"[{t:3d}] Trying token {token.decode()} ... ", end="", flush=True)

        # Create patched INSV in work dir
        out_dir = os.path.join(WORK_DIR, f"type_{t:03d}")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        tail_patch(A1_INSV, patched_insv, t)

        rc, log, frames = run_sdk(patched_insv, out_dir)

        # Determine result
        if frames > 0:
            print(f"SUCCESS! {frames} frame(s) produced")
            successes.append(t)
        elif "no implemention" in log:
            print("no implemention (unsupported type)")
        elif "timeout" in log.lower() or rc == -1:
            print("TIMEOUT")
        elif "offset is not support" in log:
            print("offset not support")
        else:
            # Might have gotten further - print last 2 lines of log
            lines = [l.strip() for l in log.splitlines() if l.strip()]
            tail = " | ".join(lines[-2:]) if lines else "(no output)"
            print(f"rc={rc} | {tail[:100]}")

        # Clean up frames (keep patched.insv for next iteration override)
        if frames == 0 and os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

    # Summary
    print()
    print("=" * 60)
    if successes:
        print(f"WORKING CAMTYPE VALUES: {successes}")
        print("These values can be used to patch the A1 INSV for SDK processing!")
        # Show a sample frame path
        for t in successes:
            out_dir = os.path.join(WORK_DIR, f"type_{t:03d}")
            frames = [f for f in os.listdir(out_dir) if f.endswith((".jpg", ".png"))]
            if frames:
                print(f"  Type {t}: sample frame at {os.path.join(out_dir, frames[0])}")
    else:
        print(f"No working camtype found in range {SWEEP_START}-{SWEEP_END}.")
        print("Consider widening the range (e.g., 200-400).")

    # Cleanup patched INSV
    if os.path.exists(patched_insv):
        os.remove(patched_insv)


if __name__ == "__main__":
    main()
