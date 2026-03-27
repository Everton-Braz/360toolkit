"""
inspect_insv_trailer.py — analyse the Insta360 trailer in an INSV file.

Usage:
    python inspect_insv_trailer.py <path_to_insv>

Prints:
  - whether the INSV trailer magic is found
  - every file offset where b'_155_' occurs
  - the raw calibration/offset string (ExtraMetadata.offset)
  - the camtype value parsed from it
  - a hex + ASCII dump of the 100 bytes around the first _155_ occurrence
"""

import sys, struct, os, re

MAGIC = b"8db42d694ccc418790edff439fe026bf"
HEADER_SIZE = 32 + 4 + 4 + 32  # = 72 bytes

TARGET = b"_155_"
THREE_DIGIT_PATTERN = re.compile(rb"_(\d{3})_")

def find_all(data: bytes, needle: bytes):
    start = 0
    while True:
        idx = data.find(needle, start)
        if idx == -1:
            break
        yield idx
        start = idx + 1


def hex_dump(data: bytes, offset_start: int = 0, width: int = 16):
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i+width]
        hex_part  = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {offset_start + i:08x}  {hex_part:<{width*3}}  {ascii_part}")
    return "\n".join(lines)


def inspect(path: str):
    print(f"\n=== Inspecting: {path}")
    print(f"    Size: {os.path.getsize(path):,} bytes")

    with open(path, "rb") as f:
        data = f.read()

    # 1. Check INSV magic at end of file
    if data[-32:] == MAGIC:
        print(f"[OK] INSV magic found at offset {len(data)-32:#010x}")
    else:
        print("[WARN] INSV magic NOT found at end of file — not a valid INSV trailer?")

    # 2. Find all occurrences of _155_
    occurrences = list(find_all(data, TARGET))
    if occurrences:
        print(f"\n[TARGET] Found {len(occurrences)} occurrence(s) of b'_155_':")
        for off in occurrences:
            ctx = data[max(0, off-20):off+25]
            ascii_ctx = "".join(chr(b) if 32 <= b < 127 else "." for b in ctx)
            print(f"  File offset {off:#010x} ({off:,})  context: {ascii_ctx!r}")
    else:
        print(f"\n[INFO] b'_155_' NOT found in this file.")

    # 3. Find the calibration string by looking for the pattern
    #    "N_float_float_..._lenstype_..._lenstype_totalframes"
    #    (ASCII text blob inside protobuf payload)
    #    Typically starts with "2_" for dual-lens cameras.
    calib_pat = re.compile(rb"2_[\d.]+_[\d.]+_[\d.]+_[^_]")
    match = calib_pat.search(data)
    if match:
        # extract a long ASCII run
        start_off = match.start()
        end_off   = start_off
        while end_off < len(data) and (32 <= data[end_off] < 127):
            end_off += 1
        calib_bytes = data[start_off:end_off]
        calib_str   = calib_bytes.decode("ascii", errors="replace")
        print(f"\n[CALIB] Calibration string (offset {start_off:#010x}, {len(calib_str)} chars):")
        # pretty-print split by _ so it's readable
        fields = calib_str.split("_")
        print(f"  Raw: {calib_str[:200]}{'...' if len(calib_str)>200 else ''}")
        # parse camtype fields (index 20 per lens block)
        # format per lens: focal cx cy tilt_cx tilt_cy yaw pitch roll p0 p1 p2 d0 d1 d2 d3 d4 tile_w tile_h camtype
        # prefix "N" then lens blocks
        try:
            n_lenses = int(fields[0])
            print(f"  N lenses: {n_lenses}")
            per_lens = 19  # focal..camtype (19 fields per lens block after the N prefix)
            for i in range(n_lenses):
                base = 1 + i * per_lens
                if base + per_lens <= len(fields):
                    camtype = fields[base + 18]
                    print(f"  Lens {i}: camtype field = {camtype!r}  (full block: {fields[base:base+per_lens]})")
        except (ValueError, IndexError) as e:
            print(f"  [parse error] {e}")
    else:
        print("\n[INFO] Could not locate calibration string via '2_...' pattern.")

    # 4. Also search for all unique 3-digit camtype values anywhere in the calibration region
    all_3digit = set(THREE_DIGIT_PATTERN.findall(data))
    if all_3digit:
        decoded = sorted(int(x) for x in all_3digit)
        print(f"\n[SCAN] All _NNN_ 3-digit tokens found in file: {decoded}")

    # 5. Hex dump 100 bytes around first _155_ occurrence
    if occurrences:
        off = occurrences[0]
        dump_start = max(0, off - 40)
        dump_end   = min(len(data), off + 60)
        print(f"\n[HEX] 100-byte context at first b'_155_' (offset {off:#010x}):")
        print(hex_dump(data[dump_start:dump_end], dump_start))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # default paths
        paths = [
            r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\VID_20260327_162728_005_Antigravity_A1_Sample.insv",
            r"G:\.shortcut-targets-by-id\12X9Cn_caDGuRMIO-hF6196FMdQyGNUDA\PROJETOS - CHICO SOMBRA\VIDEOS 360\VID_20251215_170106_00_211.insv",
        ]
    else:
        paths = sys.argv[1:]

    for p in paths:
        if os.path.exists(p):
            inspect(p)
        else:
            print(f"\n[SKIP] File not found: {p}")
