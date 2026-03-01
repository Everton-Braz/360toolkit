# HLOC + ALIKED + LightGlue + Global Mapper – Troubleshooting & Fixes

Date: 2026-02-17  
Project: 360Toolkit (Windows, RTX 5070 Ti)

This document turns the previous troubleshooting summary into a **solution‑oriented reference**. Each section lists: **symptom → root cause → concrete fixes → code / config hints**.

---

## 1. COLMAP learned extractor crash (ALIKED_N32 / ALIKED_N16ROT – 0xC0000409)

### Symptom

- COLMAP learned feature extractor with `ALIKED_N32` (and sometimes `ALIKED_N16ROT`) crashes on Windows with error:
  - `Process finished with exit code 0xC0000409` (stack buffer overrun).
- Pipeline silently falls back to SIFT and default matcher.

### Likely root causes

- Windows build of COLMAP ONNX pipeline (ALIKED) has a bug in:
  - TensorRT / ONNX runtime bindings, or
  - Buffer size assumptions for N32/N16ROT variants.
- Interaction with RTX 50xx + CUDA 12.4 and cuDNN/cuBLAS versions.

### Mitigation strategy

1. **Prefer HLOC learned pipeline instead of COLMAP’s ONNX path** for now.
2. Treat COLMAP‑ALIKED as *experimental* and gate it behind an "expert" toggle.
3. Add a **hard guard**: if learned extractor crashes once, do **not** retry in a loop; immediately switch to HLOC fallback.

### Concrete actions

- In `rig_colmap_integration.py` (or equivalent):
  - Wrap learned extractor call in a small runner that:
    - Runs COLMAP with ALIKED once.
    - Detects non‑zero exit codes or Windows `0xC0000409`.
    - On failure, records a **persistent flag** `colmap_learned_broken=True` for this binary.
    - Returns an explicit error to the GUI instead of silently downgrading.
- Expose in UI:
  - `[x] Prefer COLMAP learned extractor (ALIKED)`
  - `[x] Fallback to HLOC if COLMAP learned fails` (default ON).

### Longer‑term fix ideas

- Track upstream issues for "COLMAP ALIKED_N32 0xC0000409 Windows".
- When a new build appears, re‑test with:
  - Smaller batch sizes.
  - Fewer threads.
  - Disabled TensorRT optimizations if exposed via flags.

---

## 2. HLOC fallback integration errors

### 2.1 Features / matches argument mismatch

**Symptom**  
- Error: `Either provide both features and matches as Path or both as names.`

**Root cause**  
- Mixed usage of **feature name** (e.g., `"aliked"`) and **Path objects** for matches when calling `hloc.match_features.main`.

**Fix**

- Always pass **names** or **paths** consistently. Recommended pattern:
  - Use **names** (strings) for both, and let HLOC derive paths from the output root.

**Example**

```python
from hloc import extract_features, match_features, pairs_from_retrieval

features = extract_features.main(
    image_dir,
    output_dir,
    feature_conf,     # contains 'model': {'name': 'aliked', ...}
    feature_path=None # let HLOC manage
)

pairs = pairs_from_retrieval.main(
    image_dir,
    output_dir,
    feature_conf,
    retrieval_conf,
)

matches = match_features.main(
    output_dir / 'pairs-netvlad.txt',
    output_dir,
    feature_conf,
    matcher_conf,
)
```

- **Do not** pass a Path for `matches` while using a string for `features`.

---

### 2.2 Image ID mapping error during `import_features`

**Symptom**  
- Exception: `'NoneType' object has no attribute 'items'` inside HLOC during `import_features`.

**Root cause**  
- Newer HLOC versions: `import_images` no longer returns a mapping `image_id_by_name` (it returns `None`), but `import_features` was still expecting it.

**Fix**

- Read image IDs directly from **COLMAP database** after running `import_images`.

**Example helper**

```python
import sqlite3
from pathlib import Path


def get_image_id_map(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT name, image_id FROM images")
    mapping = {name: image_id for name, image_id in cur.fetchall()}
    conn.close()
    return mapping
```

- Then adapt the HLOC import step in your integration to forward this mapping (or replicate what HLOC does internally with `Database` helper) before importing features.

---

### 2.3 `.xmp` imported as images (BITMAP_ERROR spam)

**Symptom**  
- COLMAP logs full of:
  - `BITMAP_ERROR: Failed to read the image file format` for `frame_XXXXX.xmp`.

**Root cause**  
- The image folder iteration included sidecar `.xmp` files and passed them to COLMAP / HLOC as if they were images.

**Fix**

- Restrict image discovery to a **whitelist of extensions**.

**Example**

```python
VALID_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

images = [
    p for p in image_dir.iterdir()
    if p.suffix.lower() in VALID_EXT
]
```

- Apply this both in your **own code** and when preparing folder structures for COLMAP GUI/HLOC.

---

## 3. Global matcher loop detection issue

### Symptom

- Error from COLMAP:
  - `Could not open ""` when running sequential matcher with loop detection.

### Root cause

- Loop detection was enabled, but `vocab_tree_path` was empty.
- COLMAP expects a valid `.bin` tree filename when loop detection is ON.

### Fix

1. **Default**: Automatically disable loop detection for learned matchers.
   - Learned features + LightGlue + NetVLAD already provide good loop coverage.
2. Only enable loop detection when:
   - User provides a valid `vocab_tree.bin` path.

**Config guard example**

```python
if learned_matcher_enabled:
    loop_detection = False
    vocab_tree_path = ""
else:
    loop_detection = user_cfg.loop_detection
    vocab_tree_path = user_cfg.vocab_tree_path or ""

if loop_detection and not vocab_tree_path:
    raise ValueError("Loop detection requires a non-empty vocab_tree_path")
```

---

## 4. Stage 2 GPU split OOM → Stage 3 CPU fallback

### Symptom

- During Stage 2 (video split or heavy masking):
  - CUDA OOM in logs.
- Stage 3 (next GPU stage) silently falls back to CPU.

### Root cause

- Hard‑coded GPU batch size (`16`) exceeded VRAM for 8K frames and many cameras.
- CUDA memory fragmentation left the device in a state where later GPU probes failed.

### Fixes

1. **Remove hard override** for batch size.
2. Use an **auto‑safe estimator** based on:
   - Image resolution.
   - Number of cameras.
   - Available VRAM (queried via `torch.cuda.mem_get_info()` or NVIDIA APIs).
3. After OOM:
   - Call `torch.cuda.empty_cache()` and, if possible, re‑initialize workers before continuing.

**Example heuristic**

```python
import torch

def auto_batch_size(img_res, n_cams, safety_factor=0.6) -> int:
    free, total = torch.cuda.mem_get_info()
    # rough estimate: 32 MB per 4K image per camera
    est_per_item = 32 * 1024**2 * max(1, img_res[0] * img_res[1] / (3840 * 2160))
    max_items = int((free * safety_factor) / est_per_item)
    return max(1, min(max_items, 8))
```

- Use the auto batch size instead of hard `16`.

---

## 5. Windows logging encoding problem (UnicodeEncodeError cp1252)

### Symptom

- Python logging raises `UnicodeEncodeError` when emitting emoji or non‑ASCII characters to the Windows console (`cp1252`).

### Fixes

1. **Remove emojis** and non‑ASCII from log messages.
2. Optionally, force UTF‑8 console encoding when possible:

```python
import sys
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
```

3. Or configure a **file‑only logger** for verbose messages and keep console logs ASCII‑only.

---

## 6. Binary / version mismatch confusion

### Symptom

- GUI occasionally uses a different COLMAP binary than expected.
- Behavior changes across runs (different commit, no cuDSS, etc.).

### Fixes

1. In settings, require an **absolute path** to COLMAP executable (no reliance on `%PATH%`).
2. At startup, log:
   - Exact COLMAP path.
   - `colmap --version` output.
   - Optional: hash or short git commit if available.

**Example check**

```python
import subprocess

def get_colmap_version(colmap_exe: str) -> str:
    try:
        out = subprocess.check_output([colmap_exe, "--version"], text=True)
        return out.strip()
    except Exception as e:
        return f"error: {e}"
```

- Display this in an "About / Diagnostics" dialog so the user can screenshot when reporting bugs.

---

## 7. Open issues & recommendations

### 7.1 COLMAP ONNX ALIKED crash

- Keep treating this as "experimental" until:
  - Upstream fix is merged and confirmed stable.
- For production work:
  - Default to **HLOC + ALIKED + LightGlue** for learned features.

### 7.2 Large‑scale cancellation behavior

- Make UI cancel button clearly show "Cancelled by user" in logs to avoid confusion with crashes.
- Ensure background tasks check a shared cancellation flag and exit cleanly.

### 7.3 Memory pressure across stages

- For very large jobs (8K + many cameras + masks):
  - Offer a preset: `Memory‑safe mode`.
    - Lower resolution for masking.
    - Lower batch sizes.
    - Optional frame decimation.

---

## 8. Run checklist (operator‑friendly)

Before running a big 360 job:

1. **COLMAP path**
   - [ ] Check log shows expected `colmap.exe` and version.

2. **Learned pipeline**
   - [ ] `Enable HLOC fallback (ALIKED + LightGlue)` ON.
   - [ ] `All-or-fail: require learned pipeline` set according to needs.

3. **Inputs**
   - [ ] No `.xmp` or non‑image files in the input folder.
   - [ ] Reasonable resolution and frame count for GPU VRAM.

4. **Monitoring run**
   - [ ] Watch for `ALIKED` crash in logs.
   - [ ] Verify HLOC fallback starts if needed.
   - [ ] Confirm no `BITMAP_ERROR` spam.
   - [ ] If CUDA OOM appears, re‑run with lower batch / resolution.

5. **After run**
   - [ ] Confirm `database.db` and `sparse/0/` exist.
   - [ ] Optionally, load in COLMAP GUI or ColmapView to visually check poses.

---

*End of troubleshooting & fixes document.*
