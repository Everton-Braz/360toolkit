# Troubleshooting Summary: HLOC + ALIKED + LightGlue + COLMAP Global Mapper

Date: 2026-02-17  
Project: 360toolkit (Windows, RTX 5070 Ti)

## 1) Main objective

Target pipeline:

1. Learned features/matching (`ALIKED + LightGlue`)  
2. Global SfM (`global_mapper`)  
3. Keep GPU acceleration enabled where possible  
4. Avoid silent fallback to classic SIFT unless explicitly allowed

## 2) Main difficulties observed

## A. COLMAP learned extractor crash (critical)

**Symptom**
- `ALIKED_N32` (and often `ALIKED_N16ROT`) crashed with Windows error `0xC0000409`.

**Impact**
- Pipeline was falling back to plain SIFT / default matcher.
- User thought learned path was active, but final run was not always learned.

**Status**
- Mitigated by preferring HLOC learned fallback for larger datasets.
- Added strict all-or-fail option to stop silent downgrade.

---

## B. HLOC fallback integration errors (multiple)

### B1) Features/matches argument mismatch

**Symptom**
- `Either provide both features and matches as Path or both as names.`

**Cause**
- Mixed Path/name usage in HLOC `match_features.main` call.

**Fix applied**
- Explicit Path output used for matches.

### B2) Image IDs mapping error

**Symptom**
- `'NoneType' object has no attribute 'items'` during `import_features`.

**Cause**
- HLOC `import_images` does not return image-id map in this version.

**Fix applied**
- Read image-id mapping directly from COLMAP database (`images` table).

### B3) `.xmp` imported as images

**Symptom**
- Long spam: `BITMAP_ERROR: Failed to read the image file format` for `frame_XXXXX.xmp`.

**Cause**
- Folder iteration included sidecar `.xmp` files.

**Fix applied**
- Filter import list to valid image extensions only.

---

## C. Global matcher loop detection issue

**Symptom**
- Error opening empty vocab-tree path (`Could not open ""`) in sequential matcher.

**Cause**
- Loop detection path expected but missing.

**Fix applied**
- Auto-disable loop detection for learned matcher unless vocab-tree path is explicitly configured.

---

## D. Stage 2 GPU split OOM -> Stage 3 CPU fallback (pipeline-wide side effect)

**Symptom**
- GPU split failed with CUDA OOM.
- Then masking stage started on CPU.

**Cause**
- Auto-safe GPU batch size was overridden to `16`, exceeding VRAM for 8K + many cameras.

**Fix applied**
- Removed unsafe hard override.
- Use auto-safe batch by default.
- Added stronger CUDA cleanup on GPU failure.

**Important effect**
- If Stage 2 leaves VRAM fragmented/overused, Stage 3 often fails GPU probe and falls back to CPU.

---

## E. Windows logging encoding problem

**Symptom**
- `UnicodeEncodeError` in console logger (`cp1252`) when logging emoji warning line.

**Fix applied**
- Removed emoji from problematic log line.

---

## F. Binary/version mismatch confusion

**Symptom**
- GUI sometimes used a different COLMAP binary than expected.

**Impact**
- Different behavior across runs due to build/commit differences.

**Mitigation**
- Explicitly set COLMAP executable path in settings.
- Confirm command path in logs before diagnosing model/runtime issues.

## 3) Challenges that still remain

1. **Root crash in COLMAP ONNX ALIKED path (`0xC0000409`)** is still unresolved at source level.  
   - Current strategy: bypass with HLOC learned fallback when needed.

2. **Large-scale runtime and cancellation behavior** can look like crashes when user stops process during long HLOC extraction.

3. **Memory pressure chain reactions** between stages still possible on very large jobs (8K + many cameras + large mask batches).

## 4) What is working now

- All-or-fail controls are available in GUI:
  - `Enable HLOC fallback (ALIKED + LightGlue)`
  - `All-or-fail: require learned pipeline`
- Learned path can run via HLOC fallback and then continue to `global_mapper`.
- Automated tests pass for integration path used in this workflow.

## 5) Troubleshooting checklist for future runs

1. Confirm COLMAP binary path in logs (single known build).  
2. Confirm learned mode switches in GUI are ON.  
3. Check if ALIKED crash appears; if yes, verify HLOC fallback starts and completes DB import.  
4. Ensure no `.xmp BITMAP_ERROR` spam (if present, image filtering regression).  
5. If Stage 2 OOM occurs, lower split FPS / image size / camera count and re-run.  
6. If Stage 3 uses CPU unexpectedly, check immediately preceding Stage 2 GPU OOM entries.

## 6) Suggested search queries

Use these exact queries to research deeper fixes:

- `COLMAP ALIKED_N32 0xC0000409 Windows`  
- `COLMAP ONNX ALIKED crash stack buffer overrun`  
- `Hierarchical-Localization import_images return image_ids`  
- `pycolmap Database read images table image_id name`  
- `COLMAP sequential_matcher loop_detection vocab_tree_path empty`  
- `PyTorch CUDA out of memory after previous stage empty_cache ipc_collect`  
- `Windows Python logging cp1252 UnicodeEncodeError console`  
- `RTX 50 series PyTorch CUDA kernel compatibility sm_120`

## 7) Key code areas touched

- `src/premium/rig_colmap_integration.py`  
- `src/pipeline/batch_orchestrator.py`  
- `src/ui/main_window.py`  
- `src/main.py`  
- `tests/pipeline/test_hloc_aliked_lightglue_global_mapper_sparse.py`
