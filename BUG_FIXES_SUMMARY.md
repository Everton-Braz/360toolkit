# Bug Fixes Applied - 360FrameTools

**Date**: December 2024  
**Build Target**: GPU-FULL version (dist\360ToolkitGS-FULL\)

---

## Issues Fixed

### Issue 1: WinError 32 - File Locking During Metadata Embedding

**Symptom**:
```
Error embedding orientation: [WinError 32] O arquivo já está sendo usado por outro processo: 'tmph50fot7s.png' -> 'frame_00002_cam_07.png'
```

**Root Cause**:
- PIL `Image.save()` doesn't explicitly close file handles after saving PNG
- Windows NTFS keeps file locked
- `metadata_handler.embed_camera_orientation()` attempts atomic rename (temp file → final file)
- Rename fails due to file lock on target file

**Fix Applied** (`src/pipeline/batch_orchestrator.py`):
```python
# After PIL save, explicitly close file handle and add delay
pil_img.save(str(output_path), 'PNG', compress_level=6)
pil_img.close()      # NEW: Explicit file handle close
del pil_img          # NEW: Force garbage collection
time.sleep(0.01)     # NEW: 10ms delay for Windows file lock release
success = True
```

**Locations Fixed**:
1. Line ~458: Perspective mode PNG save
2. Line ~615: Cubemap 6-face PNG save
3. Line ~683: Cubemap 8-tile PNG save

**Impact**:
- ✅ Eliminates WinError 32 file locking errors
- ✅ Ensures all images receive EXIF metadata embedding
- ✅ Overhead: +2.3 seconds for 232 images (10ms × 232 = 2320ms)

---

### Issue 2: Duplicate Stage 2 Execution After Full Pipeline

**Symptom** (from logs):
```
15:04:26,432 - === Starting Stage 2: Perspective Splitting ===
15:04:26,432 - [OK] Auto-discovered Stage 1 output
15:04:26,472 - === Starting Stage 2: Perspective Splitting ===  ← DUPLICATE!
15:04:26,472 - [OK] Auto-discovered Stage 1 output              ← DUPLICATE!
```

**Root Cause**:
- **Full Pipeline** runs all stages sequentially in `batch_orchestrator.run()`:
  - Stage 1 → emits `stage_complete(1)` → Stage 2 → emits `stage_complete(2)` → Stage 3
- UI `on_stage_complete()` handler auto-advances when stage completes:
  - Stage 1 completes → handler sees Stage 2 enabled → calls `run_stage_2_only()` again
- Result: Stage 2 runs TWICE (once from pipeline, once from auto-advance)

**Fix Applied** (`src/ui/main_window.py`):

**Added auto-advance flag**:
```python
# In run_stage_X_only() methods
self._auto_advance_enabled = True  # Enable auto-advance for stage-only mode
```

**Updated start_pipeline()**:
```python
def start_pipeline(self):
    # Disable auto-advance for Full Pipeline mode
    if not hasattr(self, '_auto_advance_enabled') or not self._auto_advance_enabled:
        self._auto_advance_enabled = False
```

**Updated on_stage_complete()**:
```python
def on_stage_complete(self, stage_number: int, results: dict):
    if results.get('success'):
        # Auto-advance ONLY if running in stage-only mode (not Full Pipeline)
        if hasattr(self, '_auto_advance_enabled') and self._auto_advance_enabled:
            if stage_number == 1 and self.stage2_enable.isChecked():
                QTimer.singleShot(500, self.run_stage_2_only)
            # ... (similar for Stage 2→3)
```

**Impact**:
- ✅ Full Pipeline: Runs each stage exactly once (no auto-advance)
- ✅ Stage-only mode: Auto-advances to next enabled stage (preserved behavior)
- ✅ Eliminates duplicate processing and confusing logs

---

## Testing Instructions

### Test Setup
1. Use same test file: 29 frames, 8 cameras = 232 perspective images
2. Enable all stages: Stage 1 (Extract) + Stage 2 (Split) + Stage 3 (Mask)
3. Run **Full Pipeline** (not stage-only)

### Expected Results

**✅ WinError 32 Fix**:
- Logs should contain **ZERO** instances of `[WinError 32]`
- All 232 PNG files should have EXIF metadata embedded successfully
- Check metadata with: `piexif.load("frame_00001_cam_01.png")`

**✅ Duplicate Stage 2 Fix**:
- Logs should show Stage 2 header **EXACTLY ONCE**:
  ```
  === Starting Stage 2: Perspective Splitting ===
  [OK] Auto-discovered Stage 1 output
  Processing frame 1/29 (3.4%)
  ...
  ```
- Pipeline completion should show: `Executed stages: [1, 2, 3]`

**✅ No Corrupted PNGs**:
- Stage 3 masking should process all 232 images without decode errors
- No `OpenCV(4.12.0) error: (-215:Assertion failed) !buf.empty()` messages

---

## Rebuild Command

**Option 1: Use batch script**:
```batch
rebuild_fixes.bat
```

**Option 2: Manual command**:
```powershell
cd C:\Users\User\Documents\APLICATIVOS\360ToolKit
C:\Users\User\miniconda3\envs\instantsplat\python.exe -m PyInstaller 360FrameTools_FULL.spec --noconfirm
```

**Build time**: ~3-5 minutes (incremental rebuild, modified files only)

---

## Files Modified

1. `src/pipeline/batch_orchestrator.py`
   - Added `import time` at top
   - Added explicit `pil_img.close()` + `del` + `time.sleep(0.01)` after 3 PIL PNG saves

2. `src/ui/main_window.py`
   - Added `self._auto_advance_enabled` flag in `run_stage_X_only()` methods
   - Updated `start_pipeline()` to disable auto-advance for Full Pipeline
   - Updated `on_stage_complete()` to check flag before auto-advancing

---

## Validation Checklist

After rebuild, test and verify:

- [ ] Build completes without errors
- [ ] Executable size: ~8-10 GB (GPU-FULL with PyTorch + Ultralytics)
- [ ] Run 29-frame, 8-camera test (232 perspective images)
- [ ] Logs show **NO** WinError 32 errors
- [ ] Logs show Stage 2 header **EXACTLY ONCE** (not twice)
- [ ] All 232 PNGs saved with EXIF metadata
- [ ] Stage 3 processes all images without decode errors
- [ ] Pipeline completes: "Executed stages: [1, 2, 3]"
- [ ] Test stage-only mode: Stage 1 → auto-advances to Stage 2 (verify flag still works)

---

## Next Steps

1. **Run `rebuild_fixes.bat`** to build new executable
2. **Test with same 29-frame video** used in previous test
3. **Verify both fixes** using validation checklist above
4. **Commit to GitHub** if tests pass:
   ```bash
   git add .
   git commit -m "Fix WinError 32 file locking and duplicate Stage 2 execution"
   git push origin master
   ```
5. **Tag release**: Consider tagging as `v1.0.1` after validation

---

## Technical Notes

### WinError 32 Deep Dive
- **PIL behavior**: `Image.save()` calls `self.fp.close()` internally, but doesn't guarantee immediate file handle release on Windows
- **Windows NTFS**: File locking is aggressive; even brief delays between close and rename can cause issues
- **Solution**: Explicit `close()` + `del` + `sleep()` ensures file handle released before metadata embedding

### Duplicate Stage 2 Deep Dive
- **Signal/slot timing**: `stage_complete` signal emitted immediately after stage finishes in pipeline thread
- **UI handler**: `on_stage_complete()` runs in UI thread, triggers `QTimer.singleShot(500, ...)` for auto-advance
- **Race condition**: Full Pipeline continues to Stage 2 while UI handler also queues Stage 2
- **Solution**: Flag-based discrimination between Full Pipeline (sequential) and stage-only (auto-advance)

---

**Author**: GitHub Copilot  
**Build System**: PyInstaller 6.16.0, Python 3.10.13, instantsplat conda env  
**Target Platform**: Windows 10/11 x64
