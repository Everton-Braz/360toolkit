# 360FrameTools - AI Coding Agent Instructions

## Project Overview

**360FrameTools** is a unified desktop application that combines frame extraction from Insta360 cameras (.INSV/.mp4) with advanced perspective splitting and AI masking for photogrammetry workflows. 

**Three-stage pipeline**: EXTRACT FRAMES → SPLIT FRAMES → MASKING → DONE

This unified tool merges two separate applications:
1. **Frame Extraction Module**: Extracts frames from dual-fisheye .INSV files using official SDK (bypasses Insta360 Studio)
2. **360toFrame**: Converts equirectangular images to perspective views with compass-based positioning

**Key distinction**: This is a complete photogrammetry preprocessing pipeline—from raw Insta360 video to masked perspective frames—in one streamlined batch workflow.

## Project Structure

**Development location**: `C:\Users\User\Documents\APLICATIVOS\360ToolKit\`
**Source reference folders** (DO NOT MODIFY, read-only):
- `Original_Projects\Extraction_Reference\` - Frame extraction source
- `Original_Projects\360toFrame\` - Perspective splitting source

**Note**: Reuse/copy code from source folders. Ignore existing library implementations (`lib/` folders).

---

## Unified Application Architecture

### Stage 1: Frame Extraction (from Insta360toFrames)

**Input**: `.INSV` (dual-fisheye) or `.mp4` files from Insta360 cameras
**Output**: Equirectangular images (stitched panoramas)

**SDK Location**: `C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1`

**Four extraction methods** (PRIORITY ORDER):
1. **SDK Stitching** (PRIMARY - HIGHEST PRIORITY): Official Insta360 MediaSDK 3.0.5 for GPU-accelerated stitching
   - **Best quality**: AI-based stitching with seamless blending (chromatic calibration)
   - **Bypasses Insta360 Studio entirely**
   - **Direct .INSV → stitched equirectangular frames** at user-configurable FPS
   - **Stitch types**: AI Stitching (best), Optical Flow, Dynamic, Template
   - **Quality enhancements**: 
     * Chromatic calibration (EnableStitchFusion) - CRITICAL for seamless blending
     * FlowState stabilization (optional)
     * Color Plus enhancement (AI model)
     * Denoising and defringing
   - **Output formats**: JPG, PNG (SetImageSequenceInfo)
   - **GPU acceleration**: CUDA/Vulkan required (auto-detected)
   - **Model files**: ai_stitch_model_v1.ins (X3/X4), ai_stitch_model_v2.ins (X5)
   - **Frame extraction**: SetExportFrameSequence() for specific frame indices
2. **FFmpeg Method** (FALLBACK): Proven dual-stream filter chain (SDK-quality results without SDK)
3. **Dual-Fisheye Export**: Extract raw fisheye lens images (no stitching)
4. **OpenCV Method**: Frame-by-frame extraction (basic, no stitching, lowest quality)

**Key features**:
- **File analysis**: Display metadata from input file (duration, resolution, format, camera model)
- **Time range selection**: Start/end time in seconds (extract specific segment)
- **Interval-based extraction**: Frames per second (0.1 - 30 FPS)
- **Resolution options**: Original, 8K (7680×3840), 6K (6080×3040), 4K (3840×1920), 2K (1920×960)
- **Output format**: JPG (default) or PNG (SetImageSequenceInfo with IMAGE_TYPE enum)
- **Metadata preservation**: Camera info only (NO GPS/GYRO—discard those)
- **SDK quality settings**:
  * **AI Stitching** (BEST): GPU-accelerated AI-based seam blending, highest quality
  * **Optical Flow** (GOOD): High accuracy, moderate speed
  * **Dynamic Stitching** (BALANCED): Suitable for motion scenes
  * **Template** (FAST): Fast preview, lower quality for near-field scenes

### Stage 2: Perspective Splitting (from 360toFrame)

**Input**: Equirectangular images (from Stage 1 or external)
**Output**: Multiple perspective views (rectilinear photos)

**Compass-based camera positioning**:
- **Default**: 8 cameras, 110° FOV, horizontal ring
- **Look-up/Look-down**: Additional compass rings (e.g., +30°, -30° pitch)
- **Customizable**: User modifies split count, FOV, yaw/pitch/roll per ring

**Transform engines** (reuse from source):
- **`e2p_transform.py`**: Equirectangular → Pinhole perspective (spherical mapping + cache)
- **`e2c_transform.py`**: Equirectangular → Cubemap (6-face standard)
  - **NEW**: Support 8-tile cubemap variants
  - Configurable overlap and FOV

**Key features**:
- **Resolution options**: Custom width/height for output images
- **Output format**: PNG, JPEG, or TIFF
- **Real-time preview**: Dual preview system (see UI spec for details)
  - **Perspective mode**: Main camera preview + circular compass with clickable slices
  - **Cubemap mode**: Grid preview showing tile layout with overlap visualization

**InteractiveCircularCompass widget**:
- 4 camera states: Export (Blue), Preview (Yellow), Disabled (Red), Mask (Green)
- Click icons to cycle states
- Clickable slices (pizza-piece style) - each color represents a camera
- Optional look-up/look-down buttons create additional rings

### Stage 3: AI Masking (Enhanced)

**Current capabilities** (from 360toFrame):
- YOLOv8 instance segmentation (person detection)
- 5 model sizes: nano → xlarge
- Binary masks: 0 (Black/mask) = remove, 255 (White/keep) = valid

**NEW requirements**:
- **GPU acceleration**: Leverage CUDA for faster masking
- **Extended categories**: 
  - Persons (existing)
  - Personal objects (bags, phones, etc.)
  - Animals (all COCO animal classes)
- **Category selection UI**: Checkboxes for which classes to mask
- **Smart mask skipping**: Only create mask files for images with detected objects
  - If no persons/animals detected → skip mask creation entirely
  - Saves disk space and processing time
  - Log which images were skipped vs masked

**Mask format**: RealityScan compatible (`<image_name>_mask.png`)

### Core Architecture Components

#### 1. **Desktop Application** (PyQt6, NOT PyQt5)
- **UI Framework**: PyQt6 for modern, minimalist interface
- **Design principle**: Easy to maintain, beautiful, spec-driven
- **Follow**: UI specification document (to be created)

#### 2. **Batch Processing Pipeline**
- **Single workflow**: Extract → Split → Mask in one automated sequence
- **Threading**: QThread for non-blocking UI
- **Progress tracking**: Real-time status updates across all stages

#### 3. **Metadata Chain**
- Stage 1: Extract camera metadata (NO GPS/GYRO)
- Stage 2: Embed camera orientation (yaw/pitch/roll) into EXIF
- Stage 3: Preserve all metadata in masked outputs

#### 4. **Configuration Management**
- Camera group presets (JSON-based)
- Masking profiles (category combinations)
- Extraction presets (FPS, quality)

---

## Critical Workflows

### Workflow 1: Complete Batch Pipeline (PRIMARY)
```
User configures pipeline:
  1. Select .INSV/.mp4 input
  2. Set extraction interval (FPS)
  3. Configure compass splits (default: 8 cameras, 110° FOV)
  4. Enable masking categories (persons/objects/animals)
  
User clicks "Start Batch Processing"
  
STAGE 1: Extract Frames
  → Insta360 SDK stitches dual-fisheye → equirectangular images
  → Save to temp/intermediate folder
  → Preserve camera metadata (NO GPS/GYRO)
  
STAGE 2: Split Perspectives
  → For each equirectangular frame:
     → For each compass camera position:
        → E2PTransform.equirect_to_pinhole() OR E2CTransform (if cubemap)
        → Save perspective view
        → Embed camera orientation in EXIF
  
STAGE 3: Generate Masks
  → If masking enabled:
     → For each perspective image:
        → YOLOv8 detect (GPU-accelerated)
        → Filter by selected categories
        → Generate binary mask → save as `<image>_mask.png`
  
→ DONE: Output folder contains perspective images + masks
```

**Key insight**: Three-stage pipeline is sequential but fully automated. User sets config once, app handles all stages.

### Workflow 2: Stage-Only Processing (Advanced)
```
Users can run individual stages:
- Extract only: .INSV → equirectangular
- Split only: Equirectangular → perspectives (load existing images)
- Mask only: Perspectives → masks (post-process existing splits)
```

### Workflow 3: Real-time Preview (Configuration UI)
```
User adjusts compass parameters
  → Load sample equirectangular
  → E2PTransform cached preview
  → InteractiveCircularCompass visual update
  → User fine-tunes before batch
```

---

## Project-Specific Conventions

### File Organization
```
360FrameTools/
├── src/                          # Application source
│   ├── extraction/               # Stage 1: Insta360 SDK integration
│   ├── transforms/               # Stage 2: E2P, E2C engines
│   ├── masking/                  # Stage 3: YOLOv8 + GPU
│   ├── ui/                       # PyQt6 interface
│   ├── pipeline/                 # Batch orchestration
│   └── config/                   # Presets, settings
├── specs/                        # UI spec documents (spec-driven)
├── resources/                    # Icons, templates
└── tests/                        # Unit/integration tests
```

### File I/O & Naming
- **Input formats**: `.insv` (Insta360 native), `.mp4`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`
- **Intermediate**: Equirectangular frames in `temp/` or user-specified folder
- **Output**: Perspective images + masks in output directory
- **Mask files**: Pattern `<stem>_mask.png` (RealityScan compatible)
- **Metadata sidecars**: JSON files (camera params + EXIF, NO GPS/GYRO)

### Parameter Ranges & Defaults

**Stage 1 (Extraction)**:
```python
fps_interval:   0.1 to 30 FPS (frames per second, user choice)
sdk_quality:    'draft', 'good', 'best' (Insta360 SDK presets)
output_format:  'equirectangular', 'dual_fisheye'
```

**Stage 2 (Splitting)**:
```python
yaw:           -180 to 180° (0° = front, ±90° = sides, 180° = back)
pitch:         -90 to +90° (-90° = below, +90° = above, 0° = horizon)
roll:          -180 to 180° (camera roll, typically 0°)
h_fov:         30 to 150° (horizontal field of view, default 110°)
v_fov:         auto-calculated from h_fov × (height/width)
split_count:   1 to 12 cameras per compass ring
compass_rings: ['main', 'look_up', 'look_down'] with pitch offsets
cubemap_mode:  '6-face' (standard), '8-tile' (NEW)
overlap:       0 to 50% (for cubemap seam blending)
```

**Stage 3 (Masking)**:
```python
categories:    ['person', 'personal_objects', 'animals'] (multi-select)
confidence:    0.0 to 1.0 (detection threshold, default 0.5)
model_size:    'nano', 'small', 'medium', 'large', 'xlarge'
use_gpu:       True/False (auto-detect CUDA)
```

### Camera Group Presets
- **Default**: "8-Camera Horizontal" (8 cameras, 0° pitch, 110° FOV)
- **Dome**: "16-Camera Dome" (8 main + 4 up + 4 down)
- **Cardinal**: "4-Cardinal" (N/S/E/W, 90° FOV)
- **Custom**: User-defined via UI or JSON import

### GPU/Device Handling
- **Priority**: CUDA > CPU (auto-detect)
- **Masking**: Batched inference on GPU (all detections per image in one pass)
- **Transforms**: CPU-based (GPU acceleration not needed for geometric ops)
- **Fallback**: Graceful CPU mode if CUDA unavailable

---

## Common Task Patterns

### Adding Stage 1 Extraction Method
1. Implement new extractor in `src/extraction/` (e.g., `opencv_extractor.py`)
2. Conform to interface: `extract_frames(input_path, output_dir, fps, **options) → frame_list`
3. Register in extraction method selector (UI dropdown)
4. Test with various .INSV file versions

### Adding New Transform/Cubemap Type
1. Create module in `src/transforms/` (e.g., `e2s_transform.py` for stereo)
2. Inherit caching pattern from `E2PTransform` (cache key = all params + dimensions)
3. For 8-tile cubemap: Extend `E2CTransform` with tile layout logic
4. Add UI option in Stage 2 configuration panel
5. Update batch pipeline to route based on selected transform type

### Extending Masking Categories
1. Edit `src/masking/category_config.py` to add COCO class IDs
2. Example categories:
   - **Personal objects**: backpack (24), handbag (26), suitcase (28), cell phone (67)
   - **Animals**: bird, cat, dog, horse, etc. (COCO classes 14-23)
3. Update UI checkboxes for category selection
4. Modify `generate_mask()` to filter detections by selected classes
5. Test GPU batch performance with multiple categories

### Adding Camera Preset
1. Create JSON preset in `src/config/camera_presets/`
2. Structure: `{name, rings: [{pitch, cameras: [{yaw, fov, roll}]}]}`
3. Register preset in UI dropdown
4. Ensure compass widget can render multi-ring layouts

### Customizing UI Components (Spec-Driven)
1. Refer to `specs/ui_specification.md` for design guidelines
2. Create PyQt6 widget in `src/ui/widgets/`
3. Follow minimalist styling (consistent colors, fonts, spacing)
4. Emit signals for pipeline coordination (e.g., `stage_completed`)
5. Update main window layout to integrate new component

---

## Testing Patterns

### Unit Tests
- **Location**: `tests/` folder (organized by stage)
  - `tests/extraction/`: SDK, FFmpeg, OpenCV extractors
  - `tests/transforms/`: E2P, E2C, cubemap variants
  - `tests/masking/`: Multi-category detection, GPU/CPU modes
  - `tests/pipeline/`: End-to-end batch workflows

### Key Test Scenarios
```bash
# Stage 1: Extraction
python -m pytest tests/extraction/test_sdk_extractor.py
# Verify: Frame count matches FPS × duration

# Stage 2: Transforms
python -m pytest tests/transforms/test_e2p_cache.py
# Verify: Cache hits on repeated params

# Stage 3: Masking
python -m pytest tests/masking/test_multi_category.py
# Verify: Persons + animals detected, masks generated

# Integration: Full pipeline
python -m pytest tests/pipeline/test_batch_workflow.py
# Verify: .INSV → perspectives → masks in one run
```

### Manual Testing Workflow
1. **Sample data**: Use test .INSV file in `tests/data/`
2. **Stage 1**: Extract 5 frames at 1 FPS
3. **Stage 2**: Split into 8 perspectives (110° FOV)
4. **Stage 3**: Mask persons + animals
5. **Validation**: 
   - Check output count: 5 × 8 = 40 images + 40 masks
   - Verify metadata preserved
   - Inspect mask quality with visualization overlay

---

## External Dependencies & Integration

### Critical External Tools

**Stage 1 Dependencies**:
- **Insta360 SDK**: Official SDK for dual-fisheye stitching
  - **Primary extraction method** (highest quality)
  - SDK location: Copy from `Original_Projects\Insta360toFrames\sdk\`
  - Platform: Windows x64 DLL bindings
  
- **FFmpeg**: Fallback extraction + Stage 2 v360 filter
  - Required for video handling if SDK unavailable
  - Check with `shutil.which('ffmpeg')`
  
- **OpenCV**: Frame-by-frame extraction alternative
  - `cv2.VideoCapture` for basic video reading

**Stage 2 Dependencies**:
- **NumPy**: Array operations for geometric transforms
- **OpenCV (cv2)**: Remap function for perspective projection

**Stage 3 Dependencies**:
- **Ultralytics (YOLOv8)**: Multi-category object detection
  - Models auto-download on first use (~7-136 MB)
  - GPU acceleration via PyTorch CUDA
- **PyTorch**: Backend for YOLOv8 inference
  - Install with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**UI Framework**:
- **PyQt6** (NOT PyQt5): Modern desktop interface
  - Migration note: Change imports from `PyQt5` → `PyQt6`
  - QThread, signals/slots for pipeline coordination

### Data Format Integration
- **Input**: `.insv` (Insta360 native), `.mp4`, equirectangular images
- **Intermediate**: Equirectangular PNG/JPG (temp folder)
- **Output**: 
  - Perspective images (PNG/JPG/TIFF)
  - Binary masks (`<name>_mask.png`, RealityScan format)
  - JSON metadata (camera params, NO GPS/GYRO)
- **Photogrammetry tools**: RealityScan, Metashape, COLMAP, RealityCapture

---

## Performance Considerations

### Optimization Points

**Stage 1 (Extraction)**:
1. **SDK vs FFmpeg**: Insta360 SDK is 3-5× faster than Insta360 Studio
2. **Parallel frame writing**: Async save to disk while SDK processes next frame
3. **Quality presets**: 'draft' mode for quick previews, 'best' for final output

**Stage 2 (Splitting)**:
1. **Transform caching**: E2PTransform caches remapping coordinates (cache key = all params + dimensions)
2. **Batch processing**: Process all compass cameras for one frame before moving to next
3. **Memory management**: Clear cache periodically if processing 1000+ frames

**Stage 3 (Masking)**:
1. **GPU batching**: Process multiple images in GPU memory (batch size = 4-8)
2. **Model selection**: 
   - **nano** (7 MB): 0.2s/image, 85% accuracy
   - **small** (23 MB): 0.5s/image, 90% accuracy ← **recommended**
   - **medium** (52 MB): 1.0s/image, 92% accuracy
   - **large** (83 MB): 1.5s/image, 94% accuracy
   - **xlarge** (136 MB): 2.5s/image, 95% accuracy
3. **Category filtering**: Filter detections by class AFTER inference (not during)

**Pipeline Orchestration**:
1. **Threading**: QThread for each stage, emit progress signals
2. **Temp storage**: Write intermediate equirectangular frames to SSD (not HDD)
3. **Stage skipping**: If user only needs masking, skip Stages 1-2

### Known Bottlenecks
- **SDK extraction**: CPU-bound, benefits from high single-thread performance
- **GPU memory**: 8 GB VRAM handles up to batch_size=8 with medium model
- **Disk I/O**: Writing 100 GB of images → use fast SSD for output folder
- **Metadata preservation**: Piexif slow for TIFF → prefer PNG/JPG for intermediate files

---

## Common Pitfalls & Gotchas

1. **SDK platform dependency**: Insta360 SDK is Windows x64 only. No macOS/Linux support. Document clearly.

2. **GPS/GYRO exclusion**: Original apps extracted GPS/GYRO. **Discard this data** (not useful for photogrammetry). Only preserve camera metadata.

3. **Cache invalidation**: E2P transform cache key = ALL parameters + image dimensions. Changing output size = forced recomputation.

4. **Mask convention**: RealityScan binary mask format:
   - 0 (Black) = remove/mask
   - 255 (White) = keep/valid
   - **NOT inverted** (enforce this in code)

5. **FFmpeg availability**: Check `shutil.which('ffmpeg')` before Stage 2. Fallback to pure Python transforms if missing.

6. **YOLOv8 first-run delay**: Model downloads ~100 MB on first detection. Show progress indicator in UI.

7. **PyQt5 → PyQt6 migration**: Source code uses PyQt5. Must update imports:
   - `from PyQt5.QtWidgets` → `from PyQt6.QtWidgets`
   - Signal/slot syntax may differ slightly

8. **Intermediate file cleanup**: Equirectangular frames can be 50-100 GB. Provide "Delete intermediate files" option.

9. **Compass multi-ring rendering**: When adding look-up/down rings, compass widget must render 3 concentric circles. Update `InteractiveCircularCompass.paintEvent()`.

10. **CUDA device selection**: If multiple GPUs, default to `cuda:0`. Allow user to select device in settings.

---

## Quick Reference for Common Changes

### Change Default Pipeline Settings
- **Extraction FPS**: `src/config/defaults.py` → `DEFAULT_FPS = 1.0`
- **Compass split count**: `src/config/defaults.py` → `DEFAULT_SPLIT_COUNT = 8`
- **Default FOV**: `src/config/defaults.py` → `DEFAULT_FOV = 110`

### Add New Extraction Method
1. Implement in `src/extraction/new_method_extractor.py`
2. Register in `src/extraction/__init__.py`: `EXTRACTORS['new_method'] = NewMethodExtractor`
3. Add UI dropdown option: `extraction_method_combo.addItem("New Method")`

### Adjust Masking Categories
- **UI**: Checkboxes in Stage 3 config panel
- **Code**: `src/masking/category_config.py` → add COCO class IDs to `CATEGORIES` dict
- **Example**: `'personal_objects': [24, 26, 28, 67]` (backpack, handbag, suitcase, phone)

### Add New Cubemap Type (e.g., 8-tile)
1. Edit: `src/transforms/e2c_transform.py` → add `generate_8tile_cubemap()`
2. UI: Add option in cubemap dropdown
3. Pipeline: Route to new method based on `settings['cubemap_type']`

### Enable/Disable Pipeline Stages
- **UI**: Toggle checkboxes for "Enable Stage X"
- **Code**: `pipeline_config['skip_extraction'] = True` → loads existing equirectangular
- **Thread**: Check flags before calling stage methods

### Customize Compass Appearance
- **File**: `src/ui/widgets/circular_compass.py`
- **Colors**: `self.state_colors` dict (Blue/Yellow/Red/Green)
- **Size**: `self.setFixedSize(400, 400)` for larger widget
- **Rings**: Add concentric circles for look-up/down in `paintEvent()`

---

## Key Files Reference

### New Unified App Structure

| File | Purpose | Source Reference |
|------|---------|------------------|
| `src/main.py` | PyQt6 main window + pipeline orchestration | New (integrates both apps) |
| `src/pipeline/batch_orchestrator.py` | 3-stage batch workflow coordinator | New |
| `src/extraction/sdk_extractor.py` | Insta360 SDK integration | `Insta360toFrames/` |
| `src/extraction/ffmpeg_extractor.py` | FFmpeg-based frame extraction | Both apps |
| `src/transforms/e2p_transform.py` | Equirectangular→Perspective (cached) | `360toFrame/core/` |
| `src/transforms/e2c_transform.py` | Equirectangular→Cubemap (6-face, 8-tile) | `360toFrame/core/` + new |
| `src/masking/multi_category_masker.py` | YOLOv8 multi-class detection (persons/objects/animals) | `360toFrame/core/` + extended |
| `src/ui/main_window.py` | PyQt6 main UI layout | New (minimalist spec) |
| `src/ui/widgets/circular_compass.py` | Interactive compass widget (multi-ring) | `360toFrame/main.py` + extended |
| `src/ui/widgets/stage_config_panels.py` | Config panels for each stage | New |
| `src/config/defaults.py` | Default parameters (FPS, FOV, split count, etc.) | New |
| `src/config/camera_presets.json` | Camera group presets | `360toFrame/config/` |
| `specs/ui_specification.md` | UI design spec (spec-driven development) | New |
| `tests/pipeline/test_full_workflow.py` | Integration test (INSV → masks) | New |

### Source Reference Folders (Read-Only)

| Folder | Contains | Use For |
|--------|----------|---------|
| `Original_Projects/Insta360toFrames/` | SDK integration, extraction methods | Copy Stage 1 code |
| `Original_Projects/360toFrame/core/` | Transform engines, metadata handling | Copy Stage 2 code |
| `Original_Projects/360toFrame/gui/` | PyQt5 widgets (compass, batch processor) | Reference for PyQt6 migration |

---

## Development Workflow

### Initial Setup
1. **Create project structure**: Follow spec-driven approach
   - Generate folder structure: `src/`, `specs/`, `resources/`, `tests/`
   - Create `specs/ui_specification.md` (minimalist PyQt6 design)
   - Download PyQt6 templates if needed

2. **Environment setup**:
   ```bash
   cd C:\Users\User\Documents\APLICATIVOS\360ToolKit
   python -m venv venv
   venv\Scripts\activate
   pip install PyQt6 numpy opencv-python ultralytics torch piexif
   ```

3. **Copy source code**:
   - Stage 1: Copy SDK integration from `Original_Projects/Insta360toFrames/`
   - Stage 2: Copy transforms from `Original_Projects/360toFrame/core/`
   - Migrate PyQt5 → PyQt6 (imports, signals/slots)

### Development Cycle
1. **Spec-first**: Update `specs/ui_specification.md` before UI changes
2. **Stage isolation**: Test each stage independently
3. **Unit tests**: Write tests in `tests/` before implementation
4. **Integration tests**: Test full pipeline with sample .INSV
5. **UI refinement**: Follow minimalist spec (clean, maintainable, beautiful)

### Testing Workflow
```bash
# Unit tests
pytest tests/extraction/
pytest tests/transforms/
pytest tests/masking/

# Integration test
pytest tests/pipeline/test_full_workflow.py

# Manual UI test
python src/main.py
```

### Code Reuse Strategy
- **DO**: Copy functions/classes from source folders
- **DO**: Adapt code to new architecture (3-stage pipeline)
- **DO NOT**: Modify files in `Original_Projects/` (read-only reference)
- **DO NOT**: Import libraries from `lib/` folders (ignore those)

---

## Spec-Driven Development

### UI Specification (Create First)
**File**: `specs/ui_specification.md`

**Content guidelines**:
- **Layout**: Main window with 3-stage tabs + batch control panel
- **Styling**: Minimalist theme (dark mode optional)
  - Colors: Primary (accent), secondary (background), text
  - Fonts: Sans-serif, consistent sizes
  - Spacing: 8px base unit, 16px section padding
- **Widgets**:
  - Stage 1: File selector, FPS input, quality dropdown, method selector
  - Stage 2: Compass widget (multi-ring), FOV sliders, split count
  - Stage 3: Category checkboxes, confidence slider, model size
  - Pipeline: Progress bars per stage, log output, start/stop buttons
- **Interactions**: Enable/disable stages, preview before batch, save presets
- **Responsive**: Resizable window, scrollable config panels

### Architecture Specification
**File**: `specs/architecture_specification.md`

**Content guidelines**:
- Module dependencies (stage isolation)
- Signal/slot communication patterns
- Threading strategy (QThread per stage)
- Error handling (graceful degradation)
- Configuration management (JSON presets)

### Templates to Download
- PyQt6 modern UI templates (search GitHub)
- Dark theme stylesheets
- Icon packs (minimalist, CC0 license)

---

## Important Notes

### DO NOT Create
- ❌ Summary documents (e.g., "project_status.md", "changelog.md")
- ❌ Progress reports
- ❌ Status tracking files

**Exception**: If documentation is absolutely necessary, create `docs/` folder for that purpose only.

### DO Create
- ✅ Spec documents in `specs/`
- ✅ Code files in `src/`
- ✅ Test files in `tests/`
- ✅ Configuration in `src/config/`

---

## When to Escalate

- **Insta360 SDK issues**: DLL loading, stitching artifacts—refer to SDK documentation
- **Geometric math**: Spherical coordinates, cubemap projections—see academic references
- **Photogrammetry workflows**: COLMAP/RealityScan integration—consult tool docs
- **FFmpeg edge cases**: Custom v360 filter params—research FFmpeg docs
- **GPU/CUDA**: PyTorch device selection, VRAM limits—environment-specific troubleshooting
- **PyQt6 migration**: Signal/slot changes from PyQt5—check Qt 6 migration guide

---

**Last Updated**: November 2025  
**Target Folder**: `C:\Users\User\Documents\APLICATIVOS\360ToolKit\`  
**Development Approach**: Spec-driven, minimalist UI, batch-first workflow
