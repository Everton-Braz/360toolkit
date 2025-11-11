# Stage 2 Cubemap Pipeline - Visual Flow

## Pipeline Routing

```
START: _execute_stage2()
│
├─ Get transform_type from config
│
├─ Load input frames (equirectangular images)
│
└─ ROUTE BY TYPE:
    │
    ├─ IF transform_type == 'cubemap':
    │   │
    │   └─→ _execute_stage2_cubemap()
    │       │
    │       ├─ Get cubemap params (type, face_size, fov, overlap)
    │       │
    │       ├─ FOR each input frame:
    │       │   │
    │       │   ├─ Load equirectangular image
    │       │   │
    │       │   └─ BRANCH BY CUBEMAP TYPE:
    │       │       │
    │       │       ├─ IF '6-face':
    │       │       │   │
    │       │       │   └─ Generate 6 faces (90° FOV fixed):
    │       │       │       ├─ Front  (yaw=0°, pitch=0°)
    │       │       │       ├─ Back   (yaw=180°, pitch=0°)
    │       │       │       ├─ Left   (yaw=-90°, pitch=0°)
    │       │       │       ├─ Right  (yaw=90°, pitch=0°)
    │       │       │       ├─ Top    (yaw=0°, pitch=90°)
    │       │       │       └─ Bottom (yaw=0°, pitch=-90°)
    │       │       │           ↓
    │       │       │       Save: frame_00001_front.png, etc.
    │       │       │
    │       │       └─ IF '8-tile':
    │       │           │
    │       │           └─ Generate 8 tiles (custom FOV):
    │       │               ├─ Row 0 (pitch=0°):
    │       │               │   ├─ tile_0_0 (yaw=0°)
    │       │               │   ├─ tile_0_1 (yaw=90°)
    │       │               │   ├─ tile_0_2 (yaw=180°)
    │       │               │   └─ tile_0_3 (yaw=270°)
    │       │               │
    │       │               └─ Row 1 (pitch=30°):
    │       │                   ├─ tile_1_0 (yaw=0°)
    │       │                   ├─ tile_1_1 (yaw=90°)
    │       │                   ├─ tile_1_2 (yaw=180°)
    │       │                   └─ tile_1_3 (yaw=270°)
    │       │                       ↓
    │       │                   Save: frame_00001_tile_0_0.png, etc.
    │       │
    │       └─ Return: {cubemap_count, output_files, cubemap_type}
    │
    └─ IF transform_type == 'perspective':
        │
        └─→ _execute_stage2_perspective()
            │
            ├─ Get camera positions (yaw/pitch/roll/fov)
            │
            └─ FOR each frame:
                └─ FOR each camera:
                    ├─ E2P transform (perspective projection)
                    ├─ Save: frame_00001_cam_00.png
                    └─ Embed camera metadata
                        ↓
                Return: {perspective_count, output_files}
```

## 6-Face Cubemap Layout

### Spherical View
```
        ┌─────────┐
        │   TOP   │ (pitch=+90°)
        └─────────┘
┌───────┬─────────┬─────────┬─────────┐
│ LEFT  │  FRONT  │  RIGHT  │  BACK   │ (pitch=0°)
│-90°   │   0°    │  +90°   │  180°   │
└───────┴─────────┴─────────┴─────────┘
        ┌─────────┐
        │ BOTTOM  │ (pitch=-90°)
        └─────────┘
```

### Output Files (1 frame)
```
frame_00001_front.png   → Looking forward (yaw=0°)
frame_00001_back.png    → Looking backward (yaw=180°)
frame_00001_left.png    → Looking left (yaw=-90°)
frame_00001_right.png   → Looking right (yaw=+90°)
frame_00001_top.png     → Looking up (pitch=+90°)
frame_00001_bottom.png  → Looking down (pitch=-90°)
```

## 8-Tile Grid Layout

### Spherical View (4×2 Grid)
```
Row 0 (Horizon, pitch=0°):
┌──────┬──────┬──────┬──────┐
│ 0_0  │ 0_1  │ 0_2  │ 0_3  │
│ 0°   │ 90°  │ 180° │ 270° │
└──────┴──────┴──────┴──────┘

Row 1 (Below horizon, pitch=30°):
┌──────┬──────┬──────┬──────┐
│ 1_0  │ 1_1  │ 1_2  │ 1_3  │
│ 0°   │ 90°  │ 180° │ 270° │
└──────┴──────┴──────┴──────┘
```

### FOV and Overlap (Example: FOV=100°)
```
Step size: 360° / 8 = 45°
Overlap: 100° - 45° = 55°
Overlap %: (55° / 45°) × 100 = 122%

Visual representation (top view):
    ╔════════╗
    ║ tile_0 ║
    ║  (0°)  ║
    ╚════╤═══╝
      ╔═══╧═══════╗
      ║  tile_1   ║
      ║   (90°)   ║
      ╚═════╤═════╝
         ╔═══╧════════╗
         ║  tile_2    ║
         ║   (180°)   ║
         ╚═════╤══════╝
            ╔══╧══════════╗
            ║  tile_3     ║
            ║   (270°)    ║
            ╚═════════════╝
            
Note: Shaded regions show overlap between adjacent tiles
```

## Transform Flow (E2P Projection)

```
Equirectangular Input
┌─────────────────────────────────────┐
│                                     │ ← Latitude/Longitude mapping
│         360° × 180°                 │
│                                     │
└─────────────────────────────────────┘
           │
           │ E2PTransform.equirect_to_pinhole()
           │ Parameters: yaw, pitch, roll, h_fov, width, height
           ↓
Perspective Output (Rectilinear)
┌──────────────┐
│              │ ← Pinhole projection
│   Face/Tile  │
│              │
└──────────────┘
```

### Coordinate Mapping
```
Equirectangular (spherical):
- X axis: Longitude (0° to 360°)
- Y axis: Latitude (-90° to +90°)

↓ Transform ↓

Perspective (rectilinear):
- X axis: Horizontal pixels
- Y axis: Vertical pixels
- Projection: tan(θ) mapping (pinhole camera)
```

## Metadata Embedded in Each Output

```json
{
  "camera_orientation": {
    "yaw": 0,          // Horizontal rotation (degrees)
    "pitch": 0,        // Vertical rotation (degrees)
    "roll": 0,         // Camera roll (always 0 for cubemap)
    "h_fov": 90        // Horizontal field of view (degrees)
  },
  "camera_metadata": {
    "model": "Insta360 X3",
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

This metadata is embedded in EXIF tags for use in photogrammetry software.

## Performance Comparison

```
Single Equirectangular Frame (3840×1920)
│
├─ Perspective Mode (8 cameras):
│   ├─ Operations: 8 transforms
│   ├─ Output: 8 images
│   └─ Time: ~2 seconds
│
├─ 6-Face Cubemap:
│   ├─ Operations: 6 transforms
│   ├─ Output: 6 images
│   └─ Time: ~1.5 seconds
│
└─ 8-Tile Grid:
    ├─ Operations: 8 transforms
    ├─ Output: 8 images
    └─ Time: ~2 seconds
```

## Error Handling

```
_execute_stage2()
│
├─ Check input_dir exists
│   └─ If None: Return error with message
│
├─ Check transform_type valid
│   └─ If invalid: Default to 'perspective'
│
├─ Check input frames exist
│   └─ If empty: Return error "No input frames found"
│
└─ Route to handler:
    │
    ├─ _execute_stage2_cubemap()
    │   │
    │   ├─ Try: Generate faces/tiles
    │   └─ Catch: Log error, return {success: False, error: str(e)}
    │
    └─ _execute_stage2_perspective()
        │
        ├─ Try: Generate perspectives
        └─ Catch: Log error, return {success: False, error: str(e)}
```

## User Workflow

```
1. USER: Select input video/images
   └─ App: Analyze and calculate default face_size

2. USER: Configure Stage 2
   ├─ Select "Cubemap" mode
   ├─ Choose type: "6-face" or "8-tile"
   └─ Adjust face_size (optional)

3. USER: Configure 8-tile (if selected)
   ├─ Choose control mode: "Set FOV" or "Set Overlap %"
   ├─ Adjust FOV slider (45-150°)
   │   └─ App: Auto-calculate overlap %
   └─ OR adjust overlap slider (0-75%)
       └─ App: Auto-calculate FOV

4. USER: Click "Start Pipeline"
   └─ App: Process frames
       ├─ Progress: "Stage 2 CUBEMAP: Frame 5/10 (6-face)"
       └─ Output: Save faces/tiles to output folder

5. USER: Review output
   └─ Open output/stage2_cubemap/ folder
       ├─ See generated faces/tiles
       └─ Verify metadata embedded
```

## Integration Points

```
Main Window (UI)
│
├─ Collects config:
│   ├─ transform_type: 'cubemap'
│   └─ cubemap_params: {type, face_size, fov, overlap_percent}
│       ↓
│
BatchOrchestrator
│
├─ Receives config via start_pipeline()
├─ Routes to _execute_stage2()
│   └─ Branches to _execute_stage2_cubemap()
│       │
│       ├─ Uses: E2PTransform (perspective projection)
│       ├─ Uses: MetadataHandler (EXIF embedding)
│       └─ Emits: Progress signals
│           ↓
│
Main Window (UI)
│
└─ Receives progress updates
    ├─ Updates progress bar
    └─ Shows status: "Stage 2 CUBEMAP: Frame X/Y"
```
