# 360FrameTools - UI Specification

## Design Philosophy
**Minimalist, professional, spec-driven interface** for a multi-stage photogrammetry pipeline (Extract → Transform → Mask → Align → Train).

**Core principles**:
- Clean, uncluttered layout with clear visual hierarchy
- Easy to maintain - modular components with consistent patterns
- Beautiful yet functional - form follows function
- Self-documenting - tooltips and inline help where needed

---

## Color Palette

### Dark Theme (Primary)
```
Primary Background:     #2b2b2b (Dark gray)
Secondary Background:   #3c3c3c (Lighter gray)
Accent/Active:          #4a9eff (Blue)
Success:                #32cd32 (Green)
Warning:                #ffd700 (Gold)
Error:                  #dc143c (Crimson)
Text Primary:           #e0e0e0 (Light gray)
Text Secondary:         #a0a0a0 (Medium gray)
Border:                 #555555 (Gray)
Disabled:               #666666 (Dark gray)
```

### Light Theme (Optional)
```
Primary Background:     #f5f5f5 (Light gray)
Secondary Background:   #ffffff (White)
Accent/Active:          #0078d7 (Blue)
Text Primary:           #333333 (Dark gray)
Text Secondary:         #666666 (Medium gray)
Border:                 #cccccc (Light gray)
```

---

## Typography

### Fonts
- **Primary**: Segoe UI (Windows), San Francisco (macOS), Roboto (Linux)
- **Monospace**: Consolas, Monaco, 'Courier New'

### Sizes
```
H1 (Section Headers):   16pt, Bold
H2 (Group Titles):      14pt, Semi-bold
Body (Labels):          10pt, Normal
Small (Hints):          9pt, Normal
Button Text:            10pt, Normal
```

---

## Spacing & Layout

### Base Unit System
```
Base unit:              8px
Small spacing:          8px (1 unit)
Medium spacing:         16px (2 units)
Large spacing:          24px (3 units)
Section padding:        16px
Widget margins:         8px
```

### Window Layout
```
Minimum size:           1280 x 800px
Preferred size:         1400 x 900px
Maximum size:           Unlimited (resizable)
```

---

## Main Window Structure

### Layout Hierarchy
```
MainWindow
├── MenuBar (File, Tools, Help)
├── CentralWidget (QWidget)
│   ├── PipelineControlPanel (Top, fixed height)
│   │   ├── Start/Stop/Pause buttons
│   │   ├── Overall progress bar
│   │   └── Status indicator
│   ├── StageTabWidget (Center, expandable)
│   │   ├── Stage1Tab (Extract Frames)
│   │   ├── Stage2Tab (Split Perspectives)
│   │   └── Stage3Tab (Generate Masks)
│   └── LogOutputPanel (Bottom, collapsible)
│       └── QTextEdit (read-only, monospace)
└── StatusBar (File count, output path, warnings)
```

---

## Pipeline Control Panel

### Components
```
[===== Overall Progress Bar =====] 75%
[▶ Start Pipeline] [⏸ Pause] [⏹ Stop] [⚙ Settings]

Status: Processing Stage 2 (Camera 5/8)
Elapsed: 00:02:34 | Estimated remaining: 00:01:12
```

### Layout
- **Height**: 80px fixed
- **Background**: Secondary background color
- **Border**: 1px solid border color (bottom only)
- **Padding**: 16px

### Button Specifications
```
Start Button:
  - Color: Success green
  - Size: 120px × 36px
  - Icon: Play arrow
  - State: Disabled when processing

Pause Button:
  - Color: Warning gold
  - Size: 100px × 36px
  - Icon: Pause
  - State: Enabled only during processing

Stop Button:
  - Color: Error crimson
  - Size: 100px × 36px
  - Icon: Stop
  - State: Enabled only during processing
```

---

## Stage 1: Frame Extraction Tab

### Layout Sections

#### Input Configuration
```
┌─ Input Settings ─────────────────────────────────┐
│ Input File:  [________________] [Browse...]       │
│                                                    │
│ File Metadata (auto-detected):                    │
│   Type:        .INSV (Insta360 dual-fisheye)     │
│   Duration:    5m 34s (334 seconds)               │
│   Resolution:  5760×2880 dual-fisheye             │
│   FPS:         24.00                               │
│   Camera:      Insta360 ONE X2                    │
│   Total Frames: 8016 frames                       │
└──────────────────────────────────────────────────┘
```

#### Time Range Selection
```
┌─ Time Range ─────────────────────────────────────┐
│ ☑ Extract full video (0s - 334s)                 │
│                                                    │
│ ○ Extract specific range:                         │
│   Start Time: [0] seconds                         │
│   End Time:   [334] seconds                       │
│                                                    │
│ Frame Interval: [1.0] FPS (0.1 - 30.0)           │
│ Expected Frames: ~334 frames                      │
└──────────────────────────────────────────────────┘
```

#### Extraction Method & Quality
```
┌─ Extraction Method ──────────────────────────────┐
│ ○ SDK Stitching (Recommended)                     │
│   SDK Path: C:\Users\...\CameraSDK-2.0.2\        │
│   Quality:     [Good ▼]  (Draft, Good, Best)     │
│   Resolution:  [Original ▼] (Original, 8K,       │
│                              6K, 4K, 2K)          │
│   Output:      [PNG ▼] (PNG, JPEG)               │
│                                                    │
│ ○ Dual-Fisheye Export (No stitching)             │
│   Output:      [PNG ▼] (PNG, JPEG)               │
│                                                    │
│ ○ FFmpeg Fallback (Equirectangular if pre-stitched) │
│                                                    │
│ ○ OpenCV Frame-by-Frame (Basic extraction)       │
└──────────────────────────────────────────────────┘
```

#### Output Configuration
```
┌─ Output Settings ────────────────────────────────┐
│ Output Folder:  [________________] [Browse...]    │
│ □ Delete intermediate files after pipeline       │
└──────────────────────────────────────────────────┘
```

#### Preview
```
┌─ Preview ────────────────────────────────────────┐
│                                                    │
│      [Equirectangular preview thumbnail]          │
│                  640×320px                         │
│                                                    │
│            [🔄 Refresh Preview]                   │
└──────────────────────────────────────────────────┘
```

---

## Stage 2: Split Perspectives Tab

### Layout Sections

#### Input Configuration
```
┌─ Input Settings ─────────────────────────────────┐
│ ☑ Use frames from Stage 1 (auto)                 │
│ Input Folder: [________________] [Browse...]      │
│                                                    │
│ Transform Type: [Perspective (E2P) ▼]            │
│   Options: Perspective, Cubemap (6-face),        │
│            Cubemap (8-tile)                       │
└──────────────────────────────────────────────────┘
```

#### Camera Configuration
```
┌─ Camera Groups ──────────────────────────────────┐
│ Preset: [8-Camera Horizontal ▼]                  │
│   Options: 8-Camera, 16-Dome, 4-Cardinal, Custom│
│                                                    │
│ Split Count: [8] cameras                          │
│ Horizontal FOV: [110]° (30-150°)                 │
│ Vertical FOV: [Auto] (calculated)                │
│                                                    │
│ [+ Add Look-Up Ring]  [+ Add Look-Down Ring]     │
│                                                    │
│ Ring Configuration:                                │
│   • Main Ring: 8 cameras, Pitch 0°               │
└──────────────────────────────────────────────────┘
```

#### Real-Time Preview Panel (Perspective Mode)

**Layout**: Three-column horizontal layout
```
┌─ Preview (Perspective E2P) ──────────────────────────────────────────────────┐
│                                                                                │
│  [Look Down]     [Main Camera Preview Window]        [Look Up]               │
│  [+ Add Ring]           640×360px                     [+ Add Ring]            │
│                                                                                │
│                  ┌─ View Controls ─────┐                                      │
│                  │ Yaw/Pan:    [0]°    │                                      │
│                  │ Pitch/Tilt: [0]°    │                                      │
│                  │ Splits:     [8]     │                                      │
│    ┌────────┐   │ FOV:        [110]°  │   ┌────────┐                         │
│    │        │   │ First Frame:[0]     │   │        │                         │
│    │ Comp.  │   │ Last Frame: [48]    │   │ Comp.  │                         │
│    │ Look   │   └─────────────────────┘   │ Look   │                         │
│    │ Down   │                              │ Up     │                         │
│    │ Ring   │   [Circular Compass]         │ Ring   │                         │
│    │(Opt.)  │      400×400px               │(Opt.)  │                         │
│    │        │                              │        │                         │
│    └────────┘   🔵 Export  🟡 Preview      └────────┘                         │
│                 🔴 Disabled 🟢 Mask                                            │
│                                                                                │
│ Click pizza slices to select camera | Click icons to cycle states            │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **Main preview**: Shows current camera view with real-time updates
- **Circular compass**: Pizza-slice style, clickable slices select camera
  - Each slice colored by state: Blue=Export, Yellow=Preview, Red=Disabled, Green=Mask
  - Click camera icons (white circles) to cycle states
- **Optional rings**: Look-down (left) and look-up (right) buttons create additional compass rings
- **View controls**: Adjust yaw/pan, pitch, splits, FOV, frame range
- **Frame navigation**: First/last frame sliders update preview

#### Real-Time Preview Panel (Cubemap Mode)

**Layout**: Single panel with grid visualization
```
┌─ Preview (Cubemap) ──────────────────────────────────────────────────────────┐
│                                                                                │
│                   [Equirectangular Source Image]                              │
│                          1200×600px                                            │
│                                                                                │
│                   [Grid Overlay with Tile Borders]                            │
│                                                                                │
│    ┌─────────┬─────────┬─────────┬─────────┐                                 │
│    │  Left   │  Front  │  Right  │  Back   │  ← Tile layout visualization    │
│    │ (Red)   │ (Green) │ (Blue)  │(Yellow) │                                 │
│    ├─────────┼─────────┼─────────┼─────────┤                                 │
│    │   Top   │         │  Bottom │         │                                 │
│    │(Cyan)   │         │(Magenta)│         │                                 │
│    └─────────┴─────────┴─────────┴─────────┘                                 │
│                                                                                │
│ ┌─ Cubemap Settings ──────────────┐                                           │
│ │ Mode:         [6-Face ▼]        │                                           │
│ │               (6-Face, 8-Tile)  │                                           │
│ │ Tile FOV:     [90]° (H+V)       │                                           │
│ │ Overlap:      [5]% (0-50%)      │                                           │
│ │ First Frame:  [0]               │                                           │
│ │ Last Frame:   [48]              │                                           │
│ └─────────────────────────────────┘                                           │
│                                                                                │
│ Grid colors show tile positions | Overlap shown as semi-transparent borders  │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **Equirectangular display**: Shows source image with grid overlay
- **Tile visualization**: Colored borders show cubemap face positions
  - 6-Face mode: Left, Front, Right, Back, Top, Bottom
  - 8-Tile mode: Adds 4 diagonal corner tiles
- **Overlap visualization**: Semi-transparent borders show overlap percentage
- **Settings panel**: Configure mode, FOV, overlap, frame range
- **Real-time update**: Grid adjusts as settings change

#### Output Configuration
```
┌─ Output Settings ────────────────────────────────┐
│ Output Folder:  [________________] [Browse...]    │
│ Image Format:   [PNG ▼] (PNG, JPEG, TIFF)        │
│ Output Size:    [1920] × [1080] pixels           │
│ Naming Pattern: [frame_###_cam_##] .png          │
└──────────────────────────────────────────────────┘
```

---

## Stage 3: Generate Masks Tab

### Layout Sections

#### Input Configuration
```
┌─ Input Settings ─────────────────────────────────┐
│ ☑ Use images from Stage 2 (auto)                 │
│ Input Folder: [________________] [Browse...]      │
└──────────────────────────────────────────────────┘
```

#### Detection Categories
```
┌─ Masking Categories ─────────────────────────────┐
│ Select objects to mask (remove from images):      │
│                                                    │
│ ☑ Persons (COCO class 0)                          │
│ ☑ Personal Objects                                │
│   ☑ Backpack      ☑ Handbag                       │
│   ☑ Suitcase      ☑ Cell Phone                    │
│                                                    │
│ ☑ Animals                                         │
│   ☑ All Animals (bird, cat, dog, etc.)            │
│   ☐ Selective... [Configure]                      │
└──────────────────────────────────────────────────┘
```

#### Model Configuration
```
┌─ Detection Settings ─────────────────────────────┐
│ Model Size:    [Small ▼]                          │
│   nano (7MB, 0.2s) | small (23MB, 0.5s) |        │
│   medium (52MB, 1s) | large (83MB, 1.5s) |       │
│   xlarge (136MB, 2.5s)                            │
│                                                    │
│ Confidence:    [▬▬▬▬▬▬●▬▬▬] 0.50 (0.0-1.0)       │
│                                                    │
│ Device:        [Auto-detect ▼] (CUDA / CPU)      │
│ Batch Size:    [4] images (GPU only)              │
└──────────────────────────────────────────────────┘
```

#### Output Configuration & Smart Masking
```
┌─ Output Settings ────────────────────────────────┐
│ Output Folder:  [________________] [Browse...]    │
│ Mask Format:    [RealityScan (Binary) ▼]         │
│ Naming:         <image_name>_mask.png            │
│                                                    │
│ ☑ Smart mask skipping (Recommended)               │
│   Only create masks for images with detected     │
│   objects. Skip images without persons/animals.  │
│   Saves disk space and processing time.          │
│                                                    │
│ □ Save detection visualizations                   │
│ □ Generate detection report (JSON)                │
└──────────────────────────────────────────────────┘
```

**Smart Mask Skipping Logic**:
- If image has NO detections matching selected categories → skip mask creation
- Log which images were processed vs skipped
- Example: 392 images with detections = 392 masks created
           392 images without detections = 0 masks (skipped)
- Result: Only relevant masks saved (saves 50% disk space in typical scenarios)

#### Preview
```
┌─ Preview ────────────────────────────────────────┐
│ [Original Image]  |  [Generated Mask]             │
│   400×300px       |    400×300px                  │
│                                                    │
│        [🔄 Test on Sample Image]                  │
└──────────────────────────────────────────────────┘
```

---

## Interactive Circular Compass Widget

### Visual Design

#### Multi-Ring Layout (when enabled)
```
        Top View (Concentric Rings)
        
           Look-Up Ring
        (pitch +30°, 4 cams)
              ↑
       ┌──────●──────┐
       │             │
    ● ─┤   Main Ring ├─ ●
       │  (8 cameras) │
       └──────●──────┘
              ↓
         Look-Down Ring
        (pitch -30°, 4 cams)
```

#### Camera Icon States
```
🔵 Blue (Export):      Solid blue circle + arrow
🟡 Yellow (Preview):   Solid yellow circle + eye icon
🔴 Red (Disabled):     Solid red circle + X
🟢 Green (Mask):       Solid green circle + mask icon
```

#### Interaction
- **Click once**: Cycle state (Export → Preview → Disabled → Mask → Export)
- **Hover**: Show tooltip with camera details (yaw, pitch, FOV)
- **Right-click**: Open context menu (Edit Camera, Remove)

#### Compass Features
- **N/S/E/W markers**: Cardinal directions labeled
- **Angle indicators**: Degree markings every 45°
- **Center point**: Current yaw indicator
- **Ring labels**: "Main", "Look-Up", "Look-Down" (when visible)

---

## Log Output Panel

### Design
```
┌─ Processing Log ─────────────────────────────[▼]┐
│ [12:34:56] Stage 1: Extracting frames...         │
│ [12:35:02] ✓ Extracted 334 frames (1.0 FPS)      │
│ [12:35:03] Stage 2: Splitting perspectives...    │
│ [12:35:45] ✓ Camera 1/8 complete (334 images)    │
│ ...                                                │
└───────────────────────────────────────────────────┘
```

### Features
- **Collapsible**: Click header to expand/collapse
- **Auto-scroll**: Automatically scroll to latest log entry
- **Color coding**: 
  - Info: Light gray text
  - Success: Green text with ✓
  - Warning: Yellow text with ⚠
  - Error: Red text with ✗
- **Monospace font**: For aligned output
- **Height**: 150px default, 300px expanded

---

## Menu Bar

### File Menu
```
File
├── New Project...           Ctrl+N
├── Open Project...          Ctrl+O
├── Save Configuration...    Ctrl+S
├── ─────────────────────
├── Import Camera Preset...
├── Export Camera Preset...
├── ─────────────────────
└── Exit                     Alt+F4
```

### Tools Menu
```
Tools
├── Stage 1 Only
├── Stage 2 Only
├── Stage 3 Only
├── ─────────────────────
├── Verify Dependencies
├── Clear Cache
└── Settings...             Ctrl+,
```

### Help Menu
```
Help
├── Documentation
├── Quick Start Guide
├── ─────────────────────
├── Check for Updates
└── About 360FrameTools
```

---

## Status Bar

### Layout
```
[Ready] | 0 files processed | Output: C:\...\output | ⚠ 2 warnings
```

### Components
- **Status indicator**: Ready, Processing, Complete, Error
- **File counter**: Processed/Total files
- **Output path**: Truncated with tooltip showing full path
- **Warning/Error count**: Clickable to show details

---

## Settings Dialog

### Categories (Tabbed)
```
┌─ Settings ───────────────────────────────────────┐
│ [General] [Performance] [Advanced] [About]        │
│                                                    │
│ General:                                           │
│   Theme: [Dark ▼] (Dark, Light, System)          │
│   Language: [English ▼]                           │
│   □ Auto-detect file types                        │
│   □ Show tooltips                                 │
│   □ Confirm before overwriting files              │
│                                                    │
│ Performance:                                       │
│   Cache size: [2048] MB                           │
│   GPU device: [CUDA:0 ▼]                          │
│   □ Use GPU acceleration (requires CUDA)          │
│   Thread count: [Auto ▼] (1, 2, 4, 8, Auto)      │
│                                                    │
│ Advanced:                                          │
│   Temp folder: [C:\Temp\360FrameTools]            │
│   □ Keep intermediate files for debugging         │
│   Log level: [Info ▼] (Debug, Info, Warning)     │
│                                                    │
│             [Apply] [Cancel] [OK]                 │
└──────────────────────────────────────────────────┘
```

---

## Widget Specifications

### Custom Toggle Switch
```python
# Modern iOS-style toggle
Size: 60×30px
States: ON (green, circle right), OFF (gray, circle left)
Animation: Smooth slide transition (200ms)
```

### File Path Selector
```
[___________________________] [Browse...]
Width: Input=300px, Button=80px
Browse button opens QFileDialog
Show full path in tooltip if truncated
```

### Slider with Value Display
```
Label: Confidence Threshold
[▬▬▬▬▬▬●▬▬▬] 0.50
Min: 0.0, Max: 1.0, Step: 0.05
Real-time value display updates on drag
```

### Progress Bar
```
Height: 24px
Style: Flat with percentage text overlay
Color: Accent blue (fill), Secondary background (empty)
Text: "Processing: 45%" (centered, white)
```

### Groupbox
```
Border: 1px solid border color
Border-radius: 4px
Title: H2 font, positioned top-left
Padding: 16px inside
Margin: 8px outside
```

---

## Responsive Behavior

### Window Resize
- **Stage tabs**: Expand/contract horizontally
- **Compass widget**: Fixed size (400×400px), centered
- **Log panel**: Expand vertically, scroll when needed
- **Minimum size**: Enforce 1280×800px

### High DPI Support
- **Scaling**: Auto-scale all UI elements based on system DPI
- **Icons**: Use vector icons (SVG) or provide 2× assets
- **Text**: Use pt-based sizing (not px)

---

## Keyboard Shortcuts

### Global
```
Ctrl+N:       New Project
Ctrl+O:       Open Project
Ctrl+S:       Save Configuration
Ctrl+,:       Settings
F1:           Help
F5:           Refresh Preview
Ctrl+Q:       Quit
```

### Pipeline Control
```
Ctrl+Enter:   Start Pipeline
Ctrl+P:       Pause Pipeline
Ctrl+Shift+P: Stop Pipeline
```

---

## Tooltips & Help Text

### Tooltip Style
```
Background: #3c3c3c (slightly lighter than main)
Border: 1px solid #555555
Text: White, 9pt
Padding: 8px
Max-width: 300px
Delay: 500ms
```

### Tooltip Examples
```
"SDK Stitching: Uses official Insta360 SDK for highest quality. 
Requires Windows x64. Bypasses Insta360 Studio."

"Confidence Threshold: Minimum detection confidence (0.0-1.0). 
Lower values detect more objects but may include false positives."

"Mask Format: RealityScan binary masks use 0 (black) for masked 
regions and 255 (white) for valid areas."
```

---

## Error Handling & User Feedback

### Error Dialog
```
┌─ Error ──────────────────────────────────────────┐
│  ⚠                                                │
│  Failed to load input file                        │
│                                                    │
│  File not found: C:\path\to\missing.insv          │
│                                                    │
│  Details: [Show Technical Details ▼]              │
│                                                    │
│                          [OK]                      │
└──────────────────────────────────────────────────┘
```

### Warning Toast
```
┌────────────────────────────────────────┐
│ ⚠ GPU not detected - using CPU mode    │
│   Processing will be slower             │
│                               [Dismiss]  │
└────────────────────────────────────────┘
Position: Bottom-right corner
Duration: 5 seconds (auto-dismiss)
```

### Success Notification
```
┌────────────────────────────────────────┐
│ ✓ Pipeline complete!                    │
│   2,672 images generated                 │
│   Output: C:\...\output                  │
│                        [Open] [Dismiss]  │
└────────────────────────────────────────┘
```

---

## Animation & Transitions

### Smooth Interactions
```
Button hover:        100ms fade
Toggle switch:       200ms slide
Progress bar fill:   300ms ease-out
Panel expand:        250ms ease-in-out
Tab switch:          150ms fade
Compass rotation:    400ms ease-in-out
```

### Loading States
```
Spinner: 
  - Circular spinner (accent color)
  - 32×32px
  - Infinite rotation (1s per cycle)
  
Skeleton loading:
  - Pulse animation for preview thumbnails
  - Gray gradient shimmer effect
```

---

## Accessibility

### Features
- **Keyboard navigation**: Tab order follows logical flow
- **Screen reader support**: ARIA labels on all interactive elements
- **High contrast mode**: Respect system high contrast settings
- **Scalable text**: Support system font scaling (125%, 150%)
- **Focus indicators**: Visible focus ring (2px accent color)

---

## Implementation Notes

### PyQt6 Components
```python
# Main window
QMainWindow with QMenuBar, QStatusBar

# Layouts
QVBoxLayout (vertical stacks)
QHBoxLayout (horizontal rows)
QGridLayout (form-style inputs)

# Custom widgets
InteractiveCircularCompass (QWidget with custom paintEvent)
ToggleSwitch (QCheckBox with custom paintEvent)

# Standard widgets
QPushButton, QLabel, QSlider, QSpinBox, QComboBox
QCheckBox, QLineEdit, QTextEdit, QProgressBar
QGroupBox, QTabWidget, QScrollArea

# Dialogs
QFileDialog, QMessageBox, QInputDialog
QDialog (for Settings)
```

### Stylesheets (QSS)
```css
/* Example: Apply dark theme */
QMainWindow {
    background-color: #2b2b2b;
    color: #e0e0e0;
}

QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 8px 16px;
    color: #e0e0e0;
}

QPushButton:hover {
    background-color: #4a9eff;
}

QPushButton:pressed {
    background-color: #3a7fcf;
}

QPushButton#startButton {
    background-color: #32cd32;
    border: none;
}
```

---

## File Organization

### UI Module Structure
```
src/ui/
├── __init__.py
├── main_window.py              # MainWindow class
├── widgets/
│   ├── __init__.py
│   ├── circular_compass.py     # InteractiveCircularCompass
│   ├── toggle_switch.py        # ToggleSwitch
│   ├── file_path_selector.py   # File/folder picker widget
│   ├── value_slider.py         # Slider with value display
│   └── progress_panel.py       # Pipeline control panel
├── tabs/
│   ├── __init__.py
│   ├── stage1_tab.py           # Frame extraction tab
│   ├── stage2_tab.py           # Perspective splitting tab
│   └── stage3_tab.py           # Masking tab
├── dialogs/
│   ├── __init__.py
│   ├── settings_dialog.py      # Settings window
│   ├── preset_manager.py       # Camera preset import/export
│   └── about_dialog.py         # About window
└── styles/
    ├── __init__.py
    ├── dark_theme.qss          # Dark theme stylesheet
    └── light_theme.qss         # Light theme stylesheet
```

---

## UI Refactor Blueprint (v1.1 - PyQt6 Widgets)

### Decision
- **Keep PyQt6 Widgets** (no framework migration required).
- Refactor should focus on **design tokens, component consistency, and layout hierarchy**.
- QML migration is optional future work, not required for a modern/professional result.

### Refactor Goals
- Reduce visual noise and improve readability on all stages.
- Make one clear primary action per context.
- Standardize spacing, typography, and component states.
- Remove duplicated controls and inconsistent interaction patterns.
- Preserve existing pipeline logic while modernizing presentation.

### Non-Goals (for v1.1)
- No rewrite of extraction/transforms/masking business logic.
- No major navigation paradigm switch during first pass.
- No new premium feature scope creep.

### Information Architecture (Target)
```
MainWindow
├── AppHeader (fixed)
│   ├── Input selector
│   ├── Output selector
│   ├── Primary action: Start Pipeline
│   ├── Secondary actions: Pause, Stop
│   └── Global progress + short status
├── NavigationRail (left)
│   ├── Overview
│   ├── Stage 1: Extract
│   ├── Stage 2: Perspective
│   ├── Stage 2: Cubemap
│   ├── Stage 3: Masking
│   ├── Stage 4: Alignment
│   └── Stage 5: Training
├── ContentArea (stacked pages)
│   └── Stage page with form + preview + contextual actions
├── LogPanel (bottom, collapsible)
└── StatusBar (device, warnings, counts)
```

### Exact Component Map

#### Shared Components (must be reused across all stages)
- `AppHeaderBar`
  - Inputs: `input_path`, `output_path`, `pipeline_state`, `overall_progress`
  - Actions: `browse_input`, `browse_output`, `start`, `pause`, `stop`
- `StagePageScaffold`
  - Regions: `summary_strip`, `settings_column`, `preview_column`, `actions_footer`
- `CardSection`
  - Title + optional subtitle + content layout
- `FormRow`
  - Label, control, hint, validation message
- `PrimaryButton`, `SecondaryButton`, `DangerButton`
- `InlineStatusBadge` (Ready / Running / Paused / Error / Complete)
- `CollapsibleLogPanel`

#### Stage-Specific Components
- Stage 1:
  - `FileMetadataCard`
  - `TimeRangeCard`
  - `ExtractionMethodCard`
  - `Stage1PreviewCard`
- Stage 2 Perspective:
  - `OutputConfigCard`
  - `CameraGroupsCard`
  - `PerspectivePreviewCard`
  - `CompassLegendBar`
- Stage 2 Cubemap:
  - `CubemapConfigCard`
  - `CubemapGridPreviewCard`
- Stage 3:
  - `MaskTargetCard`
  - `CategoryMatrixCard`
  - `DetectionSettingsCard`
  - `MaskPreviewCard`
- Stage 4:
  - `ReconstructionMethodCard`
  - `QualityPresetCard`
  - `PerformanceCard`
  - `DependencyStatusCard`
- Stage 5:
  - `TrainingTargetCard`
  - `LichtfeldPathCard`
  - `TrainingLaunchCard`

### Screen-by-Screen Restructuring

#### Overview Page
- Keep as orchestration dashboard only.
- Replace multiple equal-weight stage panels with:
  1. Pipeline summary strip (enabled stages, estimated outputs)
  2. Stage checklist with status badges
  3. Single primary CTA: `Start Pipeline`
- Move per-stage `Run Stage X` buttons to each stage page footer.

#### Stage 1 (Extract)
- Two-column layout:
  - Left: metadata, time range, extraction settings, output settings
  - Right: preview + expected frame count + validation warnings
- Add inline computed summary:
  - `effective_fps`, `expected_frames`, `estimated_disk_usage`
- Keep one stage action row at bottom: `Run Stage 1`, `Validate Settings`.

#### Stage 2 Perspective
- Top row: output config + preset selector.
- Middle row: camera groups editor (left) + live preview/compass (right).
- Bottom row: per-camera state summary chips (`Export`, `Preview`, `Disabled`, `Mask`).
- Remove duplicated controls between preview and form where possible.

#### Stage 2 Cubemap
- Keep dedicated page (not mixed into Perspective panel).
- Left: cubemap type, tile size, overlap.
- Right: equirectangular preview with grid overlay.
- Add compact explanatory hint block under type selector.

#### Stage 3 Masking
- Keep category matrix, but convert to grouped checklists with counters.
- Add a compact "Masking profile summary" block:
  - selected classes count
  - confidence threshold
  - device/batch mode
- Add `Test on Sample` near preview, not mixed with global controls.

#### Stage 4 Alignment
- Keep mode selector but reduce text density:
  - concise mode title
  - one-line description
  - expandable technical details
- Show dependency diagnostics in a dedicated warning card.
- Keep quality/performance options below method selection.

#### Stage 5 Training
- Minimal form: enable toggle, path selector, launch readiness summary.
- If Stage 4 output missing, show actionable prerequisite card.
- Keep this stage visually lighter than processing stages.

### Interaction Rules (UX Contract)
- Exactly one primary action button per page footer.
- Disable impossible actions with explicit reason text.
- Validate immediately on field blur for critical fields (paths, ranges).
- Persist last-used settings per stage via JSON profile.
- Progress model:
  - global progress in header
  - stage progress in page summary strip
  - item-level progress in log only

### Style Token File (Concrete)

#### New file
`src/ui/styles/tokens.py`

#### Required structure
```python
TOKENS = {
    "spacing": {
        "xs": 4,
        "sm": 8,
        "md": 12,
        "lg": 16,
        "xl": 24,
        "xxl": 32,
    },
    "radius": {
        "sm": 4,
        "md": 8,
        "lg": 12,
    },
    "font": {
        "family": "Segoe UI",
        "size_xs": 9,
        "size_sm": 10,
        "size_md": 11,
        "size_lg": 14,
        "size_xl": 18,
        "weight_regular": 400,
        "weight_semibold": 600,
        "weight_bold": 700,
    },
    "color_dark": {
        "bg_app": "#0F172A",
        "bg_surface": "#131C31",
        "bg_card": "#1A2642",
        "bg_card_hover": "#223156",
        "border": "#2A3B63",
        "text_primary": "#E5E7EB",
        "text_secondary": "#9CA3AF",
        "accent": "#22D3EE",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#F43F5E",
        "focus": "#38BDF8",
    },
    "elevation": {
        "card_border": 1,
    },
    "control": {
        "height_sm": 30,
        "height_md": 36,
        "height_lg": 42,
    },
}
```

#### Styling pipeline
- Keep QSS as rendering layer.
- Generate QSS from token dictionary (single source of truth).
- Do not hardcode colors inside widget classes.

### QSS Architecture (Concrete)
```
src/ui/styles/
├── tokens.py
├── dark_theme.qss
├── light_theme.qss
├── components/
│   ├── buttons.qss
│   ├── inputs.qss
│   ├── cards.qss
│   ├── nav.qss
│   └── log_panel.qss
└── build_qss.py
```

### Naming & ObjectName Rules
- All reusable controls must declare object names:
  - `btnPrimary`, `btnSecondary`, `btnDanger`
  - `cardSection`, `formRow`, `statusBadge`
  - `navItemActive`, `navItemIdle`
- No anonymous style-critical widgets.

### Usability Acceptance Criteria
- User can configure and launch pipeline without scrolling more than one viewport per stage (except advanced sections).
- All required fields expose validation errors inline within 300ms after user commit.
- Contrast for primary text meets WCAG AA equivalent for desktop dark theme.
- Keyboard-only navigation completes full pipeline setup.
- First-time user can identify where to start within 5 seconds.

### Implementation Sequence (Low Risk)
1. Build tokens + split QSS into component files.
2. Introduce shared scaffolding widgets (`CardSection`, `FormRow`, `StagePageScaffold`).
3. Refactor `Overview` and `Stage 1` first (highest onboarding impact).
4. Refactor `Stage 2` pages (Perspective and Cubemap).
5. Refactor `Stage 3`, then `Stage 4/5`.
6. Final pass: accessibility, keyboard flow, visual consistency.

### Notes for Current App State
- Current app already has strong functional scope; v1.1 should prioritize clarity over adding features.
- Stage numbering should remain explicit in navigation.
- Preserve GPU status indicator and processing log visibility.

---

## Version History

**Version 1.0** (Initial spec)
- Three-stage pipeline interface
- Interactive compass with multi-ring support
- Dark theme with minimalist design
- Modular component architecture

**Version 1.1** (UI Refactor Blueprint)
- Added concrete PyQt6 refactor blueprint for current multi-stage app
- Added exact reusable component map and page scaffolds
- Added screen-by-screen restructuring for Overview + Stages 1-5
- Added tokenized styling architecture and QSS composition plan
- Added UX contract and acceptance criteria

---

**Last Updated**: February 12, 2026  
**Author**: 360FrameTools Development Team  
**Status**: Active Development
