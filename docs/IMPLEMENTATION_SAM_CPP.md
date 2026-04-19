# SAM3.cpp — Implementation Notes
## Image Segmentation Toolkit for Windows (C++14)

---

## 1. Overview

This project is a Windows-native C++ integration of **SAM3** (Segment Anything Model 3),
running fully **on-device** via `ggml` — no Python, no PyTorch, no server required.

Two executables are built:

| Executable | Purpose |
|---|---|
| `sam3_image.exe` | Interactive GUI — click/drag to segment, export masks |
| `segment_persons.exe` | Headless batch tool — segment via bounding boxes, save PNGs |

Both support the same `.ggml` model files and the same output formats.

---

## 2. Repository & Credits

| Resource | URL |
|---|---|
| Original repo | https://github.com/PABannier/sam3.cpp |
| SAM3 paper | https://arxiv.org/abs/2408.00714 |
| ggml framework | https://github.com/ggerganov/ggml |
| Dear ImGui | https://github.com/ocornut/imgui |
| SDL2 | https://www.libsdl.org/ |

---

## 3. Models

Two quantized GGUF/GGML model files are used. Place them in `sam3cpp_repo/models/`.

### sam3-visual-q4_0.ggml (~289 MB)
- Visual encoder only (image features).
- Supports **PVS** (Point/Visual Segmentation) and **Box** prompts.
- Does **not** support text prompts or PCS (category search).
- Faster to load; good for interactive point/box use.
- Use with `--model sam3-visual-q4_0.ggml`.

### sam3-q4_0.ggml (~707 MB)
- Full model: visual encoder + **text encoder**.
- Supports PVS, Box, and **PCS** (Prompt-guided Cosine Similarity) category search.
- Required for the category checkbox feature ("Person", "Bag", etc.).
- Use with `--model sam3-q4_0.ggml`.

> **Download**: Both files must be downloaded separately. The original repo README
> has HuggingFace links. As of 2025, the files are at:
> https://huggingface.co/PABannier/sam3.cpp-ggml

---

## 4. Build Setup (Windows MSVC)

### Prerequisites
- Visual Studio 2019 Build Tools (MSVC 19.x), x64
- CMake 3.18+
- SDL2 development libraries (Windows x64)
- Git

### Build Steps

```powershell
# Clone
git clone https://github.com/PABannier/sam3.cpp sam3cpp_repo
cd sam3cpp_repo

# Configure
cmake -B build -G "Visual Studio 16 2019" -A x64 `
  -DSDL2_DIR="C:/path/to/SDL2/cmake"

# Build
cmake --build build --config Release --target sam3_image
cmake --build build --config Release --target segment_persons
```

Runtime DLLs expected in `build/bin/Release/` (SDL2.dll, ggml*.dll). Add to PATH before running:

```powershell
$env:PATH = "C:\...\sam3cpp_repo\build\bin\Release;$env:PATH"
```

### Windows-specific fixes applied to the source
- `#define SDL_MAIN_HANDLED` before `<SDL.h>` (prevents WinMain conflict)
- `#define NOMINMAX` and `#define WIN32_LEAN_AND_MEAN` before `<Windows.h>`
- `extern "C" int stbi_write_png(...)` — forward-declaration only; `stb_image_write`
  is already compiled into `sam3.lib`, so including the full header causes link errors

---

## 5. Architecture

### 5.1 API Layer (`sam3.h`)

Three segmentation modes, each with its own param struct and function:

```cpp
// Point/visual segmentation — click points or drag a box
sam3_result sam3_segment_pvs(sam3_state&, sam3_model&, sam3_pvs_params);

// Text/category-guided segmentation
sam3_result sam3_segment_pcs(sam3_state&, sam3_model&, sam3_pcs_params);

// Utilities
sam3_model_ptr sam3_load_model(sam3_params);
sam3_state_ptr sam3_create_state(sam3_model&, sam3_params);
sam3_image     sam3_load_image(const char* path);
bool           sam3_encode_image(sam3_state&, sam3_model&, sam3_image);
bool           sam3_save_mask(sam3_mask, const char* path);
```

**Workflow**:
1. `sam3_load_model` → `sam3_create_state` (do once)
2. `sam3_load_image` → `sam3_encode_image` (once per image, ~1–5 s on CPU)
3. `sam3_segment_pvs` or `sam3_segment_pcs` (fast, ~0.1–0.5 s per call)
4. Consume `sam3_result.detections[i].mask.data` (uint8, 0/255, W×H)

### 5.2 Interactive GUI (`main_image.cpp`)

Stack: SDL2 window → OpenGL 3.0 context → Dear ImGui render loop.

**`app_state` struct** — all runtime state:

```cpp
struct app_state {
    sam3_model           model;
    sam3_state_ptr       state;
    sam3_image           image;       // raw RGB pixels
    sam3_result          result;      // current inference result
    std::vector<sam3_detection> accumulated; // multi-object stack

    // Segmentation mode
    interaction_mode     mode;        // POINTS / BOX_PVS / EXEMPLAR_PCS

    // PCS category booleans (all true by default)
    bool cat_persons, cat_bags, cat_phones, cat_hats, cat_sky;
    char cat_custom[128];             // extra comma-separated terms

    // Post-processing
    int  morph_radius;                // >0 erode, <0 dilate
    bool guided_snap;                 // guided filter edge snap (export only)
    bool flatten_export;              // union all masks → mask_flat.png
    bool alpha_export;                // export RGBA image_masked.png
    int  feather_radius;              // 0–40 soft edge feathering
    bool visual_only;                 // true = no text encoder available
};
```

**Frame loop**:
1. SDL events → ImGui input
2. If `pending != ACTION_NONE`: run inference (deferred so busy overlay renders first)
3. Draw top bar (category checkboxes / segment button)
4. Draw canvas (image + overlay texture)
5. Draw bottom panel (thresholds, morphology, export options)

### 5.3 Headless Tool (`segment_persons.cpp`)

No GUI. Used for batch/scripted workflows:

```powershell
segment_persons.exe `
  --model sam3-visual-q4_0.ggml `
  --image panorama.jpg `
  --box 100,50,800,900 `
  --feather 10
```

Outputs:
- `<stem>_persons_overlay.png` — colour blended overlay
- `<stem>_persons_masked.png` — RGBA with inverted alpha (subject transparent)
- `person_mask_00.png`, `person_mask_01.png` … — raw binary masks

---

## 6. Post-Processing Pipeline

All helpers are plain C++ (no external deps beyond STL):

### `morph_mask(mask, W, H, r)`
Separable binary erosion (`r > 0`) or dilation (`r < 0`). Applied in the overlay
preview **and** at export. Uses a 1D sliding-window count to achieve O(W×H) complexity
regardless of radius.

### `guided_filter_mask(mask, rgb, W, H)`
O(N) box-filter guided filter. Snaps mask edges to colour boundaries in the source image.
Export-only (too slow for real-time preview). Parameters: `r=8`, `eps=0.02`.

### `feather_mask(mask, W, H, radius)`
Three passes of box blur approximate a Gaussian fade on the binary mask edge.
Applied to the **alpha channel** of the RGBA export, producing a soft cut-out boundary.

### `flatten_masks(detections, W, H, ...)`
OR-union of all detection masks into a single W×H binary image.
Runs `morph_mask` and `guided_filter_mask` per detection before combining.

### Alpha export logic
```
alpha[i] = feather_mask(flat)[i]         // soft 0–255 value at mask edge
rgba[..+3] = 255 - alpha[i]              // INVERTED: subject=transparent, BG=opaque
```
This "cut-out background" convention is useful for compositing tools that treat
alpha as background opacity.

---

## 7. Segmentation Modes

### PVS — Point/Visual Segmentation
- Left-click: positive point (include this area)
- Right-click: negative point (exclude this area)
- Drag: bounding box prompt
- Works with both model sizes
- Best for precise single-object selection

### PCS — Prompt-guided Cosine Similarity (text search)
- Requires full model (`sam3-q4_0.ggml`)
- Each category runs a separate `sam3_segment_pcs` call; results are merged
- `run_pcs_all()` iterates `get_active_prompts()` and accumulates all detections
- NMS and score thresholds filter overlapping/low-confidence boxes

### Multi-object accumulation
- `[+ Add]` pushes `result.detections` into `accumulated`
- `[Clear]` removes current (unsaved) result only
- `[Clear All]` empties both `accumulated` and `result`
- Export always combines `accumulated + result.detections`

---

## 8. Challenges & Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| `WinMain` link error | SDL2 redefines `main` on Windows | `#define SDL_MAIN_HANDLED` before SDL include |
| `min`/`max` macro conflicts | `<Windows.h>` defines them | `#define NOMINMAX` before Windows include |
| `stbi_write_png` duplicate symbol | stb compiled into sam3.lib | Forward-declare only; do not include stb header |
| PCS crashes with visual-only model | Text encoder not loaded | `visual_only` flag set at load time; PCS disabled in UI |
| Slow encode on CPU | SAM3 visual encoder is heavy | Encode once per image; cache `image_encoded` flag |
| Multi-replace corrupting file | Ambiguous old-string match | Always use 5+ lines of surrounding context |
| Category prompts finding nothing | Low score threshold | Default `score_threshold = 0.5`; user can lower in slider |
| `image_masked.png` needed inverted alpha | Convention: background opaque for compositing | `rgba[3] = 255 - alpha[i]` |

---

## 9. Output Files

| File | Format | Description |
|---|---|---|
| `mask_flat.png` | 8-bit grayscale | Union of all masks, binary 0/255 |
| `mask_00.png` … | 8-bit grayscale | Per-detection binary mask (flatten=off mode) |
| `image_masked.png` | RGBA PNG | Original image + inverted alpha channel |
| `<stem>_persons_overlay.png` | RGB PNG | Colour-blended mask overlay (headless tool) |
| `<stem>_persons_masked.png` | RGBA PNG | RGBA with inverted alpha (headless tool) |

---

## 10. Using SAM3 in 360Toolkit

Your app **360toolkit** can call SAM3 as either:

### Option A — Shell out to `segment_persons.exe`

The simplest path. 360toolkit spawns the headless tool, waits for it to finish,
reads the output PNG files.

```cpp
// Pseudocode (Win32 CreateProcess or std::system)
std::string cmd =
    "segment_persons.exe"
    " --model sam3-visual-q4_0.ggml"
    " --image " + equirect_path +
    " --box "  + box_str +       // "x0,y0,x1,y1"
    " --feather 8"
    " --no-gpu";
system(cmd.c_str());

// Read back
cv::Mat masked = cv::imread("output_persons_masked.png", cv::IMREAD_UNCHANGED);
// masked has 4 channels; alpha channel = 255 where background is
```

**Box coordinates** for 360° equirectangular images: the full sphere is mapped to
a 2:1 rectangle. If persons appear near the horizon (equator of the sphere), their
bounding box will be in the center rows:

```
// Example for 7680×3840 equirect:
// Horizon row ≈ H/2 = 1920
// Person height ≈ 30–40% of image height
box = "x0,1000,x1,2800"
```

### Option B — Link `sam3.lib` directly

Embed SAM3 into 360toolkit as a static library. Add `sam3.h` to your include path
and `sam3.lib` to your linker inputs.

```cpp
#include "sam3.h"

// One-time setup
sam3_params p; p.model_path = "sam3-visual-q4_0.ggml"; p.use_gpu = false;
auto model = sam3_load_model(p);
auto state = sam3_create_state(*model, p);

// Per image
sam3_image img = sam3_load_image("equirect.jpg");
sam3_encode_image(*state, *model, img);   // ~2–5 s on CPU

// Per object
sam3_pvs_params pvs;
pvs.use_box = true;
pvs.box = { x0, y0, x1, y1 };
auto result = sam3_segment_pvs(*state, *model, pvs);

// result.detections[0].mask.data — uint8_t vector, W×H, 0/255
```

**Key notes for 360° images**:
- SAM3 expects standard perspective or equirectangular JPEG/PNG — it does not
  understand the sphere mapping. Segmentation near poles is less reliable.
- For person removal use cases: run `segment_persons.exe` to get the RGBA masked
  PNG, then composite the inpainted/clean background over the transparent subject
  region in your 360° compositor.
- Very large equirect images (8K+) take considerable RAM (several GB) during encoding.
  Consider downsampling to 4K before segmentation if speed is critical.

### Option C — PCS text-based search (full model only)

Use `sam3-q4_0.ggml` and the `sam3_segment_pcs` function to find all instances of
a category by name, without needing to draw bounding boxes:

```cpp
sam3_pcs_params pcs;
pcs.text_prompt     = "person";
pcs.score_threshold = 0.5f;
pcs.nms_threshold   = 0.1f;
auto r = sam3_segment_pcs(*state, *model, pcs);
// r.detections — one entry per found person
```

Run this for each relevant category ("person", "car", "sign", etc.) and accumulate.

---

## 11. Running the GUI

```powershell
# Visual-only model (points/box, no text)
$env:PATH = "sam3cpp_repo\build\bin\Release;$env:PATH"
.\sam3cpp_repo\build\examples\Release\sam3_image.exe `
  --model sam3cpp_repo\models\sam3-visual-q4_0.ggml `
  --image images\panorama.jpg

# Full model (enables category checkboxes)
.\sam3cpp_repo\build\examples\Release\sam3_image.exe `
  --model sam3cpp_repo\models\sam3-q4_0.ggml `
  --image images\panorama.jpg
```

**Typical workflow**:
1. Wait for image to encode (~2–10 s depending on size)
2. Check desired category boxes (Person ✓, Sky ✓, …), click **Segment**
3. Review detections; click **+ Add** to accumulate
4. Switch to Points mode, click to refine specific objects
5. Enable **Flatten to one mask**, optionally **Export alpha image** + set Feather
6. Click **Export masks** → `mask_flat.png` and `image_masked.png` saved to CWD

---

## 12. File Map

```
sam3cpp_repo/
├── sam3.h                        SAM3 public API
├── examples/
│   ├── main_image.cpp            Interactive GUI (SDL2 + ImGui)
│   └── segment_persons.cpp       Headless batch tool
├── models/
│   ├── sam3-visual-q4_0.ggml     ~289 MB — visual only
│   └── sam3-q4_0.ggml            ~707 MB — full (text+visual)
└── build/
    ├── bin/Release/              DLLs (ggml*, SDL2)
    └── examples/Release/
        ├── sam3_image.exe
        └── segment_persons.exe
```
