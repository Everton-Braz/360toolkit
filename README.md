# 360toolkit

**Unified photogrammetry preprocessing pipeline**: Extract frames from Insta360 cameras → Split to perspective views → Generate AI masks

**360toolkit** is a fully portable desktop application that combines frame extraction from Insta360 cameras with advanced perspective splitting and AI masking for professional photogrammetry workflows.



## 🎯 Three-Stage Pipeline

**360toolkit** combines two specialized applications into one streamlined workflow:

`EXTRACT FRAMES (Insta360 SDK)` → `SPLIT PERSPECTIVES (Equirectangular to Pinhole)` → `AI MASKING (YOLOv8)`

### **Stage 1: Frame Extraction** 🎬

- **Input**: `.INSV` (Insta360 native) or `.mp4` files
- **Output**: Equirectangular stitched panoramas (PNG/JPG/TIFF)
- **Methods**:
  - **Insta360 MediaSDK 3.0.5** (PRIMARY): GPU-accelerated AI stitching with seamless blending
  - **FFmpeg**: Fallback dual-stream extraction
  - **Dual-fisheye**: Raw lens images (no stitching)
  - **OpenCV**: Basic frame-by-frame extraction
- **Features**:
  - Configurable FPS (0.1 - 30 frames/second)
  - Time range selection (start/end in seconds)
  - Resolution options: Original, 8K, 6K, 4K, 2K
  - Metadata preservation (camera info, NO GPS/GYRO)
  - AI stitching quality modes: Draft, Good, Best

### **Stage 2: Perspective Splitting** 🔄

- **Input**: Equirectangular images
- **Output**: Rectilinear perspective views (PNG/JPEG/TIFF)
- **Compass-Based Camera Positioning**:
  - Default: 8 cameras arranged horizontally
  - Configurable FOV (30° - 150°)
  - Multi-ring support (main, look-up, look-down)
  - Custom yaw/pitch/roll per camera
- **Transform Engines**:
  - **E2P Transform**: Equirectangular → Pinhole perspective (cached for performance)
  - **E2C Transform**: Equirectangular → Cubemap (6-face + 8-tile variants)
  - **Real-time preview** with interactive compass widget
- **Output Customization**:
  - Custom image dimensions
  - Multiple format support
  - EXIF metadata embedding (camera orientation)

### **Stage 3: AI Masking** 🤖

- **Input**: Perspective images
- **Output**: Binary masks (`<image>_mask.png`, RealityScan compatible)
- **Detection Categories**:
  - Persons (primary)
  - Personal objects (bags, phones, backpacks, etc.)
  - Animals (all COCO animal classes)
- **Features**:
  - YOLOv8 instance segmentation (5 model sizes: nano → xlarge)
  - **ONNX Runtime Integration**: Lightweight, fast inference (CPU/GPU)
  - **Smart Skipping**: Skips mask generation for images without detected objects
  - **GPU acceleration** with CUDA support

## 🚀 Recent Updates (v1.0.0)

- **Fixed SDK Extraction**: Resolved issues with Insta360 MediaSDK integration for reliable frame extraction.
- **Optimized Masking**: Switched to ONNX Runtime for faster, lightweight masking.
- **Fixed Mask Generation**: Corrected matrix shape handling for YOLOv8 ONNX output, ensuring accurate detection and mask creation.
- **Smart Optimization**: Added pre-check to skip processing images with no target objects.

---

## Key Features

- **Official Insta360 SDK integration** (bypasses Insta360 Studio)
- **Dual-fisheye → equirectangular stitching**
- **Compass-based camera positioning** (8-camera default)
- **Multi-category detection**: Persons, personal objects, animals
- **RealityScan-compatible binary masks**

---

  - GPU acceleration (CUDA) with CPU fallback   ```powershell

  - Smart mask skipping (only create masks for detected objects)   cd C:\Users\User\Documents\APLICATIVOS\360ToolKit

  - Multi-category selection UI (checkboxes)   python -m venv venv

  - Batch processing with progress tracking   venv\Scripts\activate

- **Mask Format**: Binary (0 = mask/remove, 255 = keep/valid) - RealityScan compatible   ```



## 📦 Installation3. **Install dependencies**:

   ```powershell

### Option 1: Portable Executable (Recommended)

**No installation required!** Download `360toolkit-FULL-v1.0.zip` and extract. Works on any Windows 10/11 machine with NVIDIA GPU.



```powershell4. **For GPU support** (recommended):

# Extract and run   ```powershell

.\360ToolkitGS-FULL.exe   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

```   ```



**Requirements**:5. **Verify installation**:

- Windows 10/11 64-bit   ```powershell

- NVIDIA GPU (with driver installed)   python test_setup.py

- 20 GB free disk space   ```



### Option 2: Development Setup---

Clone repository and set up Python environment:

## Quick Start

```bash

git clone https://github.com/Everton-Braz/360toolkit.git### Run the Application

cd 360toolkit```powershell

python src/main.py

# Create virtual environment```

python -m venv venv

venv\Scripts\activate### Basic Workflow



# Install dependencies1. **Stage 1**: Load `.INSV` file → Set FPS (e.g., 1.0) → Extract equirectangular frames

pip install -r requirements.txt2. **Stage 2**: Configure compass (8 cameras, 110° FOV) → Split to perspectives

3. **Stage 3**: Enable categories (persons, objects, animals) → Generate masks

# Run application4. **Click "Start Pipeline"** → All stages run automatically

python src/main.py

```### Output Structure

```

**Requirements**:output/

- Python 3.10+├── frames/              # Stage 1: Equirectangular images

- PyTorch 2.7.1+cu118 (GPU version)├── perspectives/        # Stage 2: Camera views (8 per frame)

- Ultralytics YOLOv8└── masks/              # Stage 3: Binary masks (*_mask.png)

- PyQt6```

- OpenCV, NumPy

- Insta360 MediaSDK 3.0.5---



## 🚀 Quick Start

### Using the Portable Application

1. Launch `360ToolkitGS-FULL.exe`
2. **Stage 1**: Select `.INSV`/`.mp4` file, configure extraction settings, click "Extract"
3. **Stage 2**: Adjust compass settings, preview splits, click "Split"
4. **Stage 3**: Enable masking categories, click "Generate Masks"
5. Output saved to configured output folder

## Project Structure

```
360toolkit/
├── src/
│   ├── extraction/      # Stage 1: Frame extraction
│   ├── transforms/      # Stage 2: E2P, E2C engines
│   ├── masking/         # Stage 3: Multi-category detection

│   ├── ui/             # PyQt6 interface

### Using Python│   ├── pipeline/       # Batch orchestration

```python│   └── config/         # Settings & presets

from src.pipeline.batch_orchestrator import PipelineWorker├── specs/              # UI & architecture specs

├── tests/              # Unit & integration tests

# Configure pipeline├── Original_Projects/  # Source reference (read-only)

config = {└── requirements.txt

    'input_file': 'video.insv',```

    'output_dir': 'output/',

    'fps': 1.0,---

    'split_count': 8,

    'h_fov': 110,## Configuration

    'masking_categories': {

        'persons': True,### Camera Presets

        'personal_objects': True,- **8-Camera Horizontal**: Default (110° FOV, 0° pitch)

        'animals': True- **16-Camera Dome**: 8 main + 4 up + 4 down

    }- **4-Cardinal**: N/S/E/W views (90° FOV)

}- **Custom**: Define your own via JSON



# Run full pipelineLocated in: `src/config/camera_presets.json`

worker = PipelineWorker(config)

worker.start()### Masking Categories

```Edit in UI or `src/config/defaults.py`:

```python

## 📁 Project StructureMASKING_CATEGORIES = {

    'persons': True,

```    'personal_objects': True,  # backpack, phone, etc.

360ToolKit/    'animals': True

├── src/}

│   ├── extraction/           # Stage 1: Frame extraction```

│   │   ├── sdk_extractor.py

│   │   ├── ffmpeg_extractor.py---

│   │   └── frame_extractor.py

│   ├── transforms/           # Stage 2: Perspective splitting## Advanced Usage

│   │   ├── e2p_transform.py  (cached equirect→pinhole)

│   │   ├── e2c_transform.py  (equirect→cubemap)### Stage-Only Processing

│   │   └── metadata_handler.py

│   ├── masking/              # Stage 3: AI maskingRun individual stages:

│   │   ├── multi_category_masker.py```python

│   │   └── category_config.py# Stage 1 only: Extract frames

│   ├── ui/                   # PyQt6 interfacepython src/extraction/extract_frames.py --input video.insv --output frames/ --fps 1.0

│   │   ├── main_window.py

│   │   └── widgets/# Stage 2 only: Split perspectives

│   ├── pipeline/             # Batch orchestrationpython src/transforms/split_perspectives.py --input frames/ --output perspectives/

│   │   └── batch_orchestrator.py

│   ├── config/               # Configuration# Stage 3 only: Generate masks

│   │   ├── defaults.pypython src/masking/batch_mask.py --input perspectives/ --output masks/

│   │   └── camera_presets.json```

│   └── main.py              # Application entry point
├── specs/                    # Specification documents
├── tests/                    # Unit and integration tests
├── runtime_hook_pytorch.py   # PyInstaller runtime hook (DLL pre-loading)
├── runtime_hook_sdk.py       # SDK environment setup
├── 360toolkit_FULL.spec      # PyInstaller spec (build config)
├── requirements.txt
├── README.md
└── .gitignore
```

### Batch Processing via CLI

```python
from src.pipeline import BatchOrchestrator

orchestrator = BatchOrchestrator()
orchestrator.run_full_pipeline(
    input_file='path/to/video.insv',
    output_dir='path/to/output',
    fps=1.0,

    camera_preset='8-Camera Horizontal',

## ⚙️ Configuration    masking_enabled=True

)

### Camera Presets```

Predefined camera group configurations in `src/config/camera_presets.json`:

- **8-Camera Horizontal**: 8 cameras in horizontal ring, 110° FOV (default)---

- **4-Cardinal**: N/S/E/W positioning, 90° FOV

- **16-Camera Dome**: 8 main + 4 up + 4 down## Performance Tips

- **Custom**: User-defined via UI or JSON

### Stage 1 (Extraction)

### Extraction Settings- **SDK mode**: 3-5× faster than Insta360 Studio

```python- Use `draft` quality for quick previews

# src/config/defaults.py- Use `best` quality for final output

DEFAULT_FPS = 1.0              # Frames per second (0.1-30)

DEFAULT_H_FOV = 110            # Horizontal field of view (30-150°)### Stage 2 (Splitting)

DEFAULT_SPLIT_COUNT = 8        # Cameras per ring- Transform cache reuses mappings (10× speedup)

DEFAULT_OUTPUT_WIDTH = 1920    # Output image width- Process all cameras for one frame before moving to next

DEFAULT_OUTPUT_HEIGHT = 1920   # Output image height- SSD recommended for temp storage

DEFAULT_MODEL_SIZE = 'small'   # YOLOv8 model: nano|small|medium|large|xlarge

DEFAULT_CONFIDENCE_THRESHOLD = 0.5### Stage 3 (Masking)

DEFAULT_USE_GPU = True         # GPU acceleration- **GPU**: Small model = 0.5s/image, Medium = 1.0s/image

```- **CPU**: Small model = 3s/image (fallback)

- Batch size 4-8 for 8 GB VRAM

## 🔧 Build from Source (PyInstaller)

---

### Prerequisites

- Python 3.10.11
- PyTorch 2.7.1+cu118 with CUDA 11.8
- PyInstaller 6.16.0
- Insta360 MediaSDK 3.0.5

### Build Command

```powershell
pyinstaller 360toolkit_FULL.spec --clean
```

## Troubleshooting

### GPU Not Detected

```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA

```pip uninstall torch torchvision

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

**Output**: `dist\360ToolkitGS-FULL\` (~9 GB)```



**Key Build Features**:### Insta360 SDK Issues

- ONE-DIR mode (required for PyTorch DLLs)- **Windows x64 only** (no macOS/Linux support)

- MSVC Runtime DLL bundling (WinError 1114 fix)- Copy SDK DLLs from `Original_Projects/Extraction_Reference/sdk/`

- CUDA DLL bundling (18 DLLs, ~2.7 GB)- Use FFmpeg fallback if SDK unavailable

- Lazy import hooks (torch/ultralytics loaded only at runtime)

- Runtime DLL pre-loading via `runtime_hook_pytorch.py`### Memory Issues

- Reduce batch size in Stage 3 settings

## 📊 Performance- Lower cache size in Settings → Performance

- Process fewer frames per run

### Stage 1: Frame Extraction

- **SDK Method**: ~3-5 seconds per frame (GPU-accelerated, best quality)---

- **FFmpeg Method**: ~1-2 seconds per frame (proven quality)

- **OpenCV Method**: ~0.5-1 second per frame (basic, no stitching)## Testing



### Stage 2: Perspective Splitting### Run Tests

- **E2P Transform**: ~0.1-0.2 seconds per frame (with caching)```powershell

- **E2C Transform**: ~0.15-0.3 seconds per frame# Unit tests

pytest tests/transforms/

### Stage 3: AI Maskingpytest tests/masking/

- **GPU (NVIDIA RTX 3070+)**: ~0.3-0.5 seconds per image

- **GPU (NVIDIA RTX 4070+)**: ~0.2-0.3 seconds per image# Integration test

- **CPU (i7/Ryzen 7)**: ~2-5 seconds per imagepytest tests/pipeline/test_full_workflow.py



## 🔌 Integration with Photogrammetry Tools# Manual test

python test_setup.py

Output is compatible with:```

- **RealityScan** (primary - uses mask format)

- **Metashape**---

- **RealityCapture**

- **COLMAP**## Known Limitations

- **CloudCompare**

1. **Insta360 SDK**: Windows x64 only

## 📝 Specifications2. **GPS/GYRO**: Not extracted (not needed for photogrammetry)

3. **Large videos**: Recommend splitting videos > 10 minutes

### Stage 1 Parameters4. **VRAM**: 8 GB minimum for medium model with batch size 4

```

fps_interval:     0.1 - 30.0 (frames per second)---

output_format:    equirectangular | dual_fisheye

resolution:       original | 8k (7680×3840) | 6k | 4k | 2k## Contributing

quality:          draft | good | best

output_format:    jpg | png1. Read `specs/ui_specification.md` for design guidelines

```2. Follow spec-driven development approach

3. Test with `pytest` before submitting

### Stage 2 Parameters4. Preserve metadata chain (NO GPS/GYRO)

```

yaw:              -180 to +180° (camera horizontal rotation)---

pitch:            -90 to +90° (camera vertical angle)

roll:             -180 to +180° (camera rotation)## License

h_fov:            30 to 150° (horizontal field of view)

v_fov:            auto-calculated from h_fov and aspect ratio**Source Code**: See original project licenses

split_count:      1 to 12 (cameras per ring)- Extraction Module: Original license applies

cubemap_mode:     6-face | 8-tile- 360toFrame: Original license applies

overlap:          0 to 50% (for seamless blending)

```**This Integration**: MIT License



### Stage 3 Parameters---

```

categories:       [person | personal_objects | animals] (multi-select)## Credits

confidence:       0.0 to 1.0 (detection threshold)

model_size:       nano | small | medium | large | xlarge**360FrameTools** unifies:

use_gpu:          true | false (auto-detect CUDA)- **Frame Extraction**: Frame extraction from dual-fisheye cameras

skip_existing:    true | false (skip if mask exists)- **360toFrame**: Perspective splitting with compass positioning

```

**Technologies**:

## 🐛 Troubleshooting- PyQt6 (GUI)

- Ultralytics YOLOv8 (AI masking)

### GPU Not Detected- OpenCV (image processing)

```- NumPy (transformations)

Error: CUDA not available

Solution: Install NVIDIA GPU driver (no CUDA toolkit needed with portable build)---

```

## Version History

### Memory Issues

```**v1.0.0** (Initial Release)

Error: Out of memory during masking- Three-stage unified pipeline

Solution: Reduce batch size in config or use smaller YOLOv8 model (nano/small)- Multi-category AI masking

```- GPU acceleration

- Interactive compass UI

### SDK Not Found

```
Error: Insta360 MediaSDK not detected

Solution: Included in portable build; if building from source, ensure SDK at:
C:\Users\[User]\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1\
```

---

## ⚖️ Legal & Licensing

**360ToolKit** is open-source software licensed under the [MIT License](LICENSE).

However, this software depends on the **Insta360 Camera SDK**, which is proprietary software owned by Arashi Vision Inc. (Insta360).

- **Source Code**: The source code of 360ToolKit is free and open.
- **Binaries**: The compiled releases of 360ToolKit include the Insta360 SDK binaries, which are redistributed under the terms of the [Insta360 SDK End User License Agreement](https://www.insta360.com/support/supportcourse?post_id=20734).
- **Trademarks**: "Insta360" is a trademark of Arashi Vision Inc. This project is an unofficial tool and is not endorsed by or affiliated with Insta360.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Insta360 SDK Notice
This software uses the **Insta360 MediaSDK** for frame extraction. The SDK itself is **proprietary** and is NOT included in this source code repository.

- **For Users**: The pre-built releases include the necessary SDK runtime files (allowed under SDK redistribution terms).
- **For Developers**: If you wish to build this project from source, you must:
  1. Download the MediaSDK from the [Insta360 Developer Portal](https://www.insta360.com/sdk/home).
  2. Place the SDK files in a local directory.
  3. Update the build configuration to point to your local SDK copy.

**Note**: You are responsible for complying with the Insta360 SDK License Agreement when using or redistributing their software.

## 🙏 Acknowledgments

- **Insta360 MediaSDK 3.0.5**: Official stitching engine
- **Ultralytics YOLOv8**: Instance segmentation
- **PyTorch**: Deep learning framework
- **PyQt6**: Desktop UI framework

## 📞 Support

For issues, feature requests, or questions:
- Open an Issue on GitHub
- Check existing documentation in `specs/` folder
- Review test cases in `tests/` folder

---

**Version**: 1.0.0  
**Last Updated**: November 10, 2025  
**Status**: Production Ready ✅
