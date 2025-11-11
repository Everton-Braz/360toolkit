# 360ToolkitGS - CPU Version

**Complete photogrammetry preprocessing toolkit for Insta360 cameras**

---

## 📦 What's Included

✅ **Insta360 SDK** - Bundled  
✅ **FFmpeg** - Bundled  
✅ **PyTorch CPU** - Bundled  
✅ **YOLOv8 Models** - Auto-download on first use  

**This version works out-of-box! No Python or PyTorch installation needed.**

---

## 🚀 First-Time Setup

1. **Extract the ZIP file** to any location (e.g., `C:\360ToolkitGS-CPU\`)
2. **Double-click `360ToolkitGS-CPU.exe`** to launch
3. **That's it!** No installation required

The application is fully self-contained and ready to use.

---

## 💻 System Requirements

### Minimum
- **OS**: Windows 10 (64-bit) or Windows 11
- **RAM**: 8 GB
- **CPU**: Intel Core i5 or equivalent (4 cores)
- **Storage**: 2 GB free space
- **GPU**: Not required (CPU masking is slower but works)

### Recommended
- **RAM**: 16 GB or more
- **CPU**: Intel Core i7 or AMD Ryzen 7 (6+ cores)
- **Storage**: 5 GB free space (for large projects)

**Note**: This version does NOT use GPU acceleration for masking. If you have an NVIDIA GPU and want faster masking (~6-7× faster), use the **GPU version** instead.

---

## 🎯 Features

### Stage 1: Frame Extraction
- Extract frames from Insta360 `.INSV` and `.mp4` files
- Official Insta360 SDK integration (highest quality)
- Configurable FPS (0.1 - 30 frames per second)
- Resolution options: Original, 8K, 6K, 4K, 2K
- Output: Equirectangular panoramas

### Stage 2: Perspective Splitting
- Convert 360° images to multiple perspective views
- Compass-based camera positioning (default: 8 cameras, 110° FOV)
- Customizable split count, FOV, yaw/pitch/roll
- Interactive preview with circular compass widget
- Real-time camera positioning visualization

### Stage 3: AI Masking
- YOLOv8 instance segmentation (CPU-powered)
- Multi-category detection:
  - Persons (human figures)
  - Personal objects (bags, phones, etc.)
  - Animals (all COCO classes)
- 5 model sizes: nano, small, medium, large, xlarge
- RealityScan-compatible binary masks (`<name>_mask.png`)

---

## 📊 Complete Workflow

1. **Select Input**: Choose `.INSV` or `.mp4` file
2. **Configure Extraction**: Set FPS and quality
3. **Configure Splitting**: Adjust compass cameras and FOV
4. **Enable Masking**: Select categories to mask (optional)
5. **Start Batch Processing**: One-click automated pipeline
6. **Output**: Perspective images + masks ready for photogrammetry

---

## ⚡ Performance

### Masking Speed (CPU)

| Model Size | Time per Image | Accuracy | Recommended For |
|------------|----------------|----------|----------------|
| Nano       | ~0.3s          | 85%      | Quick previews |
| Small      | ~0.5s          | 90%      | **Most users** ✅ |
| Medium     | ~1.2s          | 92%      | Better accuracy |
| Large      | ~2.0s          | 94%      | High accuracy |
| XLarge     | ~3.5s          | 95%      | Maximum accuracy |

**Example**: 100 images with small model = ~50 seconds

**Note**: GPU version is 6-7× faster for masking:
- Small model GPU: ~0.08s per image
- Small model CPU: ~0.5s per image

---

## ❓ Troubleshooting

### Application Won't Start
1. Extract all files from ZIP (don't run from inside ZIP)
2. Check Windows SmartScreen: Click "More info" → "Run anyway"
3. Disable antivirus temporarily (false positive detection)
4. Right-click `360ToolkitGS-CPU.exe` → Properties → Unblock

### SDK Not Found Error
- The SDK is bundled in `_internal/sdk/`
- If error persists, re-extract the ZIP file
- Ensure folder structure is intact

### FFmpeg Not Found
- FFmpeg is bundled in `_internal/ffmpeg/`
- Check antivirus didn't quarantine `ffmpeg.exe`

### Masking is Slow
- **This is expected with CPU version!**
- CPU masking is 6-7× slower than GPU
- Use smaller model (nano/small) for faster processing
- For faster masking, use the **GPU version** instead

### Out of Memory
- Close other applications
- Use smaller YOLOv8 model (nano/small)
- Process fewer images per batch
- Reduce output resolution

---

## 🎯 Use Cases

- **Photogrammetry**: RealityScan, Metashape, COLMAP, RealityCapture
- **Gaussian Splatting**: 3DGS, Nerfstudio, Luma AI
- **360° Video Processing**: Frame extraction and conversion
- **Dataset Generation**: Multi-view images for computer vision

---

## 📝 Technical Details

### Bundled Components
- **Insta360 MediaSDK 3.0.5**: Official SDK for dual-fisheye stitching
- **FFmpeg**: Video processing and frame extraction
- **PyTorch CPU 2.x**: Machine learning framework
- **YOLOv8**: Instance segmentation models
- **OpenCV**: Image processing

### Output Formats
- Images: PNG, JPEG, TIFF
- Masks: Binary PNG (0=mask, 255=keep)
- Metadata: EXIF (camera orientation)

### File Naming
- Images: `frame_001_cam0.png`, `frame_001_cam1.png`, ...
- Masks: `frame_001_cam0_mask.png`, `frame_001_cam1_mask.png`, ...

---

## 📄 License

**MIT License** - See LICENSE file for details

### Third-Party Components
- **Insta360 SDK**: © Arashi Vision Inc. (Bundled with permission)
- **PyTorch**: BSD License
- **OpenCV**: Apache 2.0 License
- **FFmpeg**: LGPL License
- **Ultralytics YOLOv8**: AGPL-3.0 License

---

## 🆘 Support

For issues, feature requests, or questions:
1. Check the troubleshooting section above
2. Review the included documentation
3. Contact: [Your support email/GitHub repo]

---

## 🔄 Upgrading to GPU Version

If you have an NVIDIA GPU and want 6-7× faster masking:

1. Download **360ToolkitGS-GPU** version
2. Extract to different folder
3. Run `install_pytorch_gpu.bat` (installs PyTorch with CUDA)
4. Launch `360ToolkitGS-GPU.exe`

GPU requirements:
- NVIDIA GPU with 4+ GB VRAM
- CUDA-compatible drivers
- ~3 GB additional disk space for PyTorch CUDA

---

## 📊 Version Comparison

| Feature | CPU Version | GPU Version |
|---------|-------------|-------------|
| Installation | Extract & run | Extract + install PyTorch |
| Setup time | 0 minutes | ~5 minutes |
| Download size | ~800 MB | ~700 MB |
| Disk space | ~1 GB | ~4 GB (with PyTorch) |
| Stage 1 (Extraction) | ✅ Same speed | ✅ Same speed |
| Stage 2 (Splitting) | ✅ Same speed | ✅ Same speed |
| Stage 3 (Masking) | ⚠️ 6-7× slower | ✅ Fast |
| Works everywhere | ✅ Yes | ❌ NVIDIA GPU required |

**Choose CPU version if**:
- You don't have NVIDIA GPU
- You want simplest installation
- Masking speed is not critical
- You process <100 images at a time

**Choose GPU version if**:
- You have NVIDIA GPU (GTX 1650 or better)
- You process many images (500+)
- You want fastest masking performance
- You don't mind 5-minute setup

---

**Built with ❤️ for the photogrammetry and Gaussian splatting community**
