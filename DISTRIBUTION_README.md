# 360ToolkitGS - Distribution Package

**Complete photogrammetry preprocessing toolkit for Insta360 cameras**

Extract frames → Split perspectives → AI masking → Done!

---

## 📦 What's This?

**360ToolkitGS** is a desktop application that converts Insta360 360° videos into perspective images with AI-based masking, ready for photogrammetry (RealityScan, Metashape, COLMAP) and Gaussian Splatting (3DGS, Nerfstudio).

### Three-Stage Pipeline

1. **Stage 1: Frame Extraction**
   - Extract frames from `.INSV` or `.mp4` files
   - Official Insta360 SDK (bypasses Insta360 Studio)
   - Configurable FPS (0.1 - 30 frames/second)
   - Output: Equirectangular panoramas

2. **Stage 2: Perspective Splitting**
   - Convert 360° images to multiple perspective views
   - Compass-based camera positioning (default: 8 cameras, 110° FOV)
   - Interactive preview and configuration
   - Output: Rectilinear perspective images

3. **Stage 3: AI Masking** (Optional)
   - YOLOv8 instance segmentation
   - Detect and mask: persons, objects, animals
   - 5 model sizes (nano to xlarge)
   - Output: Binary masks for photogrammetry

---

## 🎯 Which Version Do I Need?

We provide **TWO versions**:

### 🚀 GPU Version (Recommended for NVIDIA GPU owners)
- **Download size**: ~700 MB
- **Requirements**: NVIDIA GPU (GTX 1650 or better)
- **Setup**: 5 minutes (install PyTorch once)
- **Performance**: 6-7× faster masking
- **Best for**: Professional workflows, large datasets (500+ images)

### 💻 CPU Version (Recommended for everyone else)
- **Download size**: ~800 MB
- **Requirements**: Any PC (no GPU needed)
- **Setup**: 0 minutes (extract & run)
- **Performance**: Slower masking but works everywhere
- **Best for**: Beginners, small projects (<100 images), laptops

**Not sure?** See `VERSION_COMPARISON.md` for detailed comparison.

**Quick rule**: If you have NVIDIA GPU → GPU version. Otherwise → CPU version.

---

## 📥 Download Links

| Version | Size | Download |
|---------|------|----------|
| **GPU Version** | ~700 MB | `360ToolkitGS-GPU.zip` |
| **CPU Version** | ~800 MB | `360ToolkitGS-CPU.zip` |

**Also download**: `VERSION_COMPARISON.md` (helps you choose the right version)

---

## 🚀 Quick Start

### GPU Version
1. Extract `360ToolkitGS-GPU.zip`
2. Run `install_pytorch_gpu.bat` (one-time, ~5 min)
3. Launch `360ToolkitGS-GPU.exe`
4. Process your first project!

**Full instructions**: See `README_GPU_VERSION.md` (included in ZIP)

---

### CPU Version
1. Extract `360ToolkitGS-CPU.zip`
2. Launch `360ToolkitGS-CPU.exe`
3. Process your first project!

**Full instructions**: See `README_CPU_VERSION.md` (included in ZIP)

---

## 💡 Example Workflow

1. **Select input**: Choose your Insta360 `.INSV` or `.mp4` file
2. **Configure extraction**: Set FPS (e.g., 1 frame/second)
3. **Configure splitting**: Keep default (8 cameras, 110° FOV) or customize
4. **Enable masking**: Check "Persons" category (optional)
5. **Click "Start Batch Processing"**
6. **Wait**: Application processes all stages automatically
7. **Done**: Output folder contains perspective images + masks

**Time estimate** (500 images):
- GPU version: ~1-2 minutes
- CPU version: ~5-10 minutes

---

## 📊 System Requirements

### GPU Version
**Minimum**:
- Windows 10/11 (64-bit)
- NVIDIA GPU: GTX 1650 (4 GB VRAM)
- RAM: 8 GB
- Storage: 5 GB free

**Recommended**:
- NVIDIA GPU: RTX 3060+ (8 GB VRAM)
- RAM: 16 GB
- Storage: 10 GB free

---

### CPU Version
**Minimum**:
- Windows 10/11 (64-bit)
- CPU: Intel Core i5 or AMD Ryzen 5 (4 cores)
- RAM: 8 GB
- Storage: 2 GB free

**Recommended**:
- CPU: Intel Core i7 or AMD Ryzen 7 (6+ cores)
- RAM: 16 GB
- Storage: 5 GB free

---

## 🎬 Supported Input Formats

- **Insta360 native**: `.INSV` (dual-fisheye)
- **Video**: `.mp4` (360° equirectangular)
- **Images**: `.jpg`, `.png`, `.tiff` (equirectangular, for Stage 2 only)

**Tested cameras**:
- Insta360 X3
- Insta360 X4
- Insta360 ONE RS 1-Inch 360
- Other Insta360 360° cameras

---

## 📤 Output Formats

### Images
- PNG (lossless, recommended)
- JPEG (smaller files)
- TIFF (maximum quality)

### Masks
- Binary PNG (RealityScan format)
- Naming: `<image_name>_mask.png`
- Convention: Black (0) = mask, White (255) = keep

---

## 🔧 What's Included (Technical)

Both versions include:
- **Insta360 MediaSDK 3.0.5**: Official SDK for dual-fisheye stitching
- **FFmpeg**: Video processing and extraction
- **OpenCV**: Image processing
- **NumPy**: Array operations
- **PyQt6**: Modern desktop interface
- **YOLOv8**: AI segmentation models (auto-download on first use)

**GPU version additionally requires** (user installs):
- PyTorch 2.x with CUDA 11.8 (~2.8 GB)

**CPU version additionally includes**:
- PyTorch 2.x CPU (~500 MB)

---

## 📚 Documentation

### Included in Downloads
- `README_GPU_VERSION.md` / `README_CPU_VERSION.md`: User guide for your version
- `LICENSE`: MIT License + third-party attributions
- `install_pytorch_gpu.bat`: PyTorch installer (GPU version only)

### Available Separately
- `VERSION_COMPARISON.md`: Detailed comparison of GPU vs CPU versions
- `BUILD_GUIDE.md`: For developers who want to build from source

---

## 🎓 Use Cases

### Photogrammetry
- **RealityScan**: Import images + masks for clean 3D models
- **Metashape**: Multi-camera photogrammetry workflows
- **COLMAP**: Structure-from-Motion reconstruction
- **RealityCapture**: Professional 3D scanning

### Gaussian Splatting
- **3D Gaussian Splatting**: Novel view synthesis
- **Nerfstudio**: NeRF and Gaussian Splatting framework
- **Luma AI**: Cloud-based 3D reconstruction

### Research & Development
- **Computer vision**: Multi-view datasets
- **Machine learning**: Training data generation
- **360° video analysis**: Extract and process frames

---

## ❓ FAQ

### Can I use both versions?
**Yes!** Install to different folders. Both use same project formats.

### Do I need Insta360 Studio?
**No!** This app bypasses Insta360 Studio entirely using the official SDK.

### Can I process videos from other 360° cameras?
**Yes!** Stage 2 works with any equirectangular image. Stage 1 is optimized for Insta360 but may work with other `.mp4` files.

### Is masking required?
**No!** Masking (Stage 3) is optional. You can skip it if you don't need to remove people/objects.

### Can I mask other objects besides persons?
**Yes!** Select categories: persons, personal objects (bags, phones), animals.

### What if I get "SDK not found" error?
The SDK is bundled. If error occurs, re-extract the ZIP file completely. Don't run from inside ZIP.

### GPU version says "CUDA not available"?
Run `install_pytorch_gpu.bat` included in the GPU version. This installs PyTorch with CUDA support.

### How do I update to newer version?
Download new version, extract to new folder. Your settings and projects are portable.

---

## 🐛 Troubleshooting

### Application won't start
1. Extract all files (don't run from ZIP)
2. Windows SmartScreen: Click "More info" → "Run anyway"
3. Check antivirus didn't quarantine files

### "SDK not found" error
- Re-extract ZIP completely
- Ensure `_internal/sdk/` folder exists

### Masking is very slow
- CPU version: Expected behavior
- Use smaller model (nano/small)
- For faster masking: Use GPU version

### Out of memory
- Close other applications
- Use smaller YOLOv8 model
- Reduce output resolution
- Process fewer images per batch

**More help**: See README.txt in your downloaded version

---

## 📜 License

**MIT License** - Free for personal and commercial use

### Third-Party Components
- **Insta360 SDK**: © Arashi Vision Inc. (Bundled with permission)
- **PyTorch**: BSD License
- **OpenCV**: Apache 2.0
- **FFmpeg**: LGPL 2.1+
- **YOLOv8**: AGPL-3.0 (models auto-download)

**Full license text**: See LICENSE file in download

---

## 🆘 Support

### Before asking for help:
1. Read README.txt in your version
2. Check troubleshooting section above
3. Verify system requirements met
4. Try with different input file (test file issue)

### Report issues:
- GitHub: [Your repository URL]
- Email: [Your support email]

**Include**:
- Version used (GPU or CPU)
- Windows version
- Error message (screenshot)
- Steps to reproduce

---

## 🔄 Updates

### Current Version: 1.0 (January 2025)

**Check for updates**: [Your update URL or GitHub releases]

### Changelog
- **v1.0** (2025-01): Initial release
  - Insta360 SDK integration
  - Three-stage pipeline
  - GPU and CPU versions
  - Multi-category masking

---

## 🙏 Credits

**Developed for the photogrammetry and Gaussian splatting community**

### Powered By
- Insta360 Camera SDK (Official)
- PyTorch (AI framework)
- Ultralytics YOLOv8 (Object detection)
- FFmpeg (Video processing)

### Special Thanks
- Insta360 for providing royalty-free SDK
- PyTorch team for excellent ML framework
- Ultralytics for YOLOv8
- Open source community

---

## 🚀 Getting Started

1. **Choose your version**: GPU (fast masking) or CPU (works everywhere)
2. **Download**: Get the ZIP file
3. **Extract**: To any folder
4. **Setup**: GPU: Run installer script | CPU: Skip setup
5. **Launch**: Double-click the .exe
6. **Process**: Your first Insta360 video!

**Need help choosing?** Download `VERSION_COMPARISON.md`

---

## 📞 Quick Links

| Resource | Description |
|----------|-------------|
| `README_GPU_VERSION.md` | Complete guide for GPU version |
| `README_CPU_VERSION.md` | Complete guide for CPU version |
| `VERSION_COMPARISON.md` | GPU vs CPU detailed comparison |
| `BUILD_GUIDE.md` | Build from source (developers) |
| `LICENSE` | MIT License + attributions |

---

**Ready to start?** Download your version and extract it!

**Questions?** Check the README.txt in your download first.

**Built with ❤️ for creators and researchers**
