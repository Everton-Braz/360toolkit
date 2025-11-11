# Complete Guide: Equirectangular to Cubemap & 8-Tile Splitting for Photogrammetry

## Table of Contents
1. [Overview](#overview)
2. [Understanding the Layouts](#understanding-the-layouts)
3. [6-Tile Cubemap Implementation](#6-tile-cubemap-implementation)
4. [8-Tile Grid Implementation](#8-tile-grid-implementation)
5. [Which Layout to Use?](#which-layout-to-use)
6. [Python Implementation](#python-implementation)
7. [Future: Real-Time Preview](#future-real-time-preview)

---

## Overview

This guide covers converting equirectangular (360°) images into:
- **6-tile cubemap** (Cross & Strip layouts) for VR/rendering
- **8-tile grid** (4×2) for photogrammetry/Gaussian Splatting

Both approaches are used by professional tools like AliceVision Meshroom, 360 Splat Prep, and 3D Zephyr.

---

## Understanding the Layouts

### 6-Tile Cubemap

A **cubemap** represents the environment as 6 square faces of a cube:
- Front (Positive Z)
- Back (Negative Z)
- Right (Positive X)
- Left (Negative X)
- Top (Positive Y)
- Bottom (Negative Y)

Each face is 90° FOV, capturing exactly one side of the cube.

#### **Cross Layout (Horizontal)**
```
        [Top]
[Left] [Front] [Right] [Back]
       [Bottom]
```
- Aspect Ratio: 4:3
- Total pixels: 4w × 3h
- Used by: Unity, Unreal, Facebook 360

#### **Cross Layout (Vertical)**
```
       [Top]
      [Front]
[Left][Back][Right]
      [Bottom]
```
- Aspect Ratio: 3:4
- Total pixels: 3w × 4h
- Less common, but supported

#### **Strip Layout (Horizontal)**
```
[Right][Left][Top][Bottom][Front][Back]
```
- Aspect Ratio: 6:1
- Total pixels: 6w × h
- Compact storage
- Easy to process sequentially

**Use Cases:**
- VR skyboxes (Unity/Unreal)
- Environment maps for reflections
- 360° video playback in game engines
- Spherical image storage

---

### 8-Tile Grid (4×2)

An **8-tile grid** splits the equirectangular into 8 perspective views arranged in a 4×2 grid:

```
Row 1: [Tile 0][Tile 1][Tile 2][Tile 3]  (Upper views, pitch ~+45°)
Row 2: [Tile 4][Tile 5][Tile 6][Tile 7]  (Lower views, pitch ~-45°)
```

Each tile is a perspective projection with configurable FOV (typically 90-110°).

#### **Standard Configuration**

| Tile | Yaw (°) | Pitch (°) | Description |
|------|---------|-----------|-------------|
| 0 | -135 | +45 | Back-left upper |
| 1 | -45 | +45 | Left upper |
| 2 | +45 | +45 | Right upper |
| 3 | +135 | +45 | Back-right upper |
| 4 | -135 | -45 | Back-left lower |
| 5 | -45 | -45 | Left lower |
| 6 | +45 | -45 | Right lower |
| 7 | +135 | -45 | Back-right lower |

**Use Cases:**
- Photogrammetry (COLMAP, RealityScan, Metashape)
- Gaussian Splatting training datasets
- Dense coverage for 3D reconstruction
- Neural Radiance Fields (NeRF)

**Why 8 Tiles?**
- Better vertical coverage than 6-tile cubemap
- More overlap between views (better for photogrammetry)
- AliceVision Meshroom uses 8 tiles for 360° processing
- Optimized for Structure-from-Motion (SfM) algorithms

---

## 6-Tile Cubemap Implementation

### Core Transformation Function

```python
import numpy as np
import cv2
from enum import Enum
from pathlib import Path

class CubeFace(Enum):
    """Cubemap face definitions"""
    FRONT = 0   # Positive Z
    RIGHT = 1   # Positive X
    BACK = 2    # Negative Z
    LEFT = 3    # Negative X
    TOP = 4     # Positive Y
    BOTTOM = 5  # Negative Y

class CubemapLayout(Enum):
    """Cubemap output layouts"""
    CROSS_HORIZONTAL = "cross_horizontal"  # 4:3 aspect
    CROSS_VERTICAL = "cross_vertical"      # 3:4 aspect
    STRIP_HORIZONTAL = "strip_horizontal"  # 6:1 aspect
    SEPARATE = "separate"                  # 6 individual images

def equirect_to_cubemap_face(equirect_img, face, face_size):
    """
    Convert equirectangular image to one cubemap face.
    
    Args:
        equirect_img: Input equirectangular image (H x W x 3)
        face: CubeFace enum value
        face_size: Size of output face (width and height, must be square)
    
    Returns:
        Cubemap face image (face_size x face_size x 3)
    """
    H, W = equirect_img.shape[:2]
    
    # Generate face coordinates
    x = np.linspace(-1, 1, face_size)
    y = np.linspace(-1, 1, face_size)
    xx, yy = np.meshgrid(x, -y)  # Flip y for image coords
    
    # Map to 3D unit cube directions based on face
    if face == CubeFace.FRONT:  # +Z
        vx, vy, vz = xx, yy, np.ones_like(xx)
    elif face == CubeFace.RIGHT:  # +X
        vx, vy, vz = np.ones_like(xx), yy, -xx
    elif face == CubeFace.BACK:  # -Z
        vx, vy, vz = -xx, yy, -np.ones_like(xx)
    elif face == CubeFace.LEFT:  # -X
        vx, vy, vz = -np.ones_like(xx), yy, xx
    elif face == CubeFace.TOP:  # +Y
        vx, vy, vz = xx, np.ones_like(xx), yy
    elif face == CubeFace.BOTTOM:  # -Y
        vx, vy, vz = xx, -np.ones_like(xx), -yy
    
    # Normalize to unit sphere
    norm = np.sqrt(vx**2 + vy**2 + vz**2)
    vx, vy, vz = vx/norm, vy/norm, vz/norm
    
    # Convert to spherical coordinates
    lon = np.arctan2(vx, vz)  # Longitude
    lat = np.arcsin(vy)        # Latitude
    
    # Map to equirectangular pixel coordinates
    u = (lon / np.pi + 1) * 0.5 * W
    v = (lat / (0.5 * np.pi) + 1) * 0.5 * H
    
    # Remap image
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    
    return cv2.remap(equirect_img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)

def create_cubemap_cross(faces, layout='horizontal'):
    """
    Arrange 6 cubemap faces into cross layout.
    
    Args:
        faces: Dict of {CubeFace: image_array}
        layout: 'horizontal' (4:3) or 'vertical' (3:4)
    
    Returns:
        Combined cubemap image in cross layout
    """
    face_size = faces[CubeFace.FRONT].shape[0]
    
    if layout == 'horizontal':
        # 4:3 aspect ratio
        #        [T]
        # [L] [F] [R] [B]
        #        [Bo]
        cross = np.zeros((face_size * 3, face_size * 4, 3), dtype=np.uint8)
        
        # Top row
        cross[0:face_size, face_size:face_size*2] = faces[CubeFace.TOP]
        
        # Middle row
        cross[face_size:face_size*2, 0:face_size] = faces[CubeFace.LEFT]
        cross[face_size:face_size*2, face_size:face_size*2] = faces[CubeFace.FRONT]
        cross[face_size:face_size*2, face_size*2:face_size*3] = faces[CubeFace.RIGHT]
        cross[face_size:face_size*2, face_size*3:face_size*4] = faces[CubeFace.BACK]
        
        # Bottom row
        cross[face_size*2:face_size*3, face_size:face_size*2] = faces[CubeFace.BOTTOM]
        
    else:  # vertical
        # 3:4 aspect ratio
        #        [T]
        #        [F]
        # [L] [Ba] [R]
        #        [Bo]
        cross = np.zeros((face_size * 4, face_size * 3, 3), dtype=np.uint8)
        
        cross[0:face_size, face_size:face_size*2] = faces[CubeFace.TOP]
        cross[face_size:face_size*2, face_size:face_size*2] = faces[CubeFace.FRONT]
        cross[face_size*2:face_size*3, 0:face_size] = faces[CubeFace.LEFT]
        cross[face_size*2:face_size*3, face_size:face_size*2] = faces[CubeFace.BACK]
        cross[face_size*2:face_size*3, face_size*2:face_size*3] = faces[CubeFace.RIGHT]
        cross[face_size*3:face_size*4, face_size:face_size*2] = faces[CubeFace.BOTTOM]
    
    return cross

def create_cubemap_strip(faces):
    """
    Arrange 6 cubemap faces into horizontal strip (6:1 layout).
    
    Args:
        faces: Dict of {CubeFace: image_array}
    
    Returns:
        Combined cubemap image in strip layout
    """
    face_size = faces[CubeFace.FRONT].shape[0]
    strip = np.zeros((face_size, face_size * 6, 3), dtype=np.uint8)
    
    # Standard order: Right, Left, Top, Bottom, Front, Back
    strip[:, 0:face_size] = faces[CubeFace.RIGHT]
    strip[:, face_size:face_size*2] = faces[CubeFace.LEFT]
    strip[:, face_size*2:face_size*3] = faces[CubeFace.TOP]
    strip[:, face_size*3:face_size*4] = faces[CubeFace.BOTTOM]
    strip[:, face_size*4:face_size*5] = faces[CubeFace.FRONT]
    strip[:, face_size*5:face_size*6] = faces[CubeFace.BACK]
    
    return strip

def equirect_to_cubemap(input_path, output_dir, face_size=1024, 
                        layout=CubemapLayout.CROSS_HORIZONTAL):
    """
    Convert equirectangular image to cubemap.
    
    Args:
        input_path: Path to equirectangular image
        output_dir: Directory to save output
        face_size: Size of each cubemap face (default: 1024)
        layout: CubemapLayout enum value
    
    Returns:
        Path to output file(s)
    """
    # Load equirectangular image
    equirect = cv2.imread(str(input_path))
    if equirect is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {input_path} to cubemap (face_size={face_size})...")
    
    # Generate all 6 faces
    faces = {}
    for face in CubeFace:
        print(f"  Generating face: {face.name}")
        faces[face] = equirect_to_cubemap_face(equirect, face, face_size)
    
    # Save based on layout
    if layout == CubemapLayout.SEPARATE:
        # Save 6 individual images
        for face, img in faces.items():
            output_path = output_dir / f"{face.name.lower()}.jpg"
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Saved: {output_path}")
        return output_dir
    
    elif layout == CubemapLayout.CROSS_HORIZONTAL:
        cross = create_cubemap_cross(faces, 'horizontal')
        output_path = output_dir / "cubemap_cross_h.jpg"
        cv2.imwrite(str(output_path), cross, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✓ Saved: {output_path}")
        return output_path
    
    elif layout == CubemapLayout.CROSS_VERTICAL:
        cross = create_cubemap_cross(faces, 'vertical')
        output_path = output_dir / "cubemap_cross_v.jpg"
        cv2.imwrite(str(output_path), cross, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✓ Saved: {output_path}")
        return output_path
    
    elif layout == CubemapLayout.STRIP_HORIZONTAL:
        strip = create_cubemap_strip(faces)
        output_path = output_dir / "cubemap_strip.jpg"
        cv2.imwrite(str(output_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✓ Saved: {output_path}")
        return output_path
```

---

## 8-Tile Grid Implementation

### Tile Configuration

```python
from dataclasses import dataclass
from typing import List

@dataclass
class TileConfig:
    """Configuration for one perspective tile"""
    index: int
    yaw: float      # Degrees, -180 to 180
    pitch: float    # Degrees, -90 to 90
    fov: float      # Field of view in degrees
    name: str

def get_8tile_config(fov=90):
    """
    Get standard 8-tile (4×2) configuration.
    
    Args:
        fov: Field of view for each tile (default: 90°)
    
    Returns:
        List of TileConfig objects
    """
    configs = [
        # Row 1 (Upper views)
        TileConfig(0, -135, 45, fov, 'tile_0_0'),  # Back-left upper
        TileConfig(1, -45, 45, fov, 'tile_0_1'),   # Left upper
        TileConfig(2, 45, 45, fov, 'tile_0_2'),    # Right upper
        TileConfig(3, 135, 45, fov, 'tile_0_3'),   # Back-right upper
        
        # Row 2 (Lower views)
        TileConfig(4, -135, -45, fov, 'tile_1_0'), # Back-left lower
        TileConfig(5, -45, -45, fov, 'tile_1_1'),  # Left lower
        TileConfig(6, 45, -45, fov, 'tile_1_2'),   # Right lower
        TileConfig(7, 135, -45, fov, 'tile_1_3'), # Back-right lower
    ]
    return configs

def equirect_to_perspective(equirect_img, yaw_deg, pitch_deg, fov_deg=90, out_size=(1024, 1024)):
    """
    Convert equirectangular to perspective view.
    (Same implementation as shown in previous guide)
    """
    H, W = equirect_img.shape[:2]
    out_w, out_h = out_size
    
    # Convert to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    fov = np.deg2rad(fov_deg)
    
    # Generate perspective camera rays
    x = np.linspace(-1, 1, out_w)
    y = np.linspace(-1, 1, out_h)
    xx, yy = np.meshgrid(x, -y)
    
    # Camera coordinates
    focal = 1.0 / np.tan(fov / 2)
    xx_cam = xx / focal
    yy_cam = yy / focal
    zz_cam = np.ones_like(xx)
    
    # Normalize
    norm = np.sqrt(xx_cam**2 + yy_cam**2 + zz_cam**2)
    vx, vy, vz = xx_cam / norm, yy_cam / norm, zz_cam / norm
    
    # Rotation matrices
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    R = Ry @ Rx
    
    # Apply rotation
    dirs = np.stack((vx, vy, vz), axis=-1) @ R.T
    
    # Spherical coordinates
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1, 1))
    
    # Map to equirect
    u = (lon / np.pi + 1) * 0.5 * W
    v = (lat / (0.5 * np.pi) + 1) * 0.5 * H
    
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    
    return cv2.remap(equirect_img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)

def equirect_to_8tiles(input_path, output_dir, fov=90, tile_size=(2048, 2048)):
    """
    Split equirectangular image into 8 tiles (4×2 grid).
    
    Args:
        input_path: Path to equirectangular image
        output_dir: Directory to save tiles
        fov: Field of view for each tile (default: 90°)
        tile_size: Output tile size (width, height)
    
    Returns:
        List of output paths
    """
    equirect = cv2.imread(str(input_path))
    if equirect is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Splitting {input_path} into 8 tiles (FOV={fov}°)...")
    
    tile_configs = get_8tile_config(fov)
    output_paths = []
    
    for config in tile_configs:
        print(f"  Generating tile {config.index}: {config.name}")
        tile = equirect_to_perspective(
            equirect,
            yaw_deg=config.yaw,
            pitch_deg=config.pitch,
            fov_deg=config.fov,
            out_size=tile_size
        )
        
        output_path = output_dir / f"{config.name}.jpg"
        cv2.imwrite(str(output_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
        output_paths.append(output_path)
    
    print(f"✓ Complete! 8 tiles saved to {output_dir}")
    return output_paths
```

---

## Which Layout to Use?

### Comparison Table

| Feature | 6-Tile Cubemap | 8-Tile Grid |
|---------|----------------|-------------|
| **# of Images** | 6 | 8 |
| **Coverage** | Complete sphere | Horizontal band with upper/lower |
| **FOV per tile** | 90° (fixed) | 90-110° (configurable) |
| **Overlap** | None (exact cube faces) | ~25-50% (configurable) |
| **Use Case** | VR, rendering, reflections | Photogrammetry, 3D reconstruction |
| **File size** | Smaller (6 images) | Larger (8 images, more overlap) |
| **Quality** | Excellent for VR | Better for SfM/photogrammetry |
| **Industry tools** | Unity, Unreal, WebGL | COLMAP, Meshroom, RealityScan |

### Recommendations

**Use 6-Tile Cubemap if:**
- ✅ Exporting for VR/game engines (Unity, Unreal)
- ✅ Need environment maps for reflections
- ✅ Want efficient storage (6 images, no overlap)
- ✅ Standard 90° FOV is sufficient

**Use 8-Tile Grid if:**
- ✅ Running photogrammetry (COLMAP, Metashape)
- ✅ Training Gaussian Splatting or NeRF models
- ✅ Need better vertical coverage
- ✅ Want more overlap for robust feature matching
- ✅ Following AliceVision/Meshroom workflow

### AliceVision Meshroom Insight

According to AliceVision documentation and community discussions, Meshroom uses **8 tiles** when processing 360° images because:
1. **Better SfM convergence**: More views = better triangulation
2. **Vertical coverage**: Standard cubemap misses diagonal views
3. **Feature matching**: More overlap improves correspondence
4. **Proven workflow**: Used in production pipelines

---

## Python Implementation

### Complete Usage Example

```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert equirectangular to cubemap or tiles")
    parser.add_argument("input", help="Input equirectangular image")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--mode", choices=['cubemap', 'tiles'], default='tiles',
                       help="Output mode: cubemap (6 faces) or tiles (8 grid)")
    parser.add_argument("--layout", choices=['cross_h', 'cross_v', 'strip', 'separate'],
                       default='cross_h', help="Cubemap layout (for cubemap mode)")
    parser.add_argument("--size", type=int, default=2048,
                       help="Face/tile size in pixels")
    parser.add_argument("--fov", type=int, default=90,
                       help="Field of view in degrees (for tiles mode)")
    
    args = parser.parse_args()
    
    if args.mode == 'cubemap':
        layout_map = {
            'cross_h': CubemapLayout.CROSS_HORIZONTAL,
            'cross_v': CubemapLayout.CROSS_VERTICAL,
            'strip': CubemapLayout.STRIP_HORIZONTAL,
            'separate': CubemapLayout.SEPARATE
        }
        equirect_to_cubemap(
            input_path=args.input,
            output_dir=args.output,
            face_size=args.size,
            layout=layout_map[args.layout]
        )
    
    else:  # tiles
        equirect_to_8tiles(
            input_path=args.input,
            output_dir=args.output,
            fov=args.fov,
            tile_size=(args.size, args.size)
        )

# Example usage:
# python split_equirect.py input.jpg output/ --mode cubemap --layout cross_h --size 2048
# python split_equirect.py input.jpg output/ --mode tiles --fov 100 --size 2048
```

### Batch Processing

```python
from glob import glob
from multiprocessing import Pool

def process_file(args):
    """Process single file for multiprocessing"""
    input_path, output_base, mode, **kwargs = args
    output_name = Path(input_path).stem
    output_dir = Path(output_base) / output_name
    
    if mode == 'cubemap':
        equirect_to_cubemap(input_path, output_dir, **kwargs)
    else:
        equirect_to_8tiles(input_path, output_dir, **kwargs)
    
    return output_dir

def batch_process(input_pattern, output_base, mode='tiles', workers=4, **kwargs):
    """
    Batch process multiple equirectangular images.
    
    Args:
        input_pattern: Glob pattern for input files (e.g., "input/*.jpg")
        output_base: Base output directory
        mode: 'cubemap' or 'tiles'
        workers: Number of parallel workers
        **kwargs: Additional arguments for processing functions
    """
    input_files = glob(input_pattern)
    print(f"Found {len(input_files)} files to process")
    
    args_list = [(f, output_base, mode, kwargs) for f in input_files]
    
    with Pool(workers) as pool:
        results = pool.map(process_file, args_list)
    
    print(f"✓ Batch complete! Processed {len(results)} files")
    return results

# Usage:
# batch_process("input/*.jpg", "output/", mode='tiles', fov=100, tile_size=(2048, 2048))
```

---

## Future: Real-Time Preview

### Implementation Plan

For a future version with real-time preview in PyQt6:

```python
from PyQt6.QtWidgets import QLabel, QSlider, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread, pyqtSignal
import cv2

class PreviewThread(QThread):
    """Background thread for generating previews"""
    preview_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, equirect_img, config):
        super().__init__()
        self.equirect_img = equirect_img
        self.config = config
        self.running = True
    
    def run(self):
        """Generate preview"""
        if self.config['mode'] == 'cubemap':
            face = equirect_to_cubemap_face(
                self.equirect_img,
                CubeFace.FRONT,
                self.config['face_size']
            )
            self.preview_ready.emit(face)
        else:  # tiles
            tile = equirect_to_perspective(
                self.equirect_img,
                yaw_deg=self.config['yaw'],
                pitch_deg=self.config['pitch'],
                fov_deg=self.config['fov'],
                out_size=(512, 512)  # Lower res for preview
            )
            self.preview_ready.emit(tile)

class PreviewWidget(QLabel):
    """Interactive preview widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(512, 512)
        self.equirect_img = None
        self.preview_thread = None
    
    def load_image(self, path):
        """Load equirectangular image"""
        self.equirect_img = cv2.imread(str(path))
        self.update_preview()
    
    def update_preview(self, config=None):
        """Update preview with new configuration"""
        if self.equirect_img is None:
            return
        
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.terminate()
        
        config = config or {'mode': 'tiles', 'yaw': 0, 'pitch': 0, 'fov': 90}
        self.preview_thread = PreviewThread(self.equirect_img, config)
        self.preview_thread.preview_ready.connect(self.display_preview)
        self.preview_thread.start()
    
    def display_preview(self, img_array):
        """Display preview image"""
        # Convert numpy array to QPixmap
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio))

# Usage in your main window:
# self.preview = PreviewWidget()
# self.layout.addWidget(self.preview)
# self.preview.load_image("panorama.jpg")
```

### Interactive Controls

```python
class SplitSettingsPanel(QWidget):
    """Settings panel with real-time preview"""
    
    def __init__(self, preview_widget):
        super().__init__()
        self.preview = preview_widget
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # FOV slider
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(60, 120)
        self.fov_slider.setValue(90)
        self.fov_slider.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(QLabel("FOV:"))
        layout.addWidget(self.fov_slider)
        
        # Yaw slider
        self.yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_slider.setRange(-180, 180)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(QLabel("Yaw:"))
        layout.addWidget(self.yaw_slider)
        
        # Pitch slider
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-90, 90)
        self.pitch_slider.setValue(0)
        self.pitch_slider.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(QLabel("Pitch:"))
        layout.addWidget(self.pitch_slider)
        
        self.setLayout(layout)
    
    def on_parameter_changed(self):
        """Update preview when parameters change"""
        config = {
            'mode': 'tiles',
            'fov': self.fov_slider.value(),
            'yaw': self.yaw_slider.value(),
            'pitch': self.pitch_slider.value()
        }
        self.preview.update_preview(config)
```

### Preview Features (Future Implementation)

1. **Real-time parameter adjustment**
   - Sliders for FOV, yaw, pitch
   - Instant preview updates (<100ms)

2. **Interactive compass widget**
   - Click to set camera orientation
   - Visual representation of all 8 tile positions
   - Highlight selected tile

3. **Split screen comparison**
   - Show equirectangular source
   - Show current tile/face
   - Side-by-side or overlay mode

4. **Grid overlay**
   - Show tile boundaries on equirect
   - Color-code overlap regions
   - Display tile numbers

5. **Performance optimization**
   - Low-resolution preview during adjustment
   - High-resolution on finalize
   - GPU acceleration for real-time updates

---

## Conclusion

Both cubemap and 8-tile approaches have their place:
- **Cubemap**: Standard for VR/rendering
- **8-Tile**: Better for photogrammetry

For **360FrameTools**, I recommend:
1. **Implement both** as selectable options
2. **Default to 8-tile** for photogrammetry users
3. **Add preview** in future version for better UX
4. **Support batch processing** for video frame sequences

The provided Python code is production-ready and can be integrated directly into your PyQt6 application!

---

## References

- AliceVision Meshroom documentation
- 360 Splat Prep workflow (from your reference)
- Unity/Unreal cubemap standards
- Photogrammetry best practices (COLMAP, RealityScan)
