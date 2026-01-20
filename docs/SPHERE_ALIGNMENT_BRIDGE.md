# SphereAlignment Bridge Architecture

## Overview

Our app acts as a **bridge** to SphereAlignment, which does the actual COLMAP alignment work.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        360FrameTools (Our App)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. User clicks "Run COLMAP Alignment"                                 │
│     ↓                                                                   │
│  2. Prepare perspective images                                         │
│     ↓                                                                   │
│  3. Call SphereAlignment.exe ─────────┐                                │
│                                       │                                 │
└───────────────────────────────────────┼─────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              SphereAlignment.exe (External App)                         │
│          C:\Users\Everton-PC\Documents\APLICATIVOS\SphereAlignment      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  • Reads perspective images                                            │
│  • Runs COLMAP feature extraction                                      │
│  • Performs feature matching                                           │
│  • Calculates camera positions (X, Y, Z)                               │
│  • Outputs COLMAP files:                                               │
│    - images.txt  ←─────── We read this!                               │
│    - cameras.txt                                                       │
│    - points3D.txt                                                      │
│                                                                         │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        360FrameTools (Our App)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  4. Read images.txt (COLMAP output)                                    │
│     ↓                                                                   │
│  5. Parse camera positions:                                            │
│     frame_001_cam0.jpg → (1.234, 5.678, 0.123)                        │
│     frame_001_cam1.jpg → (1.235, 5.679, 0.124)                        │
│     ...                                                                │
│     ↓                                                                   │
│  6. Group by source frame:                                             │
│     frame_001.jpg → avg of 8 camera positions                         │
│     ↓                                                                   │
│  7. Embed positions in equirectangular metadata                        │
│     ↓                                                                   │
│  8. When splitting again, propagate positions to perspectives          │
│     ↓                                                                   │
│  9. Export images with position metadata                               │
│     (RealityScan can now use positions as alignment priors!)           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why This Architecture?

### Phase 1: Bridge (Current)
- **Quick implementation**: Use existing SphereAlignment tool
- **Proven COLMAP workflow**: SphereAlignment already works
- **Focus on integration**: We handle data flow, not alignment algorithms
- **External dependency**: Requires SphereAlignment.exe

### Phase 2: Built-in (Future)
- **Integrate COLMAP**: Build alignment directly into our app
- **No external dependency**: Self-contained executable
- **Better UX**: No separate installation needed
- **More control**: Customize alignment parameters

## File Flow

### Input
```
perspectives/
├── frame_001_cam0.jpg  ┐
├── frame_001_cam1.jpg  │
├── frame_001_cam2.jpg  │ Group 1: From frame_001.jpg
├── frame_001_cam3.jpg  │
├── frame_001_cam4.jpg  │
├── frame_001_cam5.jpg  │
├── frame_001_cam6.jpg  │
├── frame_001_cam7.jpg  ┘
├── frame_002_cam0.jpg  ┐
└── ...                 └ Group 2: From frame_002.jpg
```

### SphereAlignment Output
```
output/
└── sparse/
    └── 0/
        ├── cameras.txt      (camera intrinsics)
        ├── images.txt       ← WE READ THIS!
        └── points3D.txt     (3D points)
```

### images.txt Content (What We Parse)
```
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#           ↑───────────────↑  ↑──────────↑         ↑
#          Quaternion (rot)   Position (x,y,z)   Filename

1 0.999 0.001 -0.002 0.001 1.234 5.678 0.123 1 frame_001_cam0.jpg
   ↑──────────────────────↑ ↑──────────────────↑  ↑────────────────
   Skip (rotation data)     EXTRACT THIS!         Match to source frame

2 0.998 0.002 -0.001 0.002 1.235 5.679 0.124 1 frame_001_cam1.jpg
...
```

### Our Position Extraction
```python
# Read TX, TY, TZ from images.txt
positions = {
    'frame_001_cam0.jpg': (1.234, 5.678, 0.123),
    'frame_001_cam1.jpg': (1.235, 5.679, 0.124),
    'frame_001_cam2.jpg': (1.236, 5.680, 0.125),
    # ... 8 cameras total for frame_001
}

# Group by source frame
frame_positions = {
    'frame_001.jpg': average([
        (1.234, 5.678, 0.123),
        (1.235, 5.679, 0.124),
        (1.236, 5.680, 0.125),
        # ... all 8 positions
    ])  # → (1.235, 5.679, 0.124) average
}
```

### Metadata Embedding
```json
// frame_001.jpg metadata
{
  "source": "frame_001.jpg",
  "colmap_position": {
    "x": 1.235,
    "y": 5.679,
    "z": 0.124
  },
  "colmap_aligned": true,
  "num_perspectives": 8
}
```

### Final Output (Perspectives with Positions)
```
perspectives_with_positions/
├── frame_001_cam0.jpg  ← Has position (1.235, 5.679, 0.124) from parent
├── frame_001_cam1.jpg  ← Has position (1.235, 5.679, 0.124) from parent
...
```

## Code Bridge Points

### 1. Call SphereAlignment
```python
subprocess.run([
    r"C:\Users\Everton-PC\Documents\APLICATIVOS\SphereAlignment\SphereAlignment.exe",
    "--input", "perspectives/",
    "--output", "colmap_output/",
    "--quality", "high"
])
```

### 2. Read Output
```python
images_txt = Path("colmap_output/sparse/0/images.txt")
positions = parse_colmap_images(images_txt)
```

### 3. Parse Positions
```python
def parse_colmap_images(images_file):
    positions = {}
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            # parts[5], parts[6], parts[7] = TX, TY, TZ
            # parts[9] = image name
            
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]
            
            positions[name] = (tx, ty, tz)
    
    return positions
```

### 4. Group by Frame
```python
def map_to_frames(positions):
    from collections import defaultdict
    import numpy as np
    
    groups = defaultdict(list)
    
    for image_name, position in positions.items():
        # frame_001_cam0.jpg → frame_001.jpg
        frame_name = re.sub(r'_cam\d+', '', image_name)
        groups[frame_name].append(position)
    
    frame_positions = {}
    for frame, pos_list in groups.items():
        frame_positions[frame] = tuple(np.mean(pos_list, axis=0))
    
    return frame_positions
```

### 5. Embed Metadata
```python
from src.pipeline.metadata_handler import MetadataHandler

handler = MetadataHandler()

for frame_name, position in frame_positions.items():
    metadata = {
        'colmap_position': {
            'x': position[0],
            'y': position[1],
            'z': position[2]
        },
        'colmap_aligned': True
    }
    
    handler.embed_metadata(frame_path, metadata)
```

## Testing the Bridge

### Manual Test
```powershell
# 1. Extract and split frames
python run_app.py
# → Extract 10 frames
# → Split to 8 perspectives each (80 images total)

# 2. Run SphereAlignment manually
cd C:\Users\Everton-PC\Documents\APLICATIVOS\SphereAlignment
.\SphereAlignment.exe --input "path\to\perspectives" --output "colmap_out"

# 3. Verify output
ls colmap_out\sparse\0\images.txt
# Should see camera positions

# 4. Test our bridge
python -c "
from pathlib import Path
from PREMIUM_MODULE_TEMPLATE_COLMAP import parse_colmap_images

positions = parse_colmap_images(Path('colmap_out/sparse/0/images.txt'))
print(f'Found {len(positions)} positions')
for name, pos in list(positions.items())[:3]:
    print(f'{name}: {pos}')
"
```

### Integration Test
```python
# Test full bridge workflow
def test_bridge():
    # 1. Call SphereAlignment
    result = colmap_integrator.run_alignment(
        perspectives_dir=Path("perspectives"),
        output_dir=Path("colmap_output")
    )
    
    assert result['success'] == True
    assert result['num_aligned'] > 0
    
    # 2. Extract positions
    positions = result['positions']
    assert 'frame_001_cam0.jpg' in positions
    
    # 3. Map to frames
    frame_positions = colmap_integrator.map_positions_to_frames(
        positions,
        Path("perspectives"),
        Path("frames")
    )
    
    assert 'frame_001.jpg' in frame_positions
    
    # 4. Embed metadata
    colmap_integrator.propagate_positions(
        frame_positions,
        Path("frames")
    )
    
    # 5. Verify embedding
    metadata = handler.read_embedded_metadata(Path("frames/frame_001.jpg"))
    assert 'colmap_position' in metadata
    assert 'x' in metadata['colmap_position']
```

## Future: Built-in COLMAP

When building COLMAP directly into our app:

```python
# Instead of subprocess.run(SphereAlignment.exe)
from src.colmap import ColmapRunner  # Our own implementation

runner = ColmapRunner()
positions = runner.extract_and_match(perspectives_dir)
# No external dependency!
```

**Benefits:**
- Single executable
- No separate installation
- Better error handling
- Custom parameters
- Faster (no IPC overhead)

---

**Current Status:** Bridge architecture ready for implementation!
**Next:** Create premium module with bridge code → Test with SphereAlignment
