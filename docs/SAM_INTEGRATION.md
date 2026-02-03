# SAM (Segment Anything Model) Integration

## Overview

360ToolKit now supports **SAM ViT-B** as an alternative masking engine for Stage 3, providing superior segmentation quality compared to YOLO-based methods.

## Features

### SAM Advantages
- ✨ **Superior segmentation quality**: Better edge precision and boundary detection
- 🎯 **Better handling of overlapping objects**: Accurately separates complex scenes
- 🔍 **Precise mask boundaries**: Cleaner masks with fewer artifacts
- 🤖 **Automatic mask generation**: No manual prompts needed
- 🎨 **High-quality foreground/background separation**

### YOLO Advantages
- ⚡ **Faster inference**: 3-4x faster than SAM
- 🎯 **Class-specific detection**: Knows what objects are (person, backpack, etc.)
- 💾 **Smaller models**: 10-90 MB vs 375 MB for SAM
- 📦 **Easier deployment**: ONNX Runtime vs full PyTorch

## Installation

### 1. Install SAM Package

```bash
pip install segment-anything
```

### 2. Download SAM Checkpoint

**Automatic download** (recommended):
```bash
python -c "from src.masking.sam_masker import download_sam_checkpoint; download_sam_checkpoint('vit_b')"
```

**Manual download**:
- Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- Save to: `360toolkit/sam_vit_b_01ec64.pth` (375 MB)

### 3. Verify Installation

```bash
python test_sam_integration.py
```

## Usage

### GUI Interface

1. **Open Stage 3 Configuration** in 360ToolKit
2. **Select Masking Engine**: Choose "✨ SAM ViT-B - Best Quality"
3. **Configure categories**:
   - Persons (large masks >10% of image)
   - Personal Objects (medium masks 1-15%)
   - Animals (large masks >5%)
4. **Run pipeline** as normal

### Programmatic Usage

```python
from src.masking.sam_masker import SAMMasker
from pathlib import Path

# Initialize SAM masker
masker = SAMMasker(
    model_checkpoint='sam_vit_b_01ec64.pth',
    use_gpu=True,
    mask_dilation_pixels=15,
    points_per_side=32  # Sampling grid density (higher = more masks)
)

# Set enabled categories
masker.set_enabled_categories({
    'persons': True,
    'personal_objects': True,
    'animals': False
})

# Generate mask for single image
mask = masker.generate_mask(
    image_path=Path('input.jpg'),
    output_path=Path('output_mask.png')
)

# Batch processing
masker.batch_generate_masks(
    image_paths=[Path('img1.jpg'), Path('img2.jpg')],
    output_dir=Path('masks/'),
    progress_callback=lambda cur, tot, msg: print(f"{cur}/{tot}: {msg}")
)
```

## Configuration

### SAM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_checkpoint` | `sam_vit_b_01ec64.pth` | Path to SAM ViT-B checkpoint |
| `use_gpu` | `True` | Enable GPU acceleration (CUDA) |
| `mask_dilation_pixels` | `15` | Expand mask boundaries (pixels) |
| `points_per_side` | `32` | Sampling grid density (16-64) |
| `pred_iou_thresh` | `0.88` | IoU threshold for mask filtering |
| `stability_score_thresh` | `0.95` | Stability score threshold |

### Category Heuristics

Since SAM doesn't classify objects, we use heuristic filtering:

| Category | Area Ratio | Stability Score | Description |
|----------|-----------|-----------------|-------------|
| **Persons** | >10% | >0.90 | Large central masks |
| **Personal Objects** | 1-15% | >0.85 | Medium foreground masks |
| **Animals** | >5% | >0.88 | Large or medium masks |

## Performance Comparison

### Speed (RTX 4090)

| Engine | Model | Time/Image | GPU Memory |
|--------|-------|-----------|------------|
| **YOLO ONNX** | yolo26s | 0.4s | 2 GB |
| **YOLO PyTorch** | yolov8m | 0.7s | 3 GB |
| **SAM ViT-B** | vit_b | 1.5-3s | 4 GB |

### Quality (Subjective)

| Aspect | YOLO | SAM |
|--------|------|-----|
| **Edge precision** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Overlapping objects** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Small objects** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Class accuracy** | ⭐⭐⭐⭐⭐ | N/A |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## Recommendations

### Use SAM When:
- ✅ **Quality is critical** (final production masks)
- ✅ **Complex scenes** with overlapping objects
- ✅ **Precise edges needed** for photogrammetry
- ✅ **GPU available** (CUDA)
- ✅ **Processing time not critical**

### Use YOLO When:
- ✅ **Speed is important** (large datasets)
- ✅ **Class-specific filtering** needed (only persons, only backpacks)
- ✅ **Resource-constrained** (low VRAM)
- ✅ **Batch processing** (1000+ images)
- ✅ **Real-time requirements**

## Troubleshooting

### "SAM checkpoint not found"

**Solution**: Download checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### "CUDA out of memory"

**Solutions**:
1. Reduce `points_per_side` (32 → 16)
2. Process smaller images
3. Use CPU mode (`use_gpu=False`)
4. Close other GPU applications

### "No masks detected"

**Possible causes**:
- Empty scene (no foreground objects)
- Thresholds too strict
- Image too small/large

**Solutions**:
1. Lower `stability_score_thresh` (0.95 → 0.85)
2. Lower `pred_iou_thresh` (0.88 → 0.75)
3. Increase `points_per_side` (32 → 48)

### "SAM slower than expected"

**Optimizations**:
1. Ensure CUDA is available: `torch.cuda.is_available()`
2. Use GPU with 8+ GB VRAM
3. Reduce `points_per_side` (32 → 24)
4. Process in batches (not single images)

## Technical Details

### How SAM Works

1. **Automatic Mask Generation**:
   - Grid sampling: Places prompt points across image
   - Mask generation: Generates masks for each point
   - Quality filtering: Filters by IoU and stability scores
   - Deduplication: Removes overlapping low-quality masks

2. **Heuristic Category Assignment**:
   - Since SAM doesn't classify, we use:
     * Mask area ratio (% of image)
     * Spatial position (center vs edges)
     * Stability score (quality metric)

3. **Mask Combination**:
   - Start with white background (keep all)
   - Paint selected masks black (remove)
   - Apply dilation for boundary expansion

### SAM vs YOLO Architecture

| Aspect | YOLO | SAM |
|--------|------|-----|
| **Architecture** | CNN (detection + segmentation) | ViT (transformer encoder + mask decoder) |
| **Training** | Object-specific (COCO 80 classes) | Class-agnostic (SA-1B 1B masks) |
| **Output** | Class + box + mask | Mask + quality scores |
| **Inference** | Single forward pass | Grid-based sampling + multiple passes |
| **Strengths** | Speed, class accuracy | Edge quality, generalization |

## Examples

### Simple Usage

```python
# Quick test with default settings
from src.masking.sam_masker import SAMMasker

masker = SAMMasker('sam_vit_b_01ec64.pth')
mask = masker.generate_mask('photo.jpg', 'mask.png')
```

### Advanced Configuration

```python
# Fine-tune for high-quality results
masker = SAMMasker(
    model_checkpoint='sam_vit_b_01ec64.pth',
    use_gpu=True,
    mask_dilation_pixels=20,  # More boundary expansion
    points_per_side=48,  # Denser sampling (slower but better)
    pred_iou_thresh=0.85,  # Lower threshold = more masks
    stability_score_thresh=0.90  # Lower = keep more masks
)

# Only mask large objects (persons/animals)
masker.min_mask_area = 5000  # Minimum 5000 pixels
masker.max_mask_area = 500000  # Maximum 500k pixels
```

## Future Improvements

- [ ] Add SAM2 support (video segmentation)
- [ ] Implement SAM-L and SAM-H variants
- [ ] Add prompt-based refinement (click to add/remove)
- [ ] Integrate class classifier (CLIP) for category filtering
- [ ] Optimize batch processing with parallel GPU inference
- [ ] Add mobile-friendly SAM variant (MobileSAM)

## References

- **SAM Paper**: https://arxiv.org/abs/2304.02643
- **Official Repo**: https://github.com/facebookresearch/segment-anything
- **Model Card**: https://github.com/facebookresearch/segment-anything/blob/main/MODEL_CARD.md
- **YOLO Comparison**: https://blog.roboflow.com/sam-vs-yolo/

## License

SAM is licensed under Apache 2.0. See: https://github.com/facebookresearch/segment-anything/blob/main/LICENSE
