# Performance Optimization Plan for 360toolkit

## Current Performance
- **Total time**: 4 hours for 1025 frames → 8,200 perspectives → 2,463 masks
- **Bottleneck**: Stage 2 (Perspective Splitting) = 2h 42min (66% of total time)
- **Current speed**: 1.19 seconds per output image

## Target Performance
- **Goal**: Reduce Stage 2 time by 60-70%
- **Target speed**: 0.4-0.5 seconds per output image
- **Expected total time**: ~2 hours (down from 4 hours)

---

## Critical Issues

### 1. **No Batch Processing** ❌
**Current**: Processes ONE frame at a time, then 8 cameras sequentially
```python
for frame in frames:  # Sequential
    load_frame()
    for camera in cameras:  # Sequential
        transform()
```

**Solution**: Process multiple frames simultaneously on GPU
```python
# Batch process 8-16 frames at once
batch = frames[i:i+batch_size]
all_tensors = load_batch_to_gpu(batch)  # Load 8 frames
for camera in cameras:
    # Process all 8 frames in parallel on GPU (batch operation)
    results = batch_transform(all_tensors, camera)  # GPU parallel
    save_batch(results)
```

**Expected speedup**: 5-8x faster (GPU parallelization)

### 2. **Excessive CPU↔GPU Transfers** ❌
**Current**: 8,200 individual transfers (load → process → save)

**Solution**: 
- Keep frames on GPU during all 8 camera transforms
- Only transfer once per frame (not per camera)
- Use pinned memory for faster transfers

**Expected speedup**: 20-30% faster

### 3. **No CUDA Streams** ❌
**Current**: All operations sequential on default stream

**Solution**: Use multiple CUDA streams to overlap:
- Stream 1: Load next batch
- Stream 2: Process current batch
- Stream 3: Save previous batch

**Expected speedup**: 15-25% faster

### 4. **Inefficient Image Saving** ❌
**Current**: Save each image immediately (blocking)

**Solution**: Queue images and save asynchronously
- Use ThreadPoolExecutor for I/O operations
- GPU continues processing while CPU saves

**Expected speedup**: 10-15% faster

---

## Implementation Priority

### **Phase 1: Batch Processing (HIGHEST IMPACT)** 🔥
- Modify `batch_orchestrator.py` to batch frames
- Update `TorchE2PTransform` to accept batched tensors
- Implement batch size auto-tuning based on VRAM

**Estimated time**: 2-3 hours coding
**Expected speedup**: 5-8x
**Risk**: Medium (requires careful memory management)

### **Phase 2: Optimize CPU↔GPU Transfers**
- Keep tensors on GPU between camera transforms
- Use pinned memory for async transfers
- Pre-allocate output buffers

**Estimated time**: 1-2 hours
**Expected speedup**: 1.2-1.3x additional
**Risk**: Low

### **Phase 3: CUDA Streams**
- Implement 3-stream pipeline
- Overlap data movement with computation

**Estimated time**: 2-3 hours
**Expected speedup**: 1.15-1.25x additional
**Risk**: Medium (synchronization complexity)

### **Phase 4: Async I/O**
- ThreadPoolExecutor for image saving
- Queue-based architecture

**Estimated time**: 1 hour
**Expected speedup**: 1.1-1.15x additional
**Risk**: Low

---

## Code Changes Required

### File: `src/transforms/e2p_transform.py`

**Add batch processing method:**
```python
def batch_equirect_to_pinhole(self, equirect_batch, yaw, pitch, roll, h_fov, v_fov, output_width, output_height):
    """
    Process multiple frames at once for the same camera angle.
    
    Args:
        equirect_batch: Tensor (N, 3, H, W) - multiple frames
        Other params: same camera orientation for all frames
        
    Returns:
        Batch output (N, 3, H_out, W_out)
    """
    # Generate grid once (cached)
    cache_key = (yaw, pitch, roll, h_fov, v_fov, output_width, output_height)
    if cache_key not in self.cache:
        grid = self._generate_grid(...)
        self.cache[cache_key] = grid
    else:
        grid = self.cache[cache_key]
    
    # Expand grid for batch: (1, H, W, 2) -> (N, H, W, 2)
    batch_size = equirect_batch.shape[0]
    grid_batch = grid.expand(batch_size, -1, -1, -1)
    
    # Batch transform (all frames at once)
    return F.grid_sample(equirect_batch, grid_batch, mode='bilinear', padding_mode='border', align_corners=True)
```

### File: `src/pipeline/batch_orchestrator.py`

**Replace sequential loop with batch processing:**
```python
# Current (slow):
for frame_path in input_frames:
    img_tensor = load_one_frame()
    for camera in cameras:
        result = transform(img_tensor, camera)
        save(result)

# New (fast):
batch_size = 8  # Process 8 frames simultaneously
for i in range(0, len(input_frames), batch_size):
    batch_paths = input_frames[i:i+batch_size]
    
    # Load batch to GPU (parallel loading)
    batch_tensors = []
    for path in batch_paths:
        img = cv2.imread(path)
        tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        batch_tensors.append(tensor)
    
    batch = torch.stack(batch_tensors).to(device)  # (N, 3, H, W)
    
    # Process all cameras for this batch
    for camera in cameras:
        # Transform entire batch at once (GPU parallel)
        results_batch = transformer.batch_equirect_to_pinhole(
            batch, camera['yaw'], camera['pitch'], camera['roll'], ...
        )
        
        # Save batch (can be async)
        save_batch_async(results_batch, output_dir)
    
    del batch  # Free GPU memory
    torch.cuda.empty_cache()
```

---

## Memory Management

### VRAM Requirements
- **8K equirectangular frame**: ~75 MB per image
- **Output perspective (1920×1080)**: ~8 MB per image
- **Batch of 8 frames**: 8 × 75 MB = 600 MB input + overhead
- **Total VRAM needed**: ~1.5 GB for batch_size=8

**RTX 5070 Ti has 16GB VRAM** → Safe to use batch_size=8-16

### Auto-tuning Batch Size
```python
def get_optimal_batch_size(device):
    total_vram = torch.cuda.get_device_properties(device).total_memory
    free_vram = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    
    # Conservative: use 60% of free VRAM
    available = free_vram * 0.6
    
    # Each 8K frame ~75MB, output ~8MB, overhead ~50MB
    per_frame_memory = 75 * 1024 * 1024 + 8 * 1024 * 1024 + 50 * 1024 * 1024  # ~133 MB
    
    batch_size = int(available // per_frame_memory)
    return max(4, min(batch_size, 16))  # Clamp between 4-16
```

---

## Testing Plan

1. **Unit test**: Batch transform produces identical output to sequential
2. **Memory test**: Monitor VRAM usage with different batch sizes
3. **Performance test**: Benchmark with 100 frames
4. **Quality test**: Verify no visual degradation

---

## Expected Results

### Before Optimization
- Stage 2: 2h 42min (9,720 seconds) for 8,200 images
- Speed: 1.19 seconds/image

### After Phase 1 (Batch Processing)
- Stage 2: ~20-30 minutes
- Speed: 0.15-0.22 seconds/image
- **Speedup: 5-8x**

### After All Phases
- Stage 2: ~15-20 minutes
- Speed: 0.11-0.15 seconds/image
- **Total speedup: 8-10x**

### New Total Pipeline Time
- Stage 1: 22 min (unchanged - already optimal)
- Stage 2: **15-20 min** (down from 162 min)
- Stage 3: 30 min (already fast)
- **Total: ~1h 10min** (down from 4 hours)

---

## Risks & Mitigations

### Risk 1: VRAM Overflow
**Mitigation**: 
- Auto-detect available VRAM
- Dynamic batch size adjustment
- Fallback to smaller batches on OOM

### Risk 2: Quality Degradation
**Mitigation**:
- Bit-exact comparison with reference implementation
- Visual QA on sample outputs

### Risk 3: RTX 50-series Compatibility
**Mitigation**:
- Graceful fallback to CPU multiprocessing
- Already implemented in current code

---

## Next Steps

1. **Backup current code**
2. **Implement Phase 1** (batch processing)
3. **Test with 100 frames**
4. **Deploy if successful**
5. **Iterate on Phases 2-4** if more speed needed

---

**Estimated development time**: 4-6 hours
**Expected performance gain**: 8-10x faster Stage 2
**User benefit**: 4-hour pipeline → ~1 hour pipeline
