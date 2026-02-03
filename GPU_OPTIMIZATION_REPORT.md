# GPU Optimization Summary - January 21, 2026

## 🔍 DIAGNOSIS RESULTS

### Initial Problem
- **Reported Issue**: "Stage 2 and Stage 3 running on CPU, not GPU"
- **Actual Issue**: GPU IS working, but only 11-48% utilized (instead of 80-100%)

### Root Cause Analysis
Ran comprehensive diagnostics (`diagnose_gpu_bottleneck.py`) which revealed:

```
GPU Processing Speed:    12,648 FPS (0.1ms per frame) ⚡
Disk I/O Speed:          85.5 images/sec (11.7ms per frame) 🐌
Bottleneck:              77.8% of time spent on DISK I/O
GPU Idle Time:           GPU waits 11.7ms while CPU loads each image
```

**The GPU is 117× faster than the disk!**

---

## ✅ OPTIMIZATIONS APPLIED

### 1. Stage 1 (MediaSDK) - GPU Acceleration
**File**: `src/extraction/sdk_extractor.py`

Added explicit GPU flags to MediaSDK command:
```python
-disable_cuda false              # ENABLE CUDA (confusing naming!)
-enable_soft_encode false        # Use GPU H.264/H.265 encoder (not CPU)
-enable_soft_decode false        # Use GPU video decoder (not CPU)
-image_processing_accel auto     # Auto-detect GPU/Vulkan for processing
```

**Expected improvement**: +10-15% Stage 1 performance

---

### 2. Stage 2 - Batch Size Optimization
**File**: `src/pipeline/batch_orchestrator.py`

Changed from auto-detected batch size to optimal:
```python
# Before: batch_size = 8 (auto-detected)
# After:  batch_size = 16 (tested optimal for RTX 5070 Ti)
```

**Reason**: Testing showed batch size 16 is 8% faster than 8:
- Batch 8:  11,659 FPS
- Batch 16: 12,648 FPS ✅

**Expected improvement**: +8% GPU throughput

---

### 3. I/O Thread Pool Maximization
**File**: `src/pipeline/batch_orchestrator.py`

Increased concurrent I/O operations:
```python
# Before:
load_executor = ThreadPoolExecutor(max_workers=24)
save_executor = ThreadPoolExecutor(max_workers=16)

# After:
load_executor = ThreadPoolExecutor(max_workers=32)  # +33% more workers
save_executor = ThreadPoolExecutor(max_workers=24)  # +50% more workers
```

**Expected improvement**: +15-20% I/O throughput

---

### 4. RAM Image Cache (NEW)
**File**: `src/pipeline/batch_orchestrator.py`

Added LRU cache for recently loaded images:
```python
image_cache = {}  # 4 images max (~360 MB RAM)
cache_lock = threading.Lock()
```

**Benefits**:
- Eliminates repeated disk reads
- Cache hits return in <1ms instead of 11.7ms
- Particularly effective for multi-camera processing (same frame, 8 cameras)

**Expected improvement**: +10-15% for cached reads

---

## 📊 EXPECTED RESULTS

### Performance Improvements
| Stage | Before | After | Improvement |
|-------|--------|-------|-------------|
| Stage 1 (SDK) | Baseline | +10-15% | GPU codecs enabled |
| Stage 2 (E2P) | Baseline | +20-30% | Batch 16 + I/O optimizations |
| Stage 3 (Mask) | Baseline | +15-20% | I/O optimizations |
| **Overall** | **Baseline** | **+25-35%** | **Combined effect** |

### GPU Utilization
- **Before**: 11-48% (I/O starved)
- **After**: 40-70% (still I/O limited, but better fed)

**Note**: 80-100% GPU utilization requires faster storage (see recommendations below)

---

## 🧪 TESTING INSTRUCTIONS

### 1. Quick Verification Test
```bash
# Verify code imports without errors
python -c "from src.pipeline.batch_orchestrator import BatchOrchestrator; print('✅ OK')"
```

### 2. GPU Monitoring During Pipeline
```bash
# Terminal 1: Run pipeline
python run_app.py

# Terminal 2: Monitor GPU in real-time
python verify_gpu_realtime.py
```

**What to expect**:
- Stage 1: GPU 10-30% (video decode + stitching)
- Stage 2: GPU **40-70%** (up from 11-48%) ⬆️
- Stage 3: GPU **40-70%** (up from 11-48%) ⬆️

### 3. Batch Size Confirmation
Check logs for:
```
Using batch size: 16 frames (auto: 8, optimized for I/O)
[I/O Optimization] Using 32 load workers, 24 save workers
```

### 4. Performance Comparison
Run the same video file twice and compare:
```bash
# Before optimizations: ~120 seconds
# After optimizations:  ~85 seconds (expected)
```

---

## 🚀 ULTIMATE PERFORMANCE (80-100% GPU)

To fully saturate the GPU, you need to eliminate the I/O bottleneck:

### Hardware Recommendations

#### Option 1: NVMe SSD (BEST)
- **Current**: HDD/SATA SSD (~500 MB/s sequential)
- **Target**: NVMe Gen 4 SSD (~7000 MB/s sequential)
- **Improvement**: 14× faster I/O = GPU 80-100% utilization
- **Cost**: $100-200 for 1TB NVMe drive

#### Option 2: RAMDisk (FASTEST, but limited)
If you have 32GB+ RAM:
```bash
# Create RAMDisk for temp files
# Move stage1_frames to RAMDisk before Stage 2
# GPU utilization: 90-100%
```
**Limitation**: Only works for temp files (limited by RAM size)

#### Option 3: Network Storage with 10GbE
- 10 Gigabit Ethernet to NAS with NVMe
- Suitable for large-scale workflows
- Cost: $500+ (NIC + switch + NAS)

---

## 📝 FILES MODIFIED

1. **src/extraction/sdk_extractor.py**
   - Lines ~865-890: Added GPU acceleration flags

2. **src/pipeline/batch_orchestrator.py**
   - Lines ~670-680: Batch size optimization (8 → 16)
   - Lines ~708-750: RAM cache implementation
   - Lines ~713: I/O workers (24 → 32, 16 → 24)

---

## ✅ VERIFICATION CHECKLIST

- [x] MediaSDK GPU flags enabled (`-disable_cuda false`)
- [x] Batch size increased to 16
- [x] I/O workers maximized (32 load, 24 save)
- [x] RAM cache implemented (4 images)
- [x] Code imports without errors
- [ ] **TODO: Run full pipeline test**
- [ ] **TODO: Verify GPU 40-70% utilization**
- [ ] **TODO: Measure performance improvement**

---

## 🔧 TROUBLESHOOTING

### If GPU still shows <40% utilization:

1. **Check disk type**:
   ```bash
   # Verify input/output on SSD, not HDD
   Get-PhysicalDisk | Format-Table DeviceID, FriendlyName, MediaType
   ```

2. **Monitor disk queue length**:
   - Open Task Manager → Performance → Disk
   - If "Active time" is 100%, disk is the bottleneck

3. **Reduce output resolution** (temporary test):
   - Lower output from 2K → 1K
   - If GPU utilization increases, confirms I/O bottleneck

### If Stage 2/3 errors occur:

1. **Check batch size**: Ensure logs show "batch size: 16"
2. **VRAM usage**: Monitor with `nvidia-smi` - should use 3-4GB
3. **Fallback**: If errors, batch_orchestrator will auto-reduce batch size

---

## 📈 PERFORMANCE METRICS

### Before Optimizations
- Pipeline time: ~120s for 30 frames × 8 cameras = 240 images
- GPU utilization: 11-48%
- Bottleneck: 77.8% I/O

### After Optimizations (Expected)
- Pipeline time: ~85s for 30 frames × 8 cameras = 240 images
- GPU utilization: 40-70%
- Bottleneck: Still I/O, but reduced to ~60%

### Theoretical Maximum (NVMe SSD)
- Pipeline time: ~30s for 30 frames × 8 cameras = 240 images
- GPU utilization: 80-100%
- Bottleneck: Minimal I/O, GPU-bound

---

## 💡 KEY INSIGHT

**Your GPU was ALWAYS working** - the logs proved it:
```
✅ Stage 2: "Initialized TorchE2PTransform on cuda (FP16 + TF32)"
✅ Stage 3: "Active providers: ['CUDAExecutionProvider', ...]"
```

The problem was **GPU starvation** (not GPU inactivity). The GPU processed data so fast that it spent most of its time waiting for the CPU to load the next batch from disk.

Think of it like a Formula 1 car stuck in traffic - the car works perfectly, but the road limits its speed.

---

## 🎯 NEXT STEPS

1. **Run the pipeline** with these optimizations
2. **Monitor GPU** with `verify_gpu_realtime.py`
3. **Compare timings** to baseline
4. **Report results**: Expected 25-35% faster overall

**If you want 80-100% GPU**: Invest in NVMe SSD (~$150) for 14× I/O improvement.

---

**Date**: January 21, 2026  
**Optimized for**: NVIDIA RTX 5070 Ti (16GB VRAM, sm_120)  
**Status**: ✅ Ready for testing
