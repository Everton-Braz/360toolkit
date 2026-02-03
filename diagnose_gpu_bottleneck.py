"""
GPU Bottleneck Diagnostic Tool

Analyzes why GPU utilization is low despite CUDA being enabled.
Common issues:
1. Batch size too small - GPU underutilized
2. CPU I/O bottleneck - GPU waiting for data
3. Excessive CPU↔GPU transfers
4. Small tensor operations that don't saturate GPU cores
"""

import time
import sys
from pathlib import Path

def test_batch_sizes():
    """Test different batch sizes to find GPU saturation point."""
    print("\n" + "="*80)
    print("BATCH SIZE PERFORMANCE TEST")
    print("="*80)
    
    try:
        import torch
        import numpy as np
        from src.transforms.e2p_transform import TorchE2PTransform
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return
        
        device = torch.device('cuda')
        print(f"Testing on: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        
        # Simulate 8K equirectangular → 8 cameras × 2K perspective
        input_size = (7680, 3840, 3)  # 8K equirectangular
        output_size = (2048, 1024)    # 2K perspective per camera
        num_cameras = 8
        
        transformer = TorchE2PTransform()
        
        print(f"Input: {input_size[0]}×{input_size[1]} equirectangular")
        print(f"Output: {num_cameras} cameras × {output_size[0]}×{output_size[1]} perspective\n")
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = []
        
        for batch_size in batch_sizes:
            try:
                print(f"Testing batch size {batch_size}...", end='', flush=True)
                
                # Create fake batch
                batch = torch.randn(
                    batch_size, 3, input_size[1], input_size[0],
                    device=device, dtype=torch.float16 if transformer.use_fp16 else torch.float32
                )
                
                # Warm-up
                for cam_idx in range(2):  # Just 2 cameras for speed
                    _ = transformer.batch_equirect_to_pinhole(
                        batch, yaw=cam_idx*45, pitch=0, roll=0,
                        h_fov=110, v_fov=None,
                        output_width=output_size[0], output_height=output_size[1]
                    )
                torch.cuda.synchronize()
                
                # Timed run (all 8 cameras)
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                for cam_idx in range(num_cameras):
                    output = transformer.batch_equirect_to_pinhole(
                        batch, yaw=cam_idx*45, pitch=0, roll=0,
                        h_fov=110, v_fov=None,
                        output_width=output_size[0], output_height=output_size[1]
                    )
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                # Metrics
                frames_processed = batch_size * num_cameras
                fps = frames_processed / elapsed
                ms_per_frame = (elapsed / frames_processed) * 1000
                
                vram_used = torch.cuda.max_memory_allocated(device) / 1024**3
                
                print(f" ✅ {fps:.1f} FPS ({ms_per_frame:.1f}ms/frame) | VRAM: {vram_used:.2f} GB")
                
                results.append({
                    'batch_size': batch_size,
                    'fps': fps,
                    'ms_per_frame': ms_per_frame,
                    'vram_gb': vram_used
                })
                
                torch.cuda.reset_peak_memory_stats()
                del batch, output
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f" ❌ OUT OF MEMORY")
                    break
                else:
                    raise
        
        # Analysis
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        
        if not results:
            print("❌ No successful tests")
            return
        
        best = max(results, key=lambda x: x['fps'])
        print(f"\n✅ OPTIMAL BATCH SIZE: {best['batch_size']}")
        print(f"   Performance: {best['fps']:.1f} FPS ({best['ms_per_frame']:.1f}ms/frame)")
        print(f"   VRAM usage: {best['vram_gb']:.2f} GB")
        
        # Current vs optimal
        current_batch = 8  # From auto-detection in logs
        current_result = next((r for r in results if r['batch_size'] == current_batch), None)
        
        if current_result and current_result['fps'] < best['fps'] * 0.9:
            speedup = best['fps'] / current_result['fps']
            print(f"\n⚠️  CURRENT BATCH SIZE ({current_batch}) IS SUBOPTIMAL")
            print(f"   Switching to batch size {best['batch_size']} would give {speedup:.1f}× speedup!")
        else:
            print(f"\n✅ Current batch size ({current_batch}) is near-optimal")
        
        # GPU utilization estimate
        print("\n" + "-"*80)
        print("GPU UTILIZATION ESTIMATE")
        print("-"*80)
        
        # Theoretical peak (RTX 5070 Ti: ~40 TFLOPS FP16)
        # Each pixel requires ~100 FLOPs for spherical projection
        pixels_per_frame = output_size[0] * output_size[1] * num_cameras
        flops_per_frame = pixels_per_frame * 100
        peak_fps_theoretical = 40_000_000_000_000 / flops_per_frame  # Very rough estimate
        
        actual_fps = best['fps']
        efficiency = (actual_fps / peak_fps_theoretical) * 100 if peak_fps_theoretical > 0 else 0
        
        print(f"Theoretical peak: ~{peak_fps_theoretical:.0f} FPS (rough estimate)")
        print(f"Actual achieved: {actual_fps:.1f} FPS")
        print(f"Estimated efficiency: {min(efficiency, 100):.1f}%")
        
        if efficiency < 50:
            print("\n⚠️  LOW GPU EFFICIENCY - Likely bottlenecks:")
            print("   1. CPU-bound data loading (I/O)")
            print("   2. Excessive CPU↔GPU transfers")
            print("   3. Small batch size limiting parallelism")
            print("   4. Memory bandwidth bottleneck (not compute)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_io_bottleneck():
    """Test if disk I/O is the bottleneck."""
    print("\n" + "="*80)
    print("I/O BOTTLENECK TEST")
    print("="*80)
    
    try:
        import cv2
        import numpy as np
        from pathlib import Path
        
        # Find sample equirectangular image
        test_dirs = [
            Path("C:/Users/Everton-PC/Documents/ARQUIVOS_TESTE"),
            Path("test_export"),
        ]
        
        sample_image = None
        for test_dir in test_dirs:
            if test_dir.exists():
                images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
                if images:
                    sample_image = images[0]
                    break
        
        if not sample_image:
            print("❌ No sample images found for I/O test")
            return
        
        print(f"Testing with: {sample_image.name}")
        print(f"File size: {sample_image.stat().st_size / 1024**2:.2f} MB\n")
        
        # Test 1: Sequential read (like current pipeline)
        print("Test 1: Sequential image loading (current method)")
        num_reads = 20
        start = time.perf_counter()
        for _ in range(num_reads):
            img = cv2.imread(str(sample_image))
        sequential_time = time.perf_counter() - start
        sequential_fps = num_reads / sequential_time
        print(f"  Sequential: {sequential_fps:.1f} images/sec ({sequential_time/num_reads*1000:.1f}ms each)\n")
        
        # Test 2: Pre-loaded (eliminate I/O)
        print("Test 2: Pre-loaded from memory (no I/O)")
        img_cached = cv2.imread(str(sample_image))
        start = time.perf_counter()
        for _ in range(num_reads):
            img = img_cached.copy()
        cached_time = time.perf_counter() - start
        cached_fps = num_reads / cached_time
        print(f"  Cached: {cached_fps:.1f} images/sec ({cached_time/num_reads*1000:.1f}ms each)\n")
        
        # Analysis
        print("="*80)
        print("ANALYSIS")
        print("="*80)
        
        io_overhead = ((sequential_time - cached_time) / sequential_time) * 100
        print(f"I/O overhead: {io_overhead:.1f}% of total time")
        
        if io_overhead > 30:
            print("\n⚠️  HIGH I/O OVERHEAD - Recommendations:")
            print("   1. Use SSD instead of HDD for input/output")
            print("   2. Increase batch size to amortize I/O cost")
            print("   3. Use multi-threaded image loading (already implemented)")
            print("   4. Consider caching hot images in RAM")
        else:
            print("\n✅ I/O overhead is acceptable")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_pipeline_logs():
    """Analyze recent pipeline logs for bottlenecks."""
    print("\n" + "="*80)
    print("PIPELINE LOG ANALYSIS")
    print("="*80)
    
    print("\nLooking at your logs:")
    print("  - Stage 2: 'Auto-detected optimal batch size: 8'")
    print("  - Stage 3: Processing 480 images")
    print("\nYour monitoring showed GPU usage: 11-48%")
    print("This indicates:")
    print("\n1. ✅ GPU is ACTIVE (not 0%)")
    print("2. ⚠️  GPU is UNDERUTILIZED (should be 80-100%)")
    print("\nLikely causes:")
    print("  A. Batch size 8 is too small for RTX 5070 Ti")
    print("  B. CPU can't feed data fast enough (I/O bottleneck)")
    print("  C. Frequent GPU synchronization between batches")
    print("\nRun 'python diagnose_gpu_bottleneck.py test' for detailed analysis")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python diagnose_gpu_bottleneck.py test      - Run all diagnostic tests")
        print("  python diagnose_gpu_bottleneck.py batch     - Test batch sizes only")
        print("  python diagnose_gpu_bottleneck.py io        - Test I/O bottleneck")
        print("  python diagnose_gpu_bottleneck.py analyze   - Analyze logs")
        analyze_pipeline_logs()
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == 'test':
        test_batch_sizes()
        test_io_bottleneck()
        analyze_pipeline_logs()
    elif command == 'batch':
        test_batch_sizes()
    elif command == 'io':
        test_io_bottleneck()
    elif command == 'analyze':
        analyze_pipeline_logs()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main()
