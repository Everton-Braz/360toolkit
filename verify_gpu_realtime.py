"""
Real-time GPU verification during pipeline execution.
Run this in a separate terminal WHILE the pipeline is running to see GPU utilization spikes.
"""

import subprocess
import time
import sys
from datetime import datetime

def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            gpu_util = int(parts[0].strip())
            mem_used = int(parts[1].strip())
            mem_total = int(parts[2].strip())
            temp = int(parts[3].strip())
            return gpu_util, mem_used, mem_total, temp
    except Exception as e:
        return None, None, None, None
    return None, None, None, None

def monitor_continuous():
    """Monitor GPU in real-time with visual indicators."""
    print("=" * 80)
    print("REAL-TIME GPU MONITORING FOR 360TOOLKIT PIPELINE")
    print("=" * 80)
    print("\nRun your pipeline in another terminal now!")
    print("This will show GPU utilization spikes during Stage 2 and Stage 3.\n")
    print("Stage 1 (SDK Video Decode) = High CPU, some GPU (decoder)")
    print("Stage 2 (E2P Transform) = HIGH GPU (80-100% during batches)")
    print("Stage 3 (YOLO26 Masking) = HIGH GPU (80-100% during inference)")
    print("\nPress Ctrl+C to stop monitoring.\n")
    print("=" * 80)
    
    max_util = 0
    max_vram = 0
    samples = 0
    high_util_samples = 0  # Count samples with >30% GPU usage
    
    try:
        while True:
            gpu_util, mem_used, mem_total, temp = get_gpu_utilization()
            
            if gpu_util is not None:
                samples += 1
                max_util = max(max_util, gpu_util)
                max_vram = max(max_vram, mem_used)
                
                if gpu_util > 30:
                    high_util_samples += 1
                
                # Visual bar for GPU utilization
                bar_length = 40
                filled = int(bar_length * gpu_util / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Status indicator
                if gpu_util > 80:
                    status = "🚀 STAGE 2/3 ACTIVE"
                    color = "\033[92m"  # Green
                elif gpu_util > 30:
                    status = "⚡ GPU WORKING"
                    color = "\033[93m"  # Yellow
                else:
                    status = "⚠️  IDLE/STAGE 1"
                    color = "\033[90m"  # Gray
                
                reset = "\033[0m"
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\r{color}[{timestamp}] GPU: {gpu_util:3d}% {bar} | "
                      f"VRAM: {mem_used:5d}/{mem_total:5d}MB | Temp: {temp:2d}°C | {status}{reset}", 
                      end='', flush=True)
            else:
                print("\r[ERROR] Cannot read GPU metrics. Is nvidia-smi available?", end='', flush=True)
            
            time.sleep(0.5)  # 2Hz update rate
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("MONITORING SUMMARY")
        print("=" * 80)
        print(f"Total samples: {samples}")
        print(f"Max GPU utilization: {max_util}%")
        print(f"Max VRAM used: {max_vram} MB")
        print(f"High-utilization samples (>30%): {high_util_samples}/{samples} ({high_util_samples/samples*100:.1f}%)")
        
        if max_util > 80:
            print("\n✅ GPU WAS HEAVILY UTILIZED - Stage 2/3 used GPU successfully!")
        elif max_util > 30:
            print("\n⚠️  MODERATE GPU USAGE - GPU was active but may not be fully utilized")
        else:
            print("\n❌ LOW GPU USAGE - GPU may not be active or pipeline not running")
        
        print("\nIf you saw spikes during Stage 2 and Stage 3, GPU acceleration is working!")
        print("=" * 80)

if __name__ == '__main__':
    monitor_continuous()
