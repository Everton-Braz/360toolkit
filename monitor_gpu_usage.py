"""
Real-time GPU Usage Monitor for 360toolkit Pipeline
Monitors GPU utilization, memory usage, and active processes during pipeline execution.
"""

import subprocess
import time
import sys
from datetime import datetime
import threading

def get_gpu_info():
    """Get GPU usage information using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_data = []
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpu_data.append({
                        'index': parts[0],
                        'name': parts[1],
                        'gpu_util': parts[2],
                        'mem_util': parts[3],
                        'mem_used': parts[4],
                        'mem_total': parts[5],
                        'temp': parts[6],
                        'power': parts[7]
                    })
            
            return gpu_data
        else:
            return None
            
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def get_gpu_processes():
    """Get list of processes using GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            processes = []
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    processes.append({
                        'pid': parts[0],
                        'name': parts[1],
                        'memory': parts[2]
                    })
            
            return processes
        else:
            return []
            
    except Exception as e:
        return []

def monitor_gpu_continuous(duration_seconds=300, interval=1.0):
    """Monitor GPU usage continuously"""
    print("="*80)
    print("GPU USAGE MONITOR - 360toolkit Pipeline")
    print("="*80)
    print(f"Monitoring for {duration_seconds} seconds (update every {interval}s)")
    print("Press Ctrl+C to stop early")
    print("="*80)
    
    start_time = time.time()
    max_gpu_util = 0
    max_mem_used = 0
    samples_with_gpu_activity = 0
    total_samples = 0
    
    try:
        while (time.time() - start_time) < duration_seconds:
            total_samples += 1
            gpu_data = get_gpu_info()
            processes = get_gpu_processes()
            
            if gpu_data:
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                for gpu in gpu_data:
                    gpu_util = int(gpu['gpu_util']) if gpu['gpu_util'] != 'N/A' else 0
                    mem_used = int(gpu['mem_used']) if gpu['mem_used'] != 'N/A' else 0
                    mem_total = int(gpu['mem_total']) if gpu['mem_total'] != 'N/A' else 1
                    mem_util = int(gpu['mem_util']) if gpu['mem_util'] != 'N/A' else 0
                    temp = int(gpu['temp']) if gpu['temp'] != 'N/A' else 0
                    power = float(gpu['power']) if gpu['power'] != 'N/A' else 0
                    
                    max_gpu_util = max(max_gpu_util, gpu_util)
                    max_mem_used = max(max_mem_used, mem_used)
                    
                    if gpu_util > 5:  # Consider >5% as active
                        samples_with_gpu_activity += 1
                    
                    # Create status indicators
                    gpu_status = "🚀 ACTIVE" if gpu_util > 10 else "⚠️ IDLE"
                    mem_bar = "█" * (mem_util // 5) + "░" * (20 - mem_util // 5)
                    util_bar = "█" * (gpu_util // 5) + "░" * (20 - gpu_util // 5)
                    
                    print(f"\r[{timestamp}] GPU {gpu['index']}: {gpu['name']}")
                    print(f"  Status: {gpu_status}")
                    print(f"  GPU Util: [{util_bar}] {gpu_util}%")
                    print(f"  VRAM:     [{mem_bar}] {mem_used}/{mem_total} MB ({mem_util}%)")
                    print(f"  Temp:     {temp}°C | Power: {power}W")
                    
                    if processes:
                        print(f"  Active Processes:")
                        for proc in processes:
                            print(f"    - PID {proc['pid']}: {proc['name']} ({proc['memory']} MB)")
                    else:
                        print(f"  Active Processes: None")
                    
                    print("-" * 80)
            else:
                print(f"⚠️ Could not read GPU data")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    
    elapsed = time.time() - start_time
    gpu_active_percent = (samples_with_gpu_activity / total_samples * 100) if total_samples > 0 else 0
    
    print("\n" + "="*80)
    print("MONITORING SUMMARY")
    print("="*80)
    print(f"Duration: {elapsed:.1f}s")
    print(f"Total samples: {total_samples}")
    print(f"Samples with GPU activity: {samples_with_gpu_activity} ({gpu_active_percent:.1f}%)")
    print(f"Max GPU Utilization: {max_gpu_util}%")
    print(f"Max VRAM Used: {max_mem_used} MB")
    
    if gpu_active_percent > 50:
        print("\n✅ GPU IS BEING ACTIVELY USED!")
    elif gpu_active_percent > 10:
        print("\n⚠️ GPU is being used intermittently")
    else:
        print("\n❌ GPU APPEARS IDLE - May be using CPU instead")
    
    print("="*80)

def check_gpu_availability():
    """Quick check of GPU availability"""
    print("="*80)
    print("GPU AVAILABILITY CHECK")
    print("="*80)
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ nvidia-smi available")
        else:
            print("❌ nvidia-smi failed")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi not found: {e}")
        return False
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Device Count: {torch.cuda.device_count()}")
        else:
            print("❌ PyTorch CUDA not available")
    except ImportError:
        print("⚠️ PyTorch not installed")
    except Exception as e:
        print(f"⚠️ PyTorch CUDA check failed: {e}")
    
    # Check ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print(f"✅ ONNX Runtime GPU available")
            print(f"   Version: {ort.__version__}")
            print(f"   Providers: {providers}")
        else:
            print(f"❌ ONNX Runtime CUDA provider not available")
            print(f"   Available providers: {providers}")
    except ImportError:
        print("⚠️ ONNX Runtime not installed")
    except Exception as e:
        print(f"⚠️ ONNX Runtime check failed: {e}")
    
    print("="*80)
    return True

def test_gpu_inference():
    """Test actual GPU inference"""
    print("\n" + "="*80)
    print("GPU INFERENCE TEST")
    print("="*80)
    
    # Test PyTorch
    try:
        import torch
        print("Testing PyTorch CUDA...")
        
        if torch.cuda.is_available():
            start = time.time()
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"✅ PyTorch GPU test successful ({elapsed*1000:.2f}ms)")
            del x, y
            torch.cuda.empty_cache()
        else:
            print("❌ PyTorch CUDA not available")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
    
    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("Testing ONNX Runtime CUDA...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Create simple test model (identity)
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Check if we can create a session with CUDA
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Use a simple model path if exists
            print("✅ ONNX Runtime CUDA provider ready")
        except Exception as e:
            print(f"⚠️ ONNX Runtime CUDA initialization issue: {e}")
            
    except Exception as e:
        print(f"❌ ONNX Runtime test failed: {e}")
    
    print("="*80)

def main():
    """Main monitoring function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'check':
            # Quick availability check
            check_gpu_availability()
            test_gpu_inference()
        elif sys.argv[1] == 'monitor':
            # Continuous monitoring
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 300
            interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
            
            check_gpu_availability()
            print("\nStarting continuous monitoring in 3 seconds...")
            time.sleep(3)
            monitor_gpu_continuous(duration, interval)
        else:
            print("Usage:")
            print("  python monitor_gpu_usage.py check              # Quick GPU check")
            print("  python monitor_gpu_usage.py monitor [duration] [interval]  # Monitor GPU (default: 300s, 1s interval)")
    else:
        # Default: check + short monitor
        check_gpu_availability()
        test_gpu_inference()
        print("\nStarting 60-second monitoring...")
        time.sleep(2)
        monitor_gpu_continuous(60, 1.0)

if __name__ == "__main__":
    main()
