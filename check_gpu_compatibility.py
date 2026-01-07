"""
GPU Compatibility Checker for 360toolkit
Diagnoses PyTorch CUDA compatibility issues and recommends fixes
"""

import sys
import os

print("=" * 70)
print("360toolkit - GPU Compatibility Checker")
print("=" * 70)
print()

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")
print()

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    
    if cuda_available:
        cuda_version = torch.version.cuda
        print(f"  CUDA version: {cuda_version}")
        
        device_count = torch.cuda.device_count()
        print(f"  GPU devices found: {device_count}")
        
        if device_count > 0:
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"\n  GPU {i}: {device_name}")
                
                try:
                    compute_capability = torch.cuda.get_device_capability(i)
                    compute_cap_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
                    print(f"    Compute Capability: {compute_cap_str}")
                except Exception as e:
                    print(f"    Compute Capability: Unknown ({e})")
                
                # Test GPU functionality
                print(f"    Testing GPU {i}...")
                try:
                    test_tensor = torch.zeros(100, 100, device=f'cuda:{i}')
                    result = test_tensor + 1
                    mean_val = result.mean().item()
                    del test_tensor, result
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    print(f"    ✓ GPU {i} fully functional (test passed)")
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    print(f"    ✗ GPU {i} FAILED: {e}")
                    
                    if "no kernel image" in error_msg or "sm_" in error_msg:
                        print()
                        print("    " + "=" * 60)
                        print("    ISSUE DETECTED: GPU Architecture Not Supported")
                        print("    " + "=" * 60)
                        
                        if "rtx 50" in device_name.lower() or compute_cap_str == "sm_120":
                            print("    Your RTX 50-series GPU requires PyTorch nightly build.")
                            print()
                            print("    FIX: Run this command:")
                            print("    pip uninstall torch torchvision")
                            print("    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124")
                            print()
                            print("    Or use: update_pytorch_for_rtx50.bat")
                        else:
                            print("    Your GPU architecture is not supported by this PyTorch build.")
                            print()
                            print("    FIX: Update PyTorch:")
                            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
                        
                        print("    " + "=" * 60)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    ✗ GPU {i} test error: {e}")
    else:
        print("\n  ⚠ CUDA not available")
        print("  Possible reasons:")
        print("  - No NVIDIA GPU installed")
        print("  - NVIDIA drivers not installed")
        print("  - PyTorch CPU-only version installed")
        print()
        print("  If you have an NVIDIA GPU:")
        print("  1. Install/update NVIDIA drivers from nvidia.com")
        print("  2. Reinstall PyTorch with CUDA:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        
except ImportError:
    print("✗ PyTorch NOT installed")
    print()
    print("  FIX: Install PyTorch with CUDA support:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

print()

# Check Ultralytics
try:
    import ultralytics
    print(f"✓ Ultralytics installed: {ultralytics.__version__}")
except ImportError:
    print("✗ Ultralytics NOT installed")
    print("  FIX: pip install ultralytics")

print()

# Check OpenCV
try:
    import cv2
    print(f"✓ OpenCV installed: {cv2.__version__}")
except ImportError:
    print("✗ OpenCV NOT installed")
    print("  FIX: pip install opencv-python")

print()
print("=" * 70)
print("Compatibility Check Complete")
print("=" * 70)
print()

# Summary
print("SUMMARY:")
print()
try:
    import torch
    if torch.cuda.is_available():
        try:
            test = torch.zeros(1, device='cuda')
            _ = test + 1
            torch.cuda.synchronize()
            print("✓ GPU acceleration FULLY FUNCTIONAL")
            print("  360toolkit will use GPU for masking (Stage 3)")
        except:
            print("⚠ GPU detected but NOT COMPATIBLE with current PyTorch")
            print("  360toolkit will fall back to CPU (slower)")
            print("  Run update_pytorch_for_rtx50.bat to fix RTX 50-series issues")
    else:
        print("⚠ No GPU available - will use CPU")
        print("  Performance will be slower but functional")
except:
    print("✗ PyTorch not installed - masking will not work")

print()
input("Press Enter to exit...")
