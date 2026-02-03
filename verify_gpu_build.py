"""Test GPU compatibility of the built executable."""
import subprocess
import os
import sys

def test_gpu_build():
    """Test if the Full GPU build has proper PyTorch CUDA support."""
    dist_path = r"C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit\dist\360ToolkitGS"
    exe_path = os.path.join(dist_path, "360ToolkitGS.exe")
    
    # Check if torch_cuda.dll exists
    torch_cuda = os.path.join(dist_path, "_internal", "torch_cuda.dll")
    torch_cpu = os.path.join(dist_path, "_internal", "torch_cpu.dll")
    
    print("=" * 60)
    print("360ToolkitGS Full GPU Build Verification")
    print("=" * 60)
    
    # Check critical files
    checks = [
        ("360ToolkitGS.exe", exe_path),
        ("torch_cuda.dll", torch_cuda),
        ("torch_cpu.dll", torch_cpu),
        ("MediaSDK.dll", os.path.join(dist_path, "_internal", "MediaSDK.dll")),
        ("onnxruntime.dll", os.path.join(dist_path, "_internal", "onnxruntime", "capi", "onnxruntime.dll")),
        ("onnxruntime_providers_cuda.dll", os.path.join(dist_path, "_internal", "onnxruntime", "capi", "onnxruntime_providers_cuda.dll")),
    ]
    
    all_ok = True
    for name, path in checks:
        exists = os.path.exists(path)
        size = os.path.getsize(path) / (1024*1024) if exists else 0
        status = "✓" if exists else "✗"
        if exists:
            print(f"  {status} {name}: {size:.1f} MB")
        else:
            print(f"  {status} {name}: MISSING")
            all_ok = False
    
    print()
    
    # Check CUDA DLLs
    cuda_dlls = [
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
        "cudart64_12.dll",
    ]
    
    print("CUDA Libraries:")
    for dll in cuda_dlls:
        path = os.path.join(dist_path, "_internal", dll)
        exists = os.path.exists(path)
        size = os.path.getsize(path) / (1024*1024) if exists else 0
        status = "✓" if exists else "?"
        if exists:
            print(f"  {status} {dll}: {size:.1f} MB")
        else:
            # Try alternate name
            alt_name = dll.replace("64_", "64_1")
            alt_path = os.path.join(dist_path, "_internal", alt_name)
            if os.path.exists(alt_path):
                size = os.path.getsize(alt_path) / (1024*1024)
                print(f"  ✓ {alt_name}: {size:.1f} MB")
            else:
                print(f"  ? {dll}: Not found (may be bundled differently)")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("Build verification: PASSED")
        print("All critical GPU components are present.")
    else:
        print("Build verification: FAILED")
        print("Some critical components are missing!")
    
    print("=" * 60)
    return all_ok

if __name__ == "__main__":
    success = test_gpu_build()
    sys.exit(0 if success else 1)
