"""
Simple Build Verification Script
Tests that can run without full environment setup
"""

import sys
import os
from pathlib import Path

print("="*70)
print("360ToolkitGS - Simplified Version Verification")
print("="*70)
print()

# Track results
results = []

# Test 1: Check file modifications
print("TEST 1: Checking file modifications...")
try:
    modified_files = [
        'src/extraction/frame_extractor.py',
        'src/config/defaults.py',
        'requirements.txt',
        '360FrameTools.spec',
    ]
    
    missing = []
    for file in modified_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"  ✗ FAIL: Missing files: {missing}")
        results.append(('File Modifications', False))
    else:
        print(f"  ✓ PASS: All {len(modified_files)} modified files exist")
        results.append(('File Modifications', True))
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(('File Modifications', False))

print()

# Test 2: Check new files created
print("TEST 2: Checking new files created...")
try:
    new_files = [
        'src/masking/onnx_masker.py',
        'export_yolo_to_onnx.py',
        '360FrameTools_ONNX.spec',
        'test_optimizations.py',
        'OPTIMIZATION_SUMMARY.md',
        'QUICK_START_ONNX.md',
    ]
    
    missing = []
    for file in new_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"  ✗ FAIL: Missing files: {missing}")
        results.append(('New Files Created', False))
    else:
        print(f"  ✓ PASS: All {len(new_files)} new files created")
        results.append(('New Files Created', True))
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(('New Files Created', False))

print()

# Test 3: Check frame_extractor modifications
print("TEST 3: Checking frame_extractor.py modifications...")
try:
    content = Path('src/extraction/frame_extractor.py').read_text()
    
    # Check that OpenCV methods were removed
    has_opencv_extract = '_extract_with_opencv' in content
    has_opencv_dual = '_extract_dual_lens_opencv' in content
    
    if has_opencv_extract or has_opencv_dual:
        print(f"  ✗ FAIL: OpenCV extraction methods still present")
        results.append(('OpenCV Methods Removed', False))
    else:
        print(f"  ✓ PASS: OpenCV extraction methods removed")
        results.append(('OpenCV Methods Removed', True))
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(('OpenCV Methods Removed', False))

print()

# Test 4: Check config modifications
print("TEST 4: Checking config/defaults.py modifications...")
try:
    sys.path.insert(0, 'src')
    from config.defaults import EXTRACTION_METHODS
    
    # Check that OpenCV methods are removed from config
    opencv_methods = [m for m in EXTRACTION_METHODS.keys() if m.startswith('opencv_')]
    
    if opencv_methods:
        print(f"  ✗ FAIL: OpenCV methods still in config: {opencv_methods}")
        results.append(('Config Updated', False))
    else:
        print(f"  ✓ PASS: OpenCV methods removed from config")
        print(f"  Available methods: {list(EXTRACTION_METHODS.keys())}")
        results.append(('Config Updated', True))
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(('Config Updated', False))

print()

# Test 5: Check requirements.txt
print("TEST 5: Checking requirements.txt modifications...")
try:
    content = Path('requirements.txt').read_text()
    
    # Check torchvision is commented out
    lines = content.split('\n')
    torchvision_active = False
    onnx_mentioned = False
    
    for line in lines:
        if 'torchvision' in line and not line.strip().startswith('#'):
            torchvision_active = True
        if 'onnxruntime' in line:
            onnx_mentioned = True
    
    if torchvision_active:
        print(f"  ✗ FAIL: torchvision still active (should be commented)")
        results.append(('Requirements Updated', False))
    elif not onnx_mentioned:
        print(f"  ⚠ WARN: onnxruntime not mentioned (optional)")
        results.append(('Requirements Updated', True))
    else:
        print(f"  ✓ PASS: torchvision removed, onnxruntime mentioned")
        results.append(('Requirements Updated', True))
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(('Requirements Updated', False))

print()

# Test 6: Check ONNX masker structure
print("TEST 6: Checking ONNX masker module structure...")
try:
    content = Path('src/masking/onnx_masker.py').read_text()
    
    # Check for key components
    has_class = 'class ONNXMasker' in content
    has_generate_mask = 'def generate_mask' in content
    has_process_batch = 'def process_batch' in content
    has_onnxruntime = 'import onnxruntime' in content or 'onnxruntime' in content
    
    if not all([has_class, has_generate_mask, has_process_batch, has_onnxruntime]):
        missing = []
        if not has_class: missing.append('ONNXMasker class')
        if not has_generate_mask: missing.append('generate_mask method')
        if not has_process_batch: missing.append('process_batch method')
        if not has_onnxruntime: missing.append('onnxruntime import')
        
        print(f"  ✗ FAIL: Missing components: {missing}")
        results.append(('ONNX Masker Structure', False))
    else:
        print(f"  ✓ PASS: ONNX masker has all required components")
        results.append(('ONNX Masker Structure', True))
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(('ONNX Masker Structure', False))

print()

# Test 7: Check ONNX spec file
print("TEST 7: Checking ONNX spec file structure...")
try:
    content = Path('360FrameTools_ONNX.spec').read_text()
    
    # Check for key exclusions
    excludes_torch = "'torch'" in content or '"torch"' in content
    excludes_ultralytics = "'ultralytics'" in content or '"ultralytics"' in content
    has_onnxruntime = 'onnxruntime' in content
    
    if not all([excludes_torch, excludes_ultralytics, has_onnxruntime]):
        missing = []
        if not excludes_torch: missing.append('torch exclusion')
        if not excludes_ultralytics: missing.append('ultralytics exclusion')
        if not has_onnxruntime: missing.append('onnxruntime import')
        
        print(f"  ✗ FAIL: Missing spec components: {missing}")
        results.append(('ONNX Spec File', False))
    else:
        print(f"  ✓ PASS: ONNX spec file properly configured")
        results.append(('ONNX Spec File', True))
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(('ONNX Spec File', False))

print()

# Summary
print("="*70)
print("VERIFICATION SUMMARY")
print("="*70)

passed = sum(1 for _, result in results if result)
total = len(results)

for test_name, result in results:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status:8} {test_name}")

print("-"*70)
print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
print("="*70)
print()

if passed == total:
    print("🎉 All verification tests passed!")
    print()
    print("Next steps:")
    print("  1. Export ONNX models: python export_yolo_to_onnx.py")
    print("  2. Install ONNX Runtime: pip install onnxruntime")
    print("  3. Run full tests: python test_optimizations.py")
    print("  4. Build executable: pyinstaller 360FrameTools_ONNX.spec -y")
    print()
    sys.exit(0)
else:
    print(f"⚠ {total - passed} test(s) failed.")
    print()
    print("Review errors above and fix before building.")
    print()
    sys.exit(1)
