#!/usr/bin/env python3
"""
Direct CLI test of segment_persons.exe to diagnose why no masks are being produced.
"""
import sys
import os
import tempfile
import subprocess
import ctypes
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings

input_folder = Path(r"D:\ARQUIVOS_TESTE_2\Pecem_8K\PÉCEM_FISHEYE\extracted_frames")
output_folder = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\test_cli_masks")
output_folder.mkdir(parents=True, exist_ok=True)

settings = get_settings()
exe_path = Path(settings.get('sam3_segmenter_path'))
model_path = Path(settings.get('sam3_model_path'))

print(f"Input: {input_folder}")
print(f"Output: {output_folder}")
print(f"Exe: {exe_path}")
print(f"Model: {model_path}")
print(f"Exe exists: {exe_path.exists()}")
print(f"Model exists: {model_path.exists()}")

# Get first 5 images only for testing
image_files = sorted([p for p in input_folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:5]
print(f"\nUsing {len(image_files)} test images:")
for img in image_files:
    print(f"  - {img.name}")

# Helper to convert to short path
def get_short_path(path_str):
    """Convert Windows path to 8.3 short format"""
    try:
        GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
        short_path = ctypes.create_unicode_buffer(260)
        result = GetShortPathName(path_str, short_path, 260)
        if result:
            return short_path.value
    except Exception:
        pass
    return path_str

# Write image list file using SHORT PATHS
list_fd, list_path = tempfile.mkstemp(suffix='.txt', prefix='sam3_list_')
with open(list_path, 'w', encoding='cp1252') as f:
    for p in image_files:
        path_str = str(p)
        short_path = get_short_path(path_str)
        print(f"  Short path: {path_str} -> {short_path}")
        f.write(short_path + '\n')

print(f"\nImage list file: {list_path}")
print(f"Short list path: {get_short_path(list_path)}")

try:
    # Build command using short paths
    cmd = [
        str(exe_path),
        '--model',      str(model_path),
        '--image-list', get_short_path(list_path),  # Use short path for list file too
        '--output-dir', str(output_folder),
        '--prompts',    'person',
        '--score',      '0.5',
        '--nms',        '0.1',
        '--gpu'
    ]
    
    print(f"\nCommand:")
    print(" ".join(cmd))
    print("\n" + "="*80)
    print("Running segment_persons.exe...")
    print("="*80 + "\n")
    
    # Run subprocess
    proc = subprocess.Popen(
        cmd,
        cwd=str(output_folder),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Read output
    stdout, stderr = proc.communicate(timeout=120)
    
    print("STDOUT:")
    print(stdout)
    print("\nSTDERR:")
    print(stderr)
    print(f"\nReturn code: {proc.returncode}")
    
    # Check output
    output_masks = list(output_folder.glob("*_mask.png"))
    print(f"\nOutput masks created: {len(output_masks)}")
    for mask in output_masks:
        print(f"  - {mask.name}")
    
finally:
    try:
        Path(list_path).unlink()
    except:
        pass
