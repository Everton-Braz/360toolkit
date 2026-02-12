"""Quick test for CPU multiprocessing worker"""
import sys
from pathlib import Path
import tempfile

# Test the worker function
from src.pipeline.batch_orchestrator import process_frame_cpu

# Create test task
test_frame = Path(r"C:\Users\User\Documents\APLICATIVOS\Arquivos_Teste\TESTE_14\stage1_frames\0.png")
if not test_frame.exists():
    print(f"Test frame not found: {test_frame}")
    sys.exit(1)

output_dir = Path(tempfile.mkdtemp())
print(f"Output dir: {output_dir}")

cameras = [
    {'yaw': 0, 'pitch': 0, 'roll': 0, 'fov': 110},
    {'yaw': 90, 'pitch': 0, 'roll': 0, 'fov': 110},
]

task_data = (
    str(test_frame),
    0,  # frame_idx
    cameras,
    str(output_dir),
    1920,  # width
    1080,  # height
    'png'  # format
)

print("Running CPU worker...")
result = process_frame_cpu(task_data)

if result['success']:
    print(f"✓ Success! Generated {len(result['files'])} images")
    for f in result['files']:
        size = Path(f).stat().st_size
        print(f"  - {Path(f).name}: {size:,} bytes")
        
    # Check if images are black
    import cv2
    for img_path in result['files']:
        img = cv2.imread(img_path)
        if img is not None:
            mean_val = img.mean()
            print(f"  - {Path(img_path).name}: mean pixel value = {mean_val:.2f}")
            if mean_val < 1:
                print("    ⚠ Image appears to be BLACK!")
        else:
            print(f"    ⚠ Failed to load {img_path}")
else:
    print(f"✗ Failed: {result.get('error')}")
    sys.exit(1)
