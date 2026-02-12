"""Direct test of E2PTransform"""
import cv2
import numpy as np
from pathlib import Path

# Test the transform directly
from src.transforms.e2p_transform import E2PTransform

test_frame = Path(r"C:\Users\User\Documents\APLICATIVOS\Arquivos_Teste\TESTE_14\stage1_frames\0.png")
if not test_frame.exists():
    print(f"Test frame not found: {test_frame}")
    exit(1)

print(f"Loading: {test_frame}")
equirect_img = cv2.imread(str(test_frame))
if equirect_img is None:
    print("Failed to load image!")
    exit(1)

print(f"Loaded image shape: {equirect_img.shape}, dtype: {equirect_img.dtype}")
print(f"Image stats: min={equirect_img.min()}, max={equirect_img.max()}, mean={equirect_img.mean():.2f}")

# Test transform
transformer = E2PTransform()
print("\nApplying E2PTransform with yaw=0, pitch=0, roll=0, fov=110...")

perspective_img = transformer.equirect_to_pinhole(
    equirect_img,
    yaw=0,
    pitch=0,
    roll=0,
    h_fov=110,
    v_fov=None,
    output_width=1920,
    output_height=1080
)

print(f"Output shape: {perspective_img.shape}, dtype: {perspective_img.dtype}")
print(f"Output stats: min={perspective_img.min()}, max={perspective_img.max()}, mean={perspective_img.mean():.2f}")

if perspective_img.mean() < 1:
    print("⚠ Output is BLACK!")
    
    # Debug the transform maps
    print("\nDebugging transform maps...")
    cache_key = (0, 0, 0, 110, 110 * 1080 / 1920, 1920, 1080, equirect_img.shape[0], equirect_img.shape[1])
    if cache_key in transformer.cache:
        map_x, map_y = transformer.cache[cache_key]
        print(f"map_x: shape={map_x.shape}, min={map_x.min():.2f}, max={map_x.max():.2f}, mean={map_x.mean():.2f}")
        print(f"map_y: shape={map_y.shape}, min={map_y.min():.2f}, max={map_y.max():.2f}, mean={map_y.mean():.2f}")
        
        # Sample some coordinates
        print(f"\nSample coordinates (center pixel {1920//2}, {1080//2}):")
        cx, cy = 1920//2, 1080//2
        print(f"  map_x[{cy},{cx}] = {map_x[cy,cx]:.2f}")
        print(f"  map_y[{cy},{cx}] = {map_y[cy,cx]:.2f}")
        print(f"  equirect pixel at ({map_x[cy,cx]:.0f}, {map_y[cy,cx]:.0f})")
else:
    print("✓ Output looks good!")
    
    # Save test output
    out_path = "test_perspective_direct.png"
    cv2.imwrite(out_path, perspective_img)
    print(f"Saved test output to {out_path}")
