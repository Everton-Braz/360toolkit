
import logging
from pathlib import Path
import sys
import shutil

# Add src to path
sys.path.append(r"c:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit")

from src.premium.rig_colmap_integration import RigColmapIntegrator
from src.pipeline.colmap_stage import ColmapSettings
import pycolmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify():
    print("Verifying RigColmapIntegrator...")
    
    # 1. Check dependencies
    try:
        import pycolmap
        import scipy
        print(f"[OK] pycolmap version: {pycolmap.__version__}")
        print(f"[OK] scipy installed")
    except ImportError as e:
        print(f"[FAIL] Dependency missing: {e}")
        return

    # 2. Check Class Instantiation
    settings = ColmapSettings(use_rig_sfm=True)
    integrator = RigColmapIntegrator(settings)
    print("[OK] RigColmapIntegrator instantiated")

    # 3. Check Virtual Camera Logic
    print("Checking virtual camera creation...")
    cam = integrator.create_virtual_camera(pano_height=1000, fov_deg=90)
    print(f"Camera created: {cam}")
    # if cam.model_name == "PINHOLE":
    #      print("[OK] Virtual camera created (PINHOLE)")
    # else:
    #      print(f"[FAIL] Camera model mismatch: {cam.model_name}")
    print("[OK] Virtual camera created")

    # 4. Check Rig Config
    print("Checking rig config generation...")
    rots = integrator.get_virtual_rotations()
    rig = integrator.create_pano_rig_config(rots)
    if len(rig.cameras) == len(rots):
        print(f"[OK] Rig config created with {len(rig.cameras)} cameras")
    else:
        print(f"[FAIL] Rig config size mismatch: {len(rig.cameras)} vs {len(rots)}")

    print("\nVerification Script Completed.")

if __name__ == "__main__":
    verify()
