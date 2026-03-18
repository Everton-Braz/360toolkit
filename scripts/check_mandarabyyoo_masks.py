"""Check per-image mask mapping for the mandarabyyoo dataset.

This script verifies that project-style masks (`<stem>_mask.png`) can be mapped
to COLMAP/SphereSfM-style per-image masks (`<image-name>.png`).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.colmap_stage import ColmapSettings
from src.premium.sphere_sfm_integration import SphereSfMIntegrator


DATASET_ROOT = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\mandarabyyoo")
FRAMES_DIR = DATASET_ROOT / "extracted_frames"
MASKS_DIR = DATASET_ROOT / "masks"
OUTPUT_DIR = DATASET_ROOT / "spheresfm_diagnostics" / "prepared_masks_from_script"


def main() -> int:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    integrator = SphereSfMIntegrator(ColmapSettings())
    result = integrator._prepare_colmap_masks(FRAMES_DIR, MASKS_DIR, OUTPUT_DIR)
    prepared_masks = sorted(OUTPUT_DIR.rglob("*.png"))

    payload = {
        "python_executable": sys.executable,
        "frames_dir": str(FRAMES_DIR),
        "masks_dir": str(MASKS_DIR),
        "prepared_dir": str(result) if result else None,
        "prepared_mask_count": len(prepared_masks),
        "sample_prepared_masks": [path.name for path in prepared_masks[:10]],
    }
    print(json.dumps(payload, indent=2))

    if result is None or len(prepared_masks) == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())