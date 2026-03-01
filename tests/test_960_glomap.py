# Automated test: GLOMAP reconstruction on 960-image dataset.
#
# Input:  C:/Users/Everton-PC/Documents/ARQUIVOS_TESTE/TESTE_18022026_1211/perspective_views/
# Masks:  C:/Users/Everton-PC/Documents/ARQUIVOS_TESTE/TESTE_18022026_1211/masks/  (if present)
# Output: C:/Users/Everton-PC/Documents/ARQUIVOS_TESTE/TESTE_18022026_1211/reconstruction/

import sys
import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
try:
    import torch  # noqa: F401
except Exception:
    pass

import logging
import json
import shutil
from pathlib import Path
from datetime import datetime

TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLKIT_ROOT))

TEST_ROOT  = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\TESTE_18022026_1211")
VIEWS_DIR  = TEST_ROOT / "perspective_views"
MASKS_DIR  = TEST_ROOT / "masks"
RECON_DIR  = TEST_ROOT / "reconstruction"
COLMAP_BIN = Path(r"C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit\bin\colmap\colmap.exe")
LOG_FILE   = TEST_ROOT / "test_960_glomap_log.txt"

TEST_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_960")


def print_section(title: str):
    sep = "=" * 60
    logger.info("\n%s\n  %s\n%s", sep, title, sep)


def main():
    start = datetime.now()
    print_section("TEST START -- 960-image GLOMAP reconstruction")

    n_views = len(list(VIEWS_DIR.glob("*"))) if VIEWS_DIR.exists() else 0
    n_masks = len(list(MASKS_DIR.glob("*"))) if MASKS_DIR.exists() else 0
    logger.info("Toolkit root : %s", TOOLKIT_ROOT)
    logger.info("Views dir    : %s  (%d images)", VIEWS_DIR, n_views)
    logger.info("Masks dir    : %s  (%d masks)",  MASKS_DIR, n_masks)
    logger.info("Recon dir    : %s", RECON_DIR)
    logger.info("COLMAP bin   : %s  (exists=%s)", COLMAP_BIN, COLMAP_BIN.exists())

    if not VIEWS_DIR.exists() or n_views == 0:
        logger.error("No images found in %s — aborting.", VIEWS_DIR)
        sys.exit(1)

    if RECON_DIR.exists():
        # Full clean — pairs logic changed (Soft C: no intra-frame), must redo matching + GLOMAP.
        logger.info("Removing full reconstruction folder for fresh Soft-C run: %s", RECON_DIR)
        shutil.rmtree(RECON_DIR)
    RECON_DIR.mkdir(parents=True, exist_ok=True)

    # --- Settings ---
    from src.pipeline.colmap_stage import ColmapSettings
    settings = ColmapSettings(
        alignment_mode="perspective_reconstruction",
        mapping_backend="glomap",
        use_lightglue_aliked=True,
        prefer_colmap_learned=False,
        enable_hloc_fallback=True,
        reuse_colmap_database=False,   # Fresh run — pairs changed (Soft C)
        use_gpu=True,
        gpu_index=0,
        quality="medium",
        camera_grouping="per_folder",
    )
    settings.hloc_prefer_min_images = 50
    if COLMAP_BIN.exists():
        settings.colmap_path = COLMAP_BIN
    logger.info("Settings: %s", settings.to_dict())

    # --- Reconstruction ---
    from src.premium.rig_colmap_integration import RigColmapIntegrator

    def on_progress(msg: str):
        logger.info("[PROGRESS] %s", msg)

    print_section("STAGE: GLOMAP Reconstruction")
    integrator = RigColmapIntegrator(settings)
    result = integrator.run_alignment(
        frames_dir=VIEWS_DIR,
        masks_dir=MASKS_DIR if MASKS_DIR.exists() else None,
        output_dir=RECON_DIR,
        progress_callback=on_progress,
    )

    if not result.get("success"):
        logger.error("Reconstruction FAILED: %s", result.get("error"))
        sys.exit(1)

    colmap_output = Path(result["colmap_output"])
    num_aligned   = result.get("num_aligned", 0)
    logger.info("Reconstruction SUCCESS -- %d images aligned", num_aligned)
    logger.info("   Sparse model: %s", colmap_output)

    if num_aligned == 0:
        logger.error("0 images registered -- exports will be empty. Stopping.")
        sys.exit(1)

    # Convert .bin -> .txt if needed
    images_txt = colmap_output / "images.txt"
    if not images_txt.exists() and (colmap_output / "images.bin").exists():
        import subprocess
        logger.info("Converting .bin -> .txt for export...")
        r = subprocess.run(
            [str(COLMAP_BIN), "model_converter",
             "--input_path", str(colmap_output),
             "--output_path", str(colmap_output),
             "--output_type", "TXT"],
            capture_output=True, text=True,
        )
        logger.info("model_converter exit: %d", r.returncode)

    # --- Export: RealityScan ---
    print_section("STAGE: Export --> RealityScan")
    try:
        from src.premium.pose_transfer_integration import export_for_realityscan
        rs_output = RECON_DIR / "export_realityscan"
        rs_output.mkdir(parents=True, exist_ok=True)
        ok = export_for_realityscan(
            colmap_dir=str(colmap_output),
            images_dir=str(VIEWS_DIR),
            masks_dir=str(MASKS_DIR) if MASKS_DIR.exists() else "",
            output_dir=str(rs_output),
        )
        logger.info("export_for_realityscan returned: %s", ok)
        rs_images  = list((rs_output / "images").glob("*")) if (rs_output / "images").exists() else []
        rs_pts_txt = rs_output / "sparse" / "points3D.txt"
        pts_lines  = sum(1 for ln in open(rs_pts_txt) if not ln.startswith("#")) if rs_pts_txt.exists() else 0
        logger.info("RealityScan export done: Images=%d, 3D points=%d", len(rs_images), pts_lines)
        if pts_lines == 0:
            logger.warning("points3D.txt is empty -- triangulation may have been skipped or failed")
    except Exception as exc:
        import traceback
        logger.error("RealityScan export FAILED: %s\n%s", exc, traceback.format_exc())

    # --- Export: LichtFeld ---
    print_section("STAGE: Export --> LichtFeld Studio")
    try:
        from src.pipeline.export_formats import LichtfeldExporter
        lichtfeld_output = RECON_DIR / "export_lichtfeld"
        lichtfeld_output.mkdir(parents=True, exist_ok=True)
        exporter = LichtfeldExporter(
            colmap_dir=str(colmap_output),
            output_dir=str(lichtfeld_output),
        )
        export_result = exporter.export(images_dir=str(VIEWS_DIR), fix_rotation=True)
        logger.info("LichtFeld export result: %s", export_result)
        transforms_file = lichtfeld_output / "transforms.json"
        if transforms_file.exists():
            data   = json.loads(transforms_file.read_text(encoding="utf-8"))
            frames = data.get("frames", [])
            logger.info("LichtFeld export done: %d frames in transforms.json", len(frames))
        else:
            logger.warning("transforms.json not found in %s", lichtfeld_output)
    except Exception as exc:
        import traceback
        logger.error("LichtFeld export FAILED: %s\n%s", exc, traceback.format_exc())

    # --- Summary ---
    elapsed = datetime.now() - start
    print_section("TEST COMPLETE")
    logger.info("Total time: %s", elapsed)
    logger.info("Log: %s", LOG_FILE)
    logger.info("Output: %s", RECON_DIR)


if __name__ == "__main__":
    main()
