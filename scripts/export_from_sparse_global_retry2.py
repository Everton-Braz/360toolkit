from pathlib import Path

from src.premium.pose_transfer_integration import export_for_realityscan
from src.pipeline.export_formats import LichtfeldExporter

base = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\teste_17022026-1515")
recon = base / "reconstruction"
colmap_dir = recon / "sparse_global_retry2" / "0"
images_dir = base / "perspective_views"
masks_dir = base / "masks"
db_path = recon / "database.db"

out_root = recon / "global_mapper_export"
out_root.mkdir(parents=True, exist_ok=True)

rs_out = out_root / "realityscan_export"
lf_out = out_root / "lichtfeld_export"

ok_rs = export_for_realityscan(
    colmap_dir=str(colmap_dir),
    images_dir=str(images_dir),
    masks_dir=str(masks_dir) if masks_dir.exists() else None,
    output_dir=str(rs_out),
    database_path=str(db_path) if db_path.exists() else None,
)
print(f"RealityScan export ok: {ok_rs}")

lf = LichtfeldExporter(colmap_dir=str(colmap_dir), output_dir=str(lf_out))
ok_lf = lf.export(images_dir=str(images_dir), fix_rotation=True, masks_dir=str(masks_dir) if masks_dir.exists() else None)
print(f"Lichtfeld export ok: {ok_lf}")
