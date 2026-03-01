from pathlib import Path

from src.pipeline.batch_orchestrator import PipelineWorker

base = Path(r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\teste_17022026-1515")
workspace = Path(r"C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit")

config = {
    "output_dir": str(base),
    "stage2_enabled": True,
    "stage3_enabled": False,
    "stage4_enabled": True,
    "alignment_mode": "perspective_reconstruction",
    "mapping_backend": "glomap",
    "use_gpu_colmap": True,
    "use_gpu": True,
    "use_lightglue_aliked": True,
    "enable_hloc_fallback": True,
    "prefer_colmap_learned": False,
    "require_learned_pipeline": False,
    "reuse_colmap_database": True,
    "lichtfeld_fix_rotation": True,
    "export_realityscan": True,
    "export_lichtfeld": True,
    "export_include_masks": True,
    "colmap_path": str(workspace / "bin" / "colmap" / "colmap.exe"),
}

worker = PipelineWorker(config)
result = worker._execute_stage4()
print(result)
