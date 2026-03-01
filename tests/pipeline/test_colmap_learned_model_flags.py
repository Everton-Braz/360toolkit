from pathlib import Path
from types import SimpleNamespace

from src.premium.rig_colmap_integration import RigColmapIntegrator


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        use_gpu=True,
        strict_gpu_only=True,
        use_lightglue_aliked=True,
        gpu_index=0,
    )


def test_prepare_learned_model_args_aliked_lightglue(tmp_path: Path):
    colmap_bin = tmp_path / "colmap.exe"
    colmap_bin.write_text("fake")

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "aliked-n32.onnx").write_bytes(b"onnx")
    (models_dir / "aliked-lightglue.onnx").write_bytes(b"onnx")

    integrator = RigColmapIntegrator(_settings())
    feature_args, matcher_args = integrator._prepare_learned_model_args(
        extraction_type="ALIKED_N32",
        matcher_type="ALIKED_LIGHTGLUE",
        colmap_bin=str(colmap_bin),
    )

    assert "--AlikedExtraction.n32_model_path" in feature_args
    assert str(models_dir / "aliked-n32.onnx") in feature_args

    assert "--AlikedMatching.lightglue_model_path" in matcher_args
    assert str(models_dir / "aliked-lightglue.onnx") in matcher_args


def test_prepare_learned_model_args_sift_lightglue(tmp_path: Path):
    colmap_bin = tmp_path / "colmap.exe"
    colmap_bin.write_text("fake")

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "sift-lightglue.onnx").write_bytes(b"onnx")

    integrator = RigColmapIntegrator(_settings())
    feature_args, matcher_args = integrator._prepare_learned_model_args(
        extraction_type=None,
        matcher_type="SIFT_LIGHTGLUE",
        colmap_bin=str(colmap_bin),
    )

    assert feature_args == []
    assert "--SiftMatching.lightglue_model_path" in matcher_args
    assert str(models_dir / "sift-lightglue.onnx") in matcher_args
