from pathlib import Path
from types import SimpleNamespace

from src.config.settings import SettingsManager
from src.premium.rig_colmap_integration import RigColmapIntegrator, _resolve_runtime_path
from src.utils.colmap_paths import normalize_colmap_executable, resolve_default_colmap_path


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


def test_resolve_default_colmap_path_prefers_exe_over_batch(tmp_path: Path):
    preferred_root = tmp_path / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
    preferred_root.mkdir(parents=True, exist_ok=True)
    exe_path = preferred_root / "bin" / "colmap.exe"
    exe_path.parent.mkdir(parents=True, exist_ok=True)
    exe_path.write_text("fake")
    (preferred_root / "COLMAP.bat").write_text("fake")

    resolved = resolve_default_colmap_path(tmp_path)

    assert resolved == exe_path


def test_prepare_learned_model_args_batch_wrapper_uses_bin_models(tmp_path: Path):
    colmap_root = tmp_path / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
    colmap_root.mkdir(parents=True, exist_ok=True)
    batch_path = colmap_root / "COLMAP.bat"
    batch_path.write_text("fake")

    models_dir = colmap_root / "bin" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "aliked-n16rot.onnx").write_bytes(b"onnx")
    (models_dir / "aliked-lightglue.onnx").write_bytes(b"onnx")

    integrator = RigColmapIntegrator(_settings())
    feature_args, matcher_args = integrator._prepare_learned_model_args(
        extraction_type="ALIKED_N16ROT",
        matcher_type="ALIKED_LIGHTGLUE",
        colmap_bin=str(batch_path),
    )

    assert str(models_dir / "aliked-n16rot.onnx") in feature_args
    assert str(models_dir / "aliked-lightglue.onnx") in matcher_args


def test_normalize_colmap_executable_prefers_bin_exe_for_batch_wrapper(tmp_path: Path):
    colmap_root = tmp_path / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
    colmap_root.mkdir(parents=True, exist_ok=True)
    batch_path = colmap_root / "COLMAP.bat"
    batch_path.write_text("fake")
    exe_path = colmap_root / "bin" / "colmap.exe"
    exe_path.parent.mkdir(parents=True, exist_ok=True)
    exe_path.write_text("fake")

    assert normalize_colmap_executable(batch_path) == exe_path.resolve()


def test_settings_manager_migrates_saved_batch_path_to_exe(tmp_path: Path):
    colmap_root = tmp_path / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
    colmap_root.mkdir(parents=True, exist_ok=True)
    batch_path = colmap_root / "COLMAP.bat"
    batch_path.write_text("fake")
    exe_path = colmap_root / "bin" / "colmap.exe"
    exe_path.parent.mkdir(parents=True, exist_ok=True)
    exe_path.write_text("fake")

    settings_file = tmp_path / "user_settings.json"
    settings_file.write_text(
        '{"colmap_gpu_path": "%s", "colmap_path": "%s"}'
        % (str(batch_path).replace('\\', '\\\\'), str(batch_path).replace('\\', '\\\\')),
        encoding="utf-8",
    )

    settings = SettingsManager(settings_file=settings_file)

    assert settings.get_colmap_gpu_path() == exe_path.resolve()
    assert settings.settings["colmap_gpu_path"] == str(exe_path.resolve())


def test_filter_supported_cli_args_drops_unsupported_global_mapper_flags(tmp_path: Path):
    colmap_bin = tmp_path / "colmap.exe"
    colmap_bin.write_text("fake")

    integrator = RigColmapIntegrator(_settings())
    integrator._cli_help_cache[(str(colmap_bin.resolve()).lower(), "global_mapper")] = (
        "--globalmapper.min_num_matches arg\n"
        "--globalmapper.gp_use_gpu arg\n"
    )

    filtered = integrator._filter_supported_cli_args(
        str(colmap_bin),
        "global_mapper",
        [
            "--GlobalMapper.vgc_relpose_max_error", "4",
            "--GlobalMapper.min_num_matches", "8",
        ],
    )

    assert filtered == ["--GlobalMapper.min_num_matches", "8"]


def test_resolve_runtime_path_makes_relative_paths_absolute(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_runtime_path(Path("relative") / "reconstruction")

    assert resolved == (tmp_path / "relative" / "reconstruction").resolve()
