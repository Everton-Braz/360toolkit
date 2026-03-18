from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from src.utils.app_paths import ensure_user_data_dir
from src.utils.resource_path import get_base_path, get_resource_path


logger = logging.getLogger(__name__)

COLMAP_DIR_NAME = "COLMAP-windows-latest-CUDA-cuDSS-GUI"
SPHERESFM_DIR_NAME = "SphereSfM-2025-8-18"

DEFAULT_COLMAP_URL = os.environ.get(
    "TOOLKIT_COLMAP_URL",
    "https://github.com/colmap/colmap/releases/download/4.0.1/colmap-x64-windows-cuda.zip",
)
DEFAULT_SPHERESFM_URL = os.environ.get(
    "TOOLKIT_SPHERESFM_URL",
    "https://github.com/json87/SphereSfM/releases/download/V1.2/SphereSfM-2025-8-18.zip",
)


def get_external_tools_dir() -> Path:
    path = ensure_user_data_dir() / "external-tools"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_cache_dir() -> Path:
    path = ensure_user_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_download_cache_dir() -> Path:
    path = ensure_user_data_dir() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_downloaded_colmap_candidates() -> List[Path]:
    root = get_external_tools_dir() / COLMAP_DIR_NAME
    return [
        root / "bin" / "colmap.exe",
        root / "COLMAP.bat",
        root / "colmap.exe",
        root / "colmap.bat",
        root / "colmap.cmd",
    ]


def get_downloaded_spheresfm_candidates() -> List[Path]:
    root = get_external_tools_dir() / SPHERESFM_DIR_NAME
    return [
        root / "colmap.exe",
        root / "colmap.bat",
        root / "colmap.cmd",
        root / "bin" / "colmap.exe",
    ]


def resolve_masking_model_path(model_name: str) -> Path:
    candidates = [
        get_resource_path(model_name),
        get_base_path() / model_name,
        Path.cwd() / model_name,
        get_model_cache_dir() / model_name,
    ]
    existing = _existing_path(candidates)
    return existing if existing is not None else candidates[0]


def _download_file(url: str, target_path: Path, *, label: str) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Downloading %s", label, url)
    with urllib.request.urlopen(url) as response, target_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if not target_path.exists() or target_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is missing or empty: {target_path}")
    return target_path


def _move_extracted_tree(staging_dir: Path, install_root: Path) -> None:
    children = [child for child in staging_dir.iterdir() if child.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        extracted_root = children[0]
        if install_root.exists():
            shutil.rmtree(install_root, ignore_errors=True)
        install_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_root), str(install_root))
        return

    if install_root.exists():
        shutil.rmtree(install_root, ignore_errors=True)
    install_root.mkdir(parents=True, exist_ok=True)
    for child in children:
        shutil.move(str(child), str(install_root / child.name))


def _ensure_zip_install(
    *,
    url: str,
    archive_name: str,
    install_root: Path,
    candidate_paths: Iterable[Path],
    label: str,
    force: bool = False,
) -> Path:
    existing = _existing_path(candidate_paths)
    if existing is not None and not force:
        return existing

    cache_zip = get_download_cache_dir() / archive_name
    if force or not cache_zip.exists() or cache_zip.stat().st_size == 0:
        _download_file(url, cache_zip, label=label)

    with tempfile.TemporaryDirectory(prefix=f"{label.lower()}_") as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(cache_zip, "r") as archive:
            archive.extractall(temp_path)
        _move_extracted_tree(temp_path, install_root)

    installed = _existing_path(candidate_paths)
    if installed is None:
        raise RuntimeError(f"{label} install completed but no executable was found under {install_root}")
    logger.info("[%s] Installed to %s", label, install_root)
    return installed


def ensure_colmap_downloaded(force: bool = False) -> Path:
    return _ensure_zip_install(
        url=DEFAULT_COLMAP_URL,
        archive_name="colmap-x64-windows-cuda.zip",
        install_root=get_external_tools_dir() / COLMAP_DIR_NAME,
        candidate_paths=get_downloaded_colmap_candidates(),
        label="COLMAP",
        force=force,
    )


def ensure_spheresfm_downloaded(force: bool = False) -> Path:
    return _ensure_zip_install(
        url=DEFAULT_SPHERESFM_URL,
        archive_name="SphereSfM-2025-8-18.zip",
        install_root=get_external_tools_dir() / SPHERESFM_DIR_NAME,
        candidate_paths=get_downloaded_spheresfm_candidates(),
        label="SphereSfM",
        force=force,
    )