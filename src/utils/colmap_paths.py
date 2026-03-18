from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.utils.dependency_provisioning import get_downloaded_colmap_candidates
from src.utils.resource_path import get_base_path


def get_project_root() -> Path:
    return get_base_path()


def preferred_colmap_candidates(project_root: Path | None = None) -> List[Path]:
    root = Path(project_root) if project_root else get_project_root()
    preferred_root = root / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
    legacy_root = root / "bin" / "colmap"

    return [
        preferred_root / "bin" / "colmap.exe",
        preferred_root / "COLMAP.bat",
        legacy_root / "colmap.exe",
        legacy_root / "colmap.bat",
        legacy_root / "colmap.cmd",
        *get_downloaded_colmap_candidates(),
    ]


def resolve_default_colmap_path(project_root: Path | None = None) -> Path:
    candidates = preferred_colmap_candidates(project_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_colmap_executable(colmap_path: Path | str | None) -> Path | None:
    if colmap_path is None:
        return None

    path = Path(colmap_path)

    candidate_paths: List[Path] = []
    if path.is_dir():
        candidate_paths.extend([
            path / "colmap.exe",
            path / "bin" / "colmap.exe",
            path / "colmap.bat",
            path / "colmap.cmd",
        ])
    else:
        if path.name.lower() in {"colmap.bat", "colmap.cmd"}:
            candidate_paths.append(path.parent / "bin" / "colmap.exe")
        candidate_paths.append(path)

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return path.resolve() if path.is_absolute() else path


def get_colmap_runtime_dirs(colmap_path: Path | str) -> List[Path]:
    path = Path(colmap_path)
    candidates: List[Path] = []

    if path.is_dir():
        candidates.extend([path, path / "bin"])
    else:
        candidates.append(path.parent)
        if path.parent.name.lower() == "bin":
            candidates.append(path.parent.parent)
        elif (path.parent / "bin").exists():
            candidates.append(path.parent / "bin")

    seen: set[str] = set()
    resolved: List[Path] = []
    for candidate in candidates:
        try:
            candidate_resolved = candidate.resolve()
        except Exception:
            candidate_resolved = candidate
        key = str(candidate_resolved).lower()
        if key in seen or not candidate_resolved.exists() or not candidate_resolved.is_dir():
            continue
        seen.add(key)
        resolved.append(candidate_resolved)
    return resolved


def _path_is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _detect_bundle_internal_root(executable_path: Path) -> Path | None:
    parts = [part.lower() for part in executable_path.parts]
    if "_internal" not in parts:
        return None

    internal_index = parts.index("_internal")
    return Path(*executable_path.parts[: internal_index + 1])


def build_colmap_cli_context(
    executable: Path | str,
    *,
    base_env: Dict[str, str] | None = None,
    extra_dirs: Iterable[Path | str] | None = None,
) -> Tuple[str | None, Dict[str, str] | None]:
    exe_path = normalize_colmap_executable(executable)
    run_cwd = str(exe_path.parent) if exe_path and exe_path.exists() else None
    if not exe_path or not exe_path.exists():
        return run_cwd, None

    run_env = dict(base_env or os.environ)

    preferred_dirs = get_colmap_runtime_dirs(exe_path)
    if extra_dirs:
        for extra_dir in extra_dirs:
            if not extra_dir:
                continue
            candidate = Path(extra_dir).expanduser()
            try:
                candidate = candidate.resolve()
            except Exception:
                pass
            if candidate.exists() and candidate.is_dir():
                preferred_dirs.append(candidate)

    unique_dirs: List[Path] = []
    seen_dirs: set[str] = set()
    for preferred_dir in preferred_dirs:
        try:
            resolved_dir = preferred_dir.resolve()
        except Exception:
            resolved_dir = preferred_dir
        key = str(resolved_dir).lower()
        if key in seen_dirs or not resolved_dir.exists() or not resolved_dir.is_dir():
            continue
        seen_dirs.add(key)
        unique_dirs.append(resolved_dir)

    bundle_internal_root = _detect_bundle_internal_root(exe_path)
    allowed_dir_keys = {str(path).lower() for path in unique_dirs}

    filtered_path_entries: List[str] = []
    original_path = run_env.get("PATH", "")
    for raw_entry in original_path.split(os.pathsep):
        entry = raw_entry.strip().strip('"')
        if not entry:
            continue
        entry_path = Path(entry)
        try:
            resolved_entry = entry_path.resolve()
        except Exception:
            resolved_entry = entry_path

        if bundle_internal_root and _path_is_relative_to(resolved_entry, bundle_internal_root):
            entry_key = str(resolved_entry).lower()
            if entry_key not in allowed_dir_keys:
                continue

        if entry not in filtered_path_entries:
            filtered_path_entries.append(entry)

    run_env["PATH"] = os.pathsep.join([str(path) for path in unique_dirs] + filtered_path_entries)

    for env_key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH"):
        env_value = run_env.get(env_key)
        if not env_value:
            continue
        if not bundle_internal_root:
            run_env.pop(env_key, None)
            continue

        keep_values: List[str] = []
        for raw_value in env_value.split(os.pathsep):
            value = raw_value.strip().strip('"')
            if not value:
                continue
            value_path = Path(value)
            try:
                resolved_value = value_path.resolve()
            except Exception:
                resolved_value = value_path
            if _path_is_relative_to(resolved_value, bundle_internal_root):
                continue
            keep_values.append(value)

        if keep_values:
            run_env[env_key] = os.pathsep.join(keep_values)
        else:
            run_env.pop(env_key, None)

    return run_cwd, run_env


def format_path_list(paths: Iterable[Path]) -> List[str]:
    return [str(path) for path in paths]