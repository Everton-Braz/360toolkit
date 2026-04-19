"""Shared Stage 2 naming, ordering, and output path helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from src.config.defaults import (
    DEFAULT_STAGE2_LAYOUT_MODE,
    DEFAULT_STAGE2_NUMBERING_MODE,
)

_LAST_INTEGER_RE = re.compile(r'(\d+)(?!.*\d)')
_PERSPECTIVE_NAME_RE = re.compile(r'^frame_(\d+)_cam_(\d+)\.[^.]+$', re.IGNORECASE)
_CUBEMAP_NAME_RE = re.compile(r'^frame_(\d+)_([^./\\]+)\.[^.]+$', re.IGNORECASE)


def normalize_stage2_numbering_mode(value: str | None) -> str:
    normalized = str(value or DEFAULT_STAGE2_NUMBERING_MODE).strip().lower()
    return normalized if normalized in {'preserve_source', 'sequential'} else DEFAULT_STAGE2_NUMBERING_MODE


def normalize_stage2_layout_mode(value: str | None) -> str:
    normalized = str(value or DEFAULT_STAGE2_LAYOUT_MODE).strip().lower()
    return normalized if normalized in {'flat', 'by_camera'} else DEFAULT_STAGE2_LAYOUT_MODE


def extract_frame_id(path_or_name: str | Path) -> Optional[int]:
    name = Path(path_or_name).stem
    match = _LAST_INTEGER_RE.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def stage2_input_sort_key(path_or_name: str | Path) -> tuple[int, int, str]:
    path = Path(path_or_name)
    frame_id = extract_frame_id(path)
    if frame_id is None:
        return (1, 0, path.name.lower())
    return (0, frame_id, path.name.lower())


def sort_stage2_input_frames(paths: Iterable[str | Path]) -> list[Path]:
    return sorted((Path(path) for path in paths), key=stage2_input_sort_key)


def resolve_output_frame_id(path_or_name: str | Path, sequential_index: int, numbering_mode: str | None) -> int:
    mode = normalize_stage2_numbering_mode(numbering_mode)
    if mode == 'sequential':
        return sequential_index
    source_frame_id = extract_frame_id(path_or_name)
    return sequential_index if source_frame_id is None else source_frame_id


def build_stage2_frame_records(paths: Iterable[str | Path], numbering_mode: str | None) -> list[tuple[Path, int]]:
    sorted_paths = sort_stage2_input_frames(paths)
    return [
        (path, resolve_output_frame_id(path, index, numbering_mode))
        for index, path in enumerate(sorted_paths)
    ]


def perspective_output_name(frame_id: int, cam_idx: int, extension: str) -> str:
    ext = str(extension).lstrip('.')
    return f'frame_{frame_id:05d}_cam_{cam_idx:02d}.{ext}'


def cubemap_output_name(frame_id: int, tile_name: str, extension: str) -> str:
    ext = str(extension).lstrip('.')
    return f'frame_{frame_id:05d}_{tile_name}.{ext}'


def resolve_cubemap_output_path(output_root: str | Path, frame_id: int, tile_name: str, extension: str, layout_mode: str | None) -> Path:
    root = Path(output_root)
    mode = normalize_stage2_layout_mode(layout_mode)
    if mode == 'by_camera':
        return root / str(tile_name) / cubemap_output_name(frame_id, tile_name, extension)
    return root / cubemap_output_name(frame_id, tile_name, extension)


def resolve_perspective_output_path(output_root: str | Path, frame_id: int, cam_idx: int, extension: str, layout_mode: str | None) -> Path:
    root = Path(output_root)
    mode = normalize_stage2_layout_mode(layout_mode)
    if mode == 'by_camera':
        return root / f'cam_{cam_idx:02d}' / perspective_output_name(frame_id, cam_idx, extension)
    return root / perspective_output_name(frame_id, cam_idx, extension)


def parse_perspective_output(path_or_name: str | Path) -> Optional[tuple[int, int]]:
    candidate = Path(path_or_name).name
    match = _PERSPECTIVE_NAME_RE.match(candidate)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_cubemap_output(path_or_name: str | Path) -> Optional[tuple[int, str]]:
    candidate = Path(path_or_name).name
    match = _CUBEMAP_NAME_RE.match(candidate)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def perspective_output_sort_key(path_or_name: str | Path) -> tuple[int, int, int, str]:
    parsed = parse_perspective_output(path_or_name)
    if parsed is not None:
        frame_id, cam_idx = parsed
        return (0, frame_id, cam_idx, Path(path_or_name).as_posix().lower())
    parsed_cubemap = parse_cubemap_output(path_or_name)
    if parsed_cubemap is not None:
        frame_id, tile_name = parsed_cubemap
        return (1, frame_id, 0, f'{tile_name}:{Path(path_or_name).as_posix().lower()}')
    generic_path = Path(path_or_name)
    generic_frame_id = extract_frame_id(generic_path)
    if generic_frame_id is not None:
        return (2, generic_frame_id, 0, generic_path.as_posix().lower())
    return (3, 0, 0, generic_path.as_posix().lower())


def collect_perspective_images(perspective_dir: str | Path) -> list[Path]:
    root = Path(perspective_dir)
    return sorted(
        [path for path in root.rglob('frame_*_cam_*.*') if path.is_file()],
        key=perspective_output_sort_key,
    )