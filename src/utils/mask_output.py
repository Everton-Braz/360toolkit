from __future__ import annotations

from pathlib import Path


def build_related_output_path(
    input_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    suffix: str,
) -> Path:
    source = Path(input_path)
    root = Path(input_root)
    destination_root = Path(output_root)
    try:
        relative = source.relative_to(root)
    except Exception:
        relative = Path(source.name)
    return destination_root / relative.parent / f"{relative.stem}{suffix}"


def build_unique_staged_filename(input_path: str | Path, input_root: str | Path) -> str:
    source = Path(input_path)
    root = Path(input_root)
    try:
        relative = source.relative_to(root)
    except Exception:
        relative = Path(source.name)

    parts = [part for part in relative.parts[:-1] if part not in {'', '.', '..'}]
    safe_prefix = '__'.join(part.replace(' ', '_') for part in parts)
    stem = relative.stem.replace(' ', '_')
    if safe_prefix:
        return f"{safe_prefix}__{stem}{source.suffix.lower() or relative.suffix.lower()}"
    return f"{stem}{source.suffix.lower() or relative.suffix.lower()}"