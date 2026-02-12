"""Build a complete QSS stylesheet from tokenized component partials."""

from __future__ import annotations

from pathlib import Path

from .tokens import get_theme_tokens

_COMPONENT_ORDER = [
    "base.qss",
    "nav.qss",
    "cards.qss",
    "inputs.qss",
    "buttons.qss",
    "log_panel.qss",
]


def _flatten_tokens(theme_tokens: dict) -> dict[str, str]:
    flat: dict[str, str] = {}
    for group_name in ("spacing", "radius", "font", "control", "color"):
        for key, value in theme_tokens[group_name].items():
            flat[f"{group_name}.{key}"] = str(value)
    return flat


def _render_template(raw: str, flat_tokens: dict[str, str]) -> str:
    rendered = raw
    for token_key, token_value in flat_tokens.items():
        rendered = rendered.replace(f"{{{{{token_key}}}}}", token_value)
    return rendered


def build_theme_stylesheet(theme: str = "dark") -> str:
    """Build full stylesheet by merging and rendering component QSS files."""
    theme_tokens = get_theme_tokens(theme)
    flat_tokens = _flatten_tokens(theme_tokens)

    styles_dir = Path(__file__).resolve().parent
    components_dir = styles_dir / "components"

    chunks: list[str] = []
    for file_name in _COMPONENT_ORDER:
        file_path = components_dir / file_name
        if not file_path.exists():
            continue
        raw_qss = file_path.read_text(encoding="utf-8")
        chunks.append(_render_template(raw_qss, flat_tokens))

    return "\n\n".join(chunks).strip() + "\n"
