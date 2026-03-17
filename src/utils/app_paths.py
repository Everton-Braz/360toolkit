"""Application-specific runtime and user-data paths."""

from __future__ import annotations

import os
from pathlib import Path


APP_DIRNAME = "360toolkit"


def get_user_data_dir() -> Path:
    """Return the user-writable application data directory."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / APP_DIRNAME

    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_DIRNAME

    return Path.home() / f".{APP_DIRNAME}"


def ensure_user_data_dir() -> Path:
    """Create and return the user data directory."""
    user_data_dir = get_user_data_dir()
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


def get_settings_file_path() -> Path:
    """Return the canonical settings file path."""
    return ensure_user_data_dir() / "settings.json"


def get_log_file_path() -> Path:
    """Return the canonical log file path."""
    return ensure_user_data_dir() / "360toolkit.log"