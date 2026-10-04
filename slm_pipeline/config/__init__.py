"""Portable configuration with explicit environment overrides and no model downloads."""
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("settings.yaml")


def _merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def load_config(path=None) -> dict[str, Any]:
    """Relative paths resolve against the repository, never the calling directory.

    SLM_CONFIG merges a YAML override with defaults. SLM_DATA_ROOT relocates all
    generated data; SLM_<PATH_KEY> (e.g. SLM_RAW_DIR) overrides a single path.
    """
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    override = path or os.environ.get("SLM_CONFIG")
    if override:
        with Path(override).expanduser().open(encoding="utf-8") as stream:
            _merge(result, yaml.safe_load(stream) or {})
    data_root = os.environ.get("SLM_DATA_ROOT")
    for key, value in result["paths"].items():
        if data_root:
            value = str(Path(data_root) / key.removesuffix("_dir"))
        value = os.environ.get("SLM_" + key.upper(), value)
        target = Path(os.path.expandvars(value)).expanduser()
        result["paths"][key] = str((PROJECT_ROOT / target).resolve())
    if os.environ.get("SLM_ALLOW_DOWNLOADS") == "1":
        result["pipeline"]["allow_model_downloads"] = True
    if result["pipeline"]["frame_sampling_fps"] <= 0:
        raise ValueError("frame_sampling_fps must be positive")
    return result


config = load_config()
