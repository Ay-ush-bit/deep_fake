from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load project config from YAML or return defaults."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def get_project_paths(config: Dict[str, Any] | None = None) -> Dict[str, Path]:
    """Resolve key directories for the project."""
    cfg = load_config() if config is None else config
    paths = cfg.get("paths", {})
    data_root = (ROOT / paths.get("data_root", "data")).resolve()

    def resolve_path(name: str, default: str) -> Path:
        configured = Path(paths.get(name, str(data_root / default)))
        if not configured.is_absolute():
            configured = ROOT / configured
        return configured.resolve()

    return {
        "root": ROOT,
        "data_root": data_root,
        "raw_dir": resolve_path("raw_dir", "raw"),
        "processed_dir": resolve_path("processed_dir", "processed"),
        "metadata_dir": resolve_path("metadata_dir", "metadata"),
        "models_dir": resolve_path("models_dir", "models"),
        "reports_dir": resolve_path("reports_dir", "reports"),
    }
