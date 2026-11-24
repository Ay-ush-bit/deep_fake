from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(*paths: str | Path) -> list[Path]:
    """Create directory paths if missing and return them."""
    created: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def valid_media_file(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.is_file() and file_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}


def dump_json(path: str | Path, payload: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return file_path
