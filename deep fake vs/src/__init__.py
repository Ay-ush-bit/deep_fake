"""Multimodal deepfake detection project package."""

from .config import get_project_paths, load_config
from .utils import ensure_directories, set_seed

__all__ = ["load_config", "get_project_paths", "ensure_directories", "set_seed"]
