from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyncConfig:
    use_optional_syncnet: bool = False
    checkpoint_path: str | None = None


class SynchronizationAnalyzer:
    """Availability interface for an optional, externally supplied SyncNet implementation."""

    def __init__(self, config: SyncConfig | None = None):
        self.config = config or SyncConfig()
        self.available = False

    def score(self, visual_features, audio_features) -> float | None:
        if self.config.use_optional_syncnet:
            checkpoint = Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
            detail = f" Checkpoint path: {checkpoint}." if checkpoint else ""
            raise RuntimeError(
                "SyncNet scoring is not implemented in this project; no synchronization score was generated."
                + detail
            )
        self.available = False
        return None

    def save(self, output_path: str | Path) -> Path:
        raise RuntimeError("Cannot save synchronization weights: no SyncNet model is implemented.")
