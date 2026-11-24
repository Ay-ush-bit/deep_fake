from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision import models


@dataclass
class VisualModelConfig:
    model_name: str = "efficientnet_b0"
    num_classes: int = 2
    pretrained: bool = False


class VisualDeepfakeModel(nn.Module):
    """EfficientNet-B0 binary classifier architecture; it is not trained by default."""

    def __init__(self, config: VisualModelConfig | None = None):
        super().__init__()
        self.config = config or VisualModelConfig()
        weights = models.EfficientNet_B0_Weights.DEFAULT if self.config.pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, self.config.num_classes
        )
        self.trained = False

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Visual model input must have shape (batch, 3, height, width).")
        return self.model(images)

    @torch.inference_mode()
    def predict_proba(self, images: Tensor) -> Tensor:
        if not self.trained:
            raise RuntimeError("Visual model has no validated trained checkpoint; prediction is unavailable.")
        self.eval()
        return torch.softmax(self(images), dim=1)

    def save(self, output_path: str | Path) -> Path:
        if not self.trained:
            raise RuntimeError("Refusing to save untrained visual model weights as a detector checkpoint.")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"trained": True, "labels": ["real", "fake"], "state_dict": self.state_dict()},
            out,
        )
        return out

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("trained") is not True:
            raise ValueError("Visual checkpoint is missing validated training metadata.")
        if payload.get("labels") != ["real", "fake"] or "state_dict" not in payload:
            raise ValueError("Visual checkpoint has an unsupported label mapping or format.")
        self.load_state_dict(payload["state_dict"])
        self.trained = True
