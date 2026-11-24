from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoModel, Wav2Vec2Config, Wav2Vec2Model


@dataclass
class AudioModelConfig:
    model_name: str = "facebook/wav2vec2-base"
    num_classes: int = 2
    pretrained: bool = False


class AudioDeepfakeModel(nn.Module):
    """Wav2Vec2 encoder with a binary head; it is not trained by default."""

    def __init__(self, config: AudioModelConfig | None = None):
        super().__init__()
        self.config = config or AudioModelConfig()
        if self.config.pretrained:
            self.encoder = AutoModel.from_pretrained(self.config.model_name)
        else:
            model_config = Wav2Vec2Config()
            self.encoder = Wav2Vec2Model(model_config)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, self.config.num_classes)
        self.trained = False

    def forward(self, waveforms: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if waveforms.ndim != 2:
            raise ValueError("Audio model input must have shape (batch, samples).")
        encoded = self.encoder(input_values=waveforms, attention_mask=attention_mask).last_hidden_state
        return self.classifier(encoded.mean(dim=1))

    @torch.inference_mode()
    def predict_proba(self, waveforms: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if not self.trained:
            raise RuntimeError("Audio model has no validated trained checkpoint; prediction is unavailable.")
        self.eval()
        return torch.softmax(self(waveforms, attention_mask), dim=1)

    def save(self, output_path: str | Path) -> Path:
        if not self.trained:
            raise RuntimeError("Refusing to save untrained audio model weights as a detector checkpoint.")
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
            raise ValueError("Audio checkpoint is missing validated training metadata.")
        if payload.get("labels") != ["real", "fake"] or "state_dict" not in payload:
            raise ValueError("Audio checkpoint has an unsupported label mapping or format.")
        self.load_state_dict(payload["state_dict"])
        self.trained = True
