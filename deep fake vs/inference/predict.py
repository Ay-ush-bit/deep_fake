from __future__ import annotations

from pathlib import Path
from typing import Any

from models.baseline_detector import load_baseline_model, model_supports_audio, predict_features
from src.multimodal_features import (
    AUDIO_FEATURE_COUNT,
    VISUAL_FEATURE_COUNT,
    extract_multimodal_features,
)


class InferenceNotReadyError(RuntimeError):
    """Raised when no actual trained baseline detector is available."""


def predict_video(video_path: str | Path, model_dir: str | Path = "data/models") -> dict[str, Any]:
    """Run the trained visual/audio feature baseline; never synthesize scores."""
    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"Video not found: {video_file}")

    checkpoint = Path(model_dir) / "multimodal_baseline.joblib"
    if not checkpoint.is_file():
        raise InferenceNotReadyError(
            "No trained multimodal baseline is available. Train it with "
            "`python -m training.train_detector --train-metadata data/processed/train.csv "
            "--validation-metadata data/processed/val.csv`."
        )

    model = load_baseline_model(checkpoint)
    features, modalities = extract_multimodal_features(video_file)
    audio_supported = model_supports_audio(model)
    if not audio_supported:
        features[VISUAL_FEATURE_COUNT : VISUAL_FEATURE_COUNT + AUDIO_FEATURE_COUNT] = 0
        modalities["audio"] = False
    prediction = predict_features(model, features)
    warnings = [
        "This handcrafted-feature logistic-regression baseline is for research evaluation, not forensic proof.",
        "Confidence is the classifier's probability estimate and is not a guarantee of correctness.",
    ]
    if not modalities["audio"]:
        if audio_supported:
            warnings.append("No usable audio was found; the model used visual features only.")
        else:
            warnings.append(
                "The training data contained no audio; this checkpoint is visual-only and ignores uploaded audio."
            )

    return {
        "status": "available",
        "model": (
            "handcrafted visual and MFCC features with logistic regression"
            if audio_supported
            else "handcrafted visual-only features with logistic regression"
        ),
        **prediction,
        "modalities_used": [name for name, available in modalities.items() if available],
        "warnings": warnings,
    }
