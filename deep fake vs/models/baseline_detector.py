from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.multimodal_features import (
    AUDIO_FEATURE_COUNT,
    FEATURE_COUNT,
    FEATURE_VERSION,
    VISUAL_FEATURE_COUNT,
)

MODEL_FORMAT_VERSION = 1


def create_baseline_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def save_baseline_model(model: Pipeline, output_path: str | Path) -> Path:
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "classes_"):
        raise RuntimeError("Refusing to save an unfitted baseline detector.")
    if set(classifier.classes_.tolist()) != {0, 1}:
        raise ValueError("Baseline model must be trained on both real (0) and fake (1) classes.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_count": FEATURE_COUNT,
            "labels": {"real": 0, "fake": 1},
            "model": model,
        },
        path,
    )
    return path


def load_baseline_model(checkpoint_path: str | Path) -> Pipeline:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained detector checkpoint not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("Detector checkpoint has an invalid format.")
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Detector checkpoint format version is not supported.")
    if payload.get("feature_version") != FEATURE_VERSION or payload.get("feature_count") != FEATURE_COUNT:
        raise ValueError("Detector checkpoint was trained with an incompatible feature definition.")
    if payload.get("labels") != {"real": 0, "fake": 1}:
        raise ValueError("Detector checkpoint has an unsupported label mapping.")
    model = payload.get("model")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Detector checkpoint does not contain a probability classifier.")
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "classes_") or set(classifier.classes_.tolist()) != {0, 1}:
        raise ValueError("Detector checkpoint is not fitted for both classes.")
    return model


def model_supports_audio(model: Pipeline) -> bool:
    scaler = model.named_steps.get("scaler")
    if scaler is None or not hasattr(scaler, "mean_"):
        raise ValueError("Detector checkpoint does not contain a fitted feature scaler.")
    audio_presence_index = VISUAL_FEATURE_COUNT + AUDIO_FEATURE_COUNT - 1
    return bool(scaler.mean_[audio_presence_index] > 0)


def predict_features(model: Pipeline, features: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (1, FEATURE_COUNT) or not np.isfinite(values).all():
        raise ValueError(f"Expected one finite feature vector of length {FEATURE_COUNT}.")
    probabilities = model.predict_proba(values)[0]
    classes = model.named_steps["classifier"].classes_.tolist()
    fake_probability = float(probabilities[classes.index(1)])
    label = "fake" if fake_probability >= 0.5 else "real"
    return {
        "prediction": label,
        "fake_probability": fake_probability,
        "confidence": max(fake_probability, 1.0 - fake_probability),
    }
