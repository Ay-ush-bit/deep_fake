from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from models.baseline_detector import (
    create_baseline_model,
    model_supports_audio,
    save_baseline_model,
)
from src.multimodal_features import (
    AUDIO_FEATURE_COUNT,
    VISUAL_FEATURE_COUNT,
    extract_multimodal_features,
)

LOGGER = logging.getLogger(__name__)
LABEL_MAP = {"real": 0, "fake": 1}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _load_labeled_videos(metadata_path: str | Path) -> pd.DataFrame:
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    metadata = pd.read_csv(path)
    if not {"file_path", "label"}.issubset(metadata.columns):
        raise ValueError("Metadata must contain file_path and label columns.")

    labels = metadata["label"].astype(str).str.strip().str.casefold()
    videos = metadata[
        labels.isin(LABEL_MAP) & metadata["file_path"].astype(str).map(lambda value: Path(value).suffix.casefold() in VIDEO_EXTENSIONS)
    ].copy()
    videos["label"] = labels.loc[videos.index].map(LABEL_MAP)

    if "is_valid" in videos:
        videos = videos[videos["is_valid"].astype(str).str.casefold().isin({"true", "1"})]
    if "is_duplicate" in videos:
        videos = videos[~videos["is_duplicate"].astype(str).str.casefold().isin({"true", "1"})]
    videos = videos[videos["file_path"].map(lambda value: Path(value).is_file())]
    if videos.empty:
        raise ValueError("No existing, valid, non-duplicate videos with verified real/fake labels were found.")
    if set(videos["label"].unique()) != {0, 1}:
        raise ValueError("Training requires both real and fake videos.")
    return videos


def _source_groups(metadata: pd.DataFrame) -> pd.Series:
    if "source_id" in metadata:
        groups = metadata["source_id"].fillna("").astype(str)
        return groups.where(groups.str.len() > 0, metadata["file_path"].astype(str))
    return metadata["file_path"].astype(str)


def extract_dataset_features(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in metadata.itertuples(index=False):
        media_path = Path(row.file_path)
        LOGGER.info("Extracting features from %s", media_path)
        vector, _ = extract_multimodal_features(media_path)
        features.append(vector)
        labels.append(int(row.label))
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def _evaluate(model: Pipeline, metadata: pd.DataFrame) -> dict[str, Any]:
    X, y_true = extract_dataset_features(metadata)
    if not model_supports_audio(model):
        X[:, VISUAL_FEATURE_COUNT : VISUAL_FEATURE_COUNT + AUDIO_FEATURE_COUNT] = 0
    probabilities = model.predict_proba(X)[:, list(model.named_steps["classifier"].classes_).index(1)]
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "note": "Metrics are measured only on the supplied validation CSV and are not a guarantee of real-world performance.",
    }


def train_detector(
    train_metadata_path: str | Path,
    output_checkpoint: str | Path,
    validation_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    training_metadata = _load_labeled_videos(train_metadata_path)
    X_train, y_train = extract_dataset_features(training_metadata)
    model = create_baseline_model()
    model.fit(X_train, y_train)
    audio_supported = model_supports_audio(model)

    validation_metrics = None
    if validation_metadata_path is not None:
        validation_metadata = _load_labeled_videos(validation_metadata_path)
        train_groups = set(_source_groups(training_metadata))
        validation_groups = set(_source_groups(validation_metadata))
        overlap = train_groups & validation_groups
        if overlap:
            raise ValueError(
                f"Training and validation source groups overlap ({len(overlap)} groups). "
                "Regenerate group-disjoint splits to avoid leakage."
            )
        validation_metrics = _evaluate(model, validation_metadata)

    checkpoint_path = save_baseline_model(model, output_checkpoint)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "training_samples": len(training_metadata),
        "training_class_counts": {
            "real": int((y_train == 0).sum()),
            "fake": int((y_train == 1).sum()),
        },
        "validation": validation_metrics,
        "model": (
            "handcrafted visual and MFCC statistics + logistic regression"
            if audio_supported
            else "handcrafted visual-only statistics + logistic regression"
        ),
        "warning": (
            "This baseline must be evaluated on an appropriate licensed dataset. "
            "It is a research baseline, not a forensic guarantee."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a real multimodal baseline from verified labeled videos."
    )
    parser.add_argument("--train-metadata", default="data/processed/train.csv")
    parser.add_argument("--validation-metadata", default="data/processed/val.csv")
    parser.add_argument("--output", default="data/models/multimodal_baseline.joblib")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    validation_path = None if args.no_validation else args.validation_metadata
    result = train_detector(args.train_metadata, args.output, validation_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
