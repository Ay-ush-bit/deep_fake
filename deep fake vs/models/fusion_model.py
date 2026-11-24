from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


@dataclass
class FusionConfig:
    model_type: str = "logistic_regression"
    use_standard_scaler: bool = True
    missing_modality_strategy: str = "reject"


class FusionClassifier:
    """Logistic-regression fusion over real, training-set branch predictions."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        steps = []
        if self.config.use_standard_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", LogisticRegression(max_iter=500, random_state=42)))
        self.model = Pipeline(steps)

    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Fusion features must have shape (n_samples, 3): visual, audio, sync.")
        if not np.isfinite(values).all():
            raise ValueError("Fusion features contain missing or non-finite modality scores.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Fusion modality scores must be between 0 and 1.")
        return values

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FusionClassifier":
        features = self._validate_features(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("Fusion labels must be a one-dimensional array matching X.")
        if len(np.unique(labels)) != 2:
            raise ValueError("Fusion training requires examples from both classes.")
        self.model.fit(features, labels)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = self._validate_features(X)
        check_is_fitted(self.model)
        return self.model.predict_proba(features)

    def save(self, output_path: str | Path) -> Path:
        try:
            check_is_fitted(self.model)
        except NotFittedError as exc:
            raise RuntimeError("Refusing to save an untrained fusion model.") from exc
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        classifier = self.model.named_steps["classifier"]
        payload = {
            "coef": classifier.coef_,
            "intercept": classifier.intercept_,
            "classes": classifier.classes_,
            "n_iter": classifier.n_iter_,
            "n_features": np.asarray([classifier.n_features_in_], dtype=np.int64),
            "use_standard_scaler": np.asarray([self.config.use_standard_scaler], dtype=np.bool_),
        }
        if self.config.use_standard_scaler:
            scaler = self.model.named_steps["scaler"]
            payload.update(
                {
                    "scale_mean": scaler.mean_,
                    "scale_var": scaler.var_,
                    "scale_scale": scaler.scale_,
                    "scale_samples": np.asarray([scaler.n_samples_seen_]),
                }
            )
        with out.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        return out

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as payload:
            use_scaler = bool(payload["use_standard_scaler"][0])
            self.config = FusionConfig(use_standard_scaler=use_scaler)
            self.__init__(self.config)
            classifier = self.model.named_steps["classifier"]
            classifier.coef_ = payload["coef"].copy()
            classifier.intercept_ = payload["intercept"].copy()
            classifier.classes_ = payload["classes"].copy()
            classifier.n_iter_ = payload["n_iter"].copy()
            classifier.n_features_in_ = int(payload["n_features"][0])
            if use_scaler:
                scaler = self.model.named_steps["scaler"]
                scaler.mean_ = payload["scale_mean"].copy()
                scaler.var_ = payload["scale_var"].copy()
                scaler.scale_ = payload["scale_scale"].copy()
                scaler.n_samples_seen_ = payload["scale_samples"][0]
