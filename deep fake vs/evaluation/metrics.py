from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
) -> dict:
    labels = np.asarray(list(y_true))
    predictions = np.asarray(list(y_pred))
    if labels.ndim != 1 or predictions.ndim != 1 or len(labels) != len(predictions):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length.")
    if not len(labels):
        raise ValueError("Metrics cannot be computed for empty inputs.")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Binary labels and predictions must contain only 0 or 1.")

    scores = None if y_score is None else np.asarray(list(y_score), dtype=float)
    if scores is not None:
        if scores.ndim != 1 or len(scores) != len(labels):
            raise ValueError("y_score must be one-dimensional and have the same length as y_true.")
        if not np.isfinite(scores).all():
            raise ValueError("y_score contains non-finite values.")

    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if scores is not None and len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    return result
