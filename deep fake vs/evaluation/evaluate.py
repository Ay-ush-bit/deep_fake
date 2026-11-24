from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import binary_metrics


def _encode_labels(series: pd.Series, column_name: str) -> pd.Series:
    encoded = series.astype(str).str.strip().str.casefold().map({"real": 0, "fake": 1})
    if encoded.isna().any():
        invalid = sorted(series[encoded.isna()].astype(str).unique().tolist())
        raise ValueError(f"{column_name} contains unsupported labels: {invalid}")
    return encoded


def evaluate_predictions(
    labels_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
) -> dict:
    labels_df = pd.read_csv(labels_path)
    preds_df = pd.read_csv(predictions_path)
    if "label" not in labels_df.columns or "prediction" not in preds_df.columns:
        raise KeyError("Expected columns: label in metadata and prediction in the predictions CSV.")
    if len(labels_df) != len(preds_df):
        raise ValueError(
            f"Row count mismatch: {len(labels_df)} labels and {len(preds_df)} predictions."
        )
    true_labels = _encode_labels(labels_df["label"], "label")
    predicted_labels = _encode_labels(preds_df["prediction"], "prediction")
    scores = preds_df["score"] if "score" in preds_df.columns else None
    metrics = binary_metrics(true_labels, predicted_labels, scores)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary predictions against labeled metadata.")
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata.csv")
    parser.add_argument("--predictions", type=str, default="data/reports/predictions.csv")
    parser.add_argument("--output", type=str, default="data/reports/evaluation_report.json")
    args = parser.parse_args()
    metrics = evaluate_predictions(args.metadata, args.predictions, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
