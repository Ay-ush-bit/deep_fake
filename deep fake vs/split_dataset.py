from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_dataset(
    metadata_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")
    metadata = pd.read_csv(metadata_path)
    required = {"file_path", "label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must contain columns: {', '.join(sorted(required))}")
    metadata = metadata[metadata["label"].isin(["real", "fake"])].copy()
    if metadata.empty:
        raise ValueError("Metadata has no verified real/fake labels to split.")
    if "is_valid" in metadata:
        metadata = metadata[metadata["is_valid"].astype(str).str.casefold().isin(["true", "1"])]
    if "is_duplicate" in metadata:
        metadata = metadata[~metadata["is_duplicate"].astype(str).str.casefold().isin(["true", "1"])]
    if metadata.empty:
        raise ValueError("No valid, non-duplicate labeled media remain after filtering.")

    groups = metadata["source_id"].fillna(metadata["file_path"]).astype(str) if "source_id" in metadata else metadata["file_path"].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 3:
        raise ValueError("At least 3 distinct source groups are required for train/validation/test splits.")

    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    train_indices, remainder_indices = next(first_split.split(metadata, groups=groups))
    remainder = metadata.iloc[remainder_indices].copy()
    remainder_groups = groups.iloc[remainder_indices]
    if remainder_groups.nunique() < 2:
        raise ValueError("Not enough independent source groups remain for validation and test splits.")

    relative_val_size = val_size / (test_size + val_size)
    second_split = GroupShuffleSplit(n_splits=1, test_size=1 - relative_val_size, random_state=seed)
    val_relative_indices, test_relative_indices = next(
        second_split.split(remainder, groups=remainder_groups)
    )
    train_df = metadata.iloc[train_indices].copy()
    val_df = remainder.iloc[val_relative_indices].copy()
    test_df = remainder.iloc[test_relative_indices].copy()

    train_groups = set(groups.iloc[train_indices])
    val_groups = set(remainder_groups.iloc[val_relative_indices])
    test_groups = set(remainder_groups.iloc[test_relative_indices])
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise RuntimeError("Grouped split unexpectedly assigned a source group to multiple splits.")

    for split_name, split_df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if set(split_df["label"]) != {"real", "fake"}:
            raise ValueError(
                f"The {split_name} split does not contain both classes. "
                "Add more independent source groups or try a different --seed."
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output / "train.csv", index=False)
    val_df.to_csv(output / "val.csv", index=False)
    test_df.to_csv(output / "test.csv", index=False)
    return {"train": len(train_df), "val": len(val_df), "test": len(test_df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create source-grouped train/validation/test CSV splits.")
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of all rows assigned to test")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of all rows assigned to validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(split_dataset(args.metadata, args.output_dir, args.test_size, args.val_size, args.seed))


if __name__ == "__main__":
    main()
