from __future__ import annotations

import argparse
from pathlib import Path


def inspect_dataset(dataset_dir: str | Path) -> dict:
    dataset_path = Path(dataset_dir)
    labels = {}
    files = []
    if dataset_path.exists():
        for child in sorted(dataset_path.rglob("*")):
            if child.is_file():
                files.append(str(child.relative_to(dataset_path)))
                if child.parent.name not in labels:
                    labels[child.parent.name] = 0
                labels[child.parent.name] += 1
    summary = {
        "dataset_dir": str(dataset_path),
        "total_files": len(files),
        "labels": labels,
        "file_examples": files[:10],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the dataset directory.")
    parser.add_argument("--dataset-dir", type=str, default="data/raw", help="Directory containing raw media files")
    args = parser.parse_args()
    summary = inspect_dataset(args.dataset_dir)
    print(summary)


if __name__ == "__main__":
    main()
