from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_ID = "angads24/deepfake-video"
DATASET_REVISION = "3470af66b50e6d715f108bdfa67342907137e1f4"


def download_dataset(output_dir: str | Path, accept_terms: bool = False) -> Path:
    """Download the pinned public dataset after explicit acknowledgement of its limitations."""
    if not accept_terms:
        raise ValueError(
            "Review the dataset card and CC BY-SA 4.0 terms, including the undocumented source-media "
            "provenance, then pass --accept-cc-by-sa-and-provenance-risk."
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(destination),
        allow_patterns=["dataset/real/*", "dataset/fake/*"],
        max_workers=2,
    )
    return Path(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the pinned Hugging Face video dataset for experimental research."
    )
    parser.add_argument("--output-dir", default="data/raw/angads24-deepfake-video")
    parser.add_argument(
        "--accept-cc-by-sa-and-provenance-risk",
        action="store_true",
        help=(
            "Acknowledge CC BY-SA 4.0 attribution/share-alike terms and that the repository "
            "does not document source-media provenance."
        ),
    )
    args = parser.parse_args()
    if not args.accept_cc_by_sa_and_provenance_risk:
        parser.error(
            "review the CC BY-SA 4.0 terms and undocumented source provenance, then pass "
            "--accept-cc-by-sa-and-provenance-risk"
        )
    result = download_dataset(args.output_dir, args.accept_cc_by_sa_and_provenance_risk)
    print(f"Dataset downloaded to: {result.resolve()}")


if __name__ == "__main__":
    main()
