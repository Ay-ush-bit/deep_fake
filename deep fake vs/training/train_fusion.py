from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal fusion from out-of-fold branch predictions.")
    parser.add_argument("--metadata", default="data/metadata/metadata.csv")
    parser.add_argument("--output-dir", default="data/models")
    parser.parse_args()
    parser.error(
        "fusion training is not implemented: out-of-fold visual/audio/synchronization predictions are "
        "required to avoid data leakage. No checkpoint was created. See README.md."
    )


if __name__ == "__main__":
    main()
