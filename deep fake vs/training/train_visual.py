from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the visual deepfake branch.")
    parser.add_argument("--metadata", default="data/metadata/metadata.csv")
    parser.add_argument("--output-dir", default="data/models")
    parser.parse_args()
    parser.error(
        "visual training is not implemented: this project has no image dataset loader or validated "
        "training loop. No checkpoint was created. See README.md."
    )


if __name__ == "__main__":
    main()
