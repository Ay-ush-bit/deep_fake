from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import soundfile as sf

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3"}
LABEL_NAMES = {
    "real": "real",
    "original": "real",
    "authentic": "real",
    "bonafide": "real",
    "fake": "fake",
    "manipulated": "fake",
    "deepfake": "fake",
    "synthetic": "fake",
}
LOGGER = logging.getLogger(__name__)
SOURCE_GROUP_PATTERN = re.compile(r"^(\d+)(?:_\d+)?__")
METADATA_COLUMNS = [
    "file_path",
    "label",
    "source_id",
    "manipulation_type",
    "modality",
    "file_size_bytes",
    "sha256",
    "is_valid",
    "is_duplicate",
    "duplicate_of",
    "notes",
]


def infer_label_from_path(file_path: Path) -> str:
    """Infer labels only from exact recognized folder names; never substring-match."""
    labels = {
        LABEL_NAMES[component.casefold()]
        for component in file_path.parts[:-1]
        if component.casefold() in LABEL_NAMES
    }
    return labels.pop() if len(labels) == 1 else "unknown"


def _check_media(path: Path) -> tuple[bool, str]:
    if path.stat().st_size == 0:
        return False, "empty file"
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return False, "video could not be opened"
            ok, _ = capture.read()
            if not ok:
                return False, "video has no decodable frames"
        finally:
            capture.release()
    else:
        try:
            sf.info(str(path))
        except (RuntimeError, OSError, ValueError) as exc:
            return False, f"audio could not be decoded: {exc}"
    return True, ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(file_path: Path) -> str:
    """Group media variants sharing a leading numeric identity/source token."""
    match = SOURCE_GROUP_PATTERN.match(file_path.stem)
    if match:
        return f"source_{match.group(1)}"
    return file_path.stem


def build_metadata(raw_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    root = Path(raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        notes: list[str] = []
        try:
            file_hash = _sha256(file_path)
            is_valid, validation_note = _check_media(file_path)
        except OSError as exc:
            LOGGER.warning("Could not inspect %s: %s", file_path, exc)
            file_hash = ""
            is_valid = False
            validation_note = str(exc)

        if validation_note:
            notes.append(validation_note)
        label = infer_label_from_path(file_path.relative_to(root))
        if label == "unknown":
            notes.append("label not inferred; verify labels using the dataset documentation")

        duplicate_of = seen_hashes.get(file_hash) if file_hash else None
        is_duplicate = duplicate_of is not None
        if is_duplicate:
            notes.append("duplicate content")
        elif file_hash:
            seen_hashes[file_hash] = str(file_path)

        records.append(
            {
                "file_path": str(file_path),
                "label": label,
                "source_id": _source_id(file_path),
                "manipulation_type": "unknown",
                "modality": "video" if file_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "audio",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": file_hash,
                "is_valid": is_valid,
                "is_duplicate": is_duplicate,
                "duplicate_of": duplicate_of or "",
                "notes": "; ".join(notes),
            }
        )

    df = pd.DataFrame.from_records(records, columns=METADATA_COLUMNS)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect media and build a conservative metadata CSV.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw files")
    parser.add_argument("--output", type=str, default="data/metadata/metadata.csv", help="Output CSV path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    metadata = build_metadata(args.raw_dir, args.output)
    print(f"Metadata rows: {len(metadata)}")
    if metadata.empty:
        print("No supported media files found.")
    else:
        print(metadata.groupby(["modality", "label", "is_valid"], dropna=False).size().to_string())
        print(f"Duplicates: {int(metadata['is_duplicate'].sum())}")
        print(f"Output: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
