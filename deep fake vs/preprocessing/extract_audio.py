from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import shutil


def extract_audio(video_path: str | Path, output_path: str | Path, sample_rate: int = 16000) -> str:
    video_file = Path(video_path)
    audio_file = Path(output_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_file}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and add it to PATH.")
    audio_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-n",
        "-nostdin",
        "-i",
        str(video_file),
        "-vn",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(audio_file),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("FFmpeg audio extraction timed out after 5 minutes.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr.strip()[-1000:]}")
    if not audio_file.is_file() or audio_file.stat().st_size == 0:
        raise RuntimeError("FFmpeg completed without producing an audio file.")
    return str(audio_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio track from a video file.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/audio.wav")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    print(extract_audio(args.video, args.output, args.sample_rate))


if __name__ == "__main__":
    main()
