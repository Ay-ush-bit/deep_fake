from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: str | Path, output_dir: str | Path, frames_per_second: int = 1, max_frames: int = 20) -> list[str]:
    video_file = Path(video_path)
    if frames_per_second <= 0 or max_frames <= 0:
        raise ValueError("frames_per_second and max_frames must be positive integers.")
    if not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_file}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS) or frames_per_second
    frame_interval = max(1, int(round(fps / max(frames_per_second, 1))))
    saved_paths: list[str] = []
    frame_index = 0
    saved_count = 0

    try:
        while saved_count < max_frames:
            success, frame = cap.read()
            if not success:
                break
            if frame_index % frame_interval == 0:
                frame_path = out_dir / f"frame_{saved_count:04d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Could not write extracted frame: {frame_path}")
                saved_paths.append(str(frame_path))
                saved_count += 1
            frame_index += 1
    finally:
        cap.release()
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sampled frames from a video.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/frames")
    parser.add_argument("--frames-per-second", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=20)
    args = parser.parse_args()
    paths = extract_frames(args.video, args.output_dir, args.frames_per_second, args.max_frames)
    print(paths)


if __name__ == "__main__":
    main()
