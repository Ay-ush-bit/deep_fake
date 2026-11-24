from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import av
import cv2
import librosa
import numpy as np

FEATURE_VERSION = 1
VISUAL_FEATURE_COUNT = 26
AUDIO_FEATURE_COUNT = 33
FEATURE_COUNT = VISUAL_FEATURE_COUNT + AUDIO_FEATURE_COUNT + 1
MAX_AUDIO_SECONDS = 60


@lru_cache(maxsize=1)
def _face_cascade() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("OpenCV could not load its frontal-face cascade classifier.")
    return cascade


def _sample_video_frames(video_path: str | Path, max_frames: int = 12) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(fps)) if fps > 0 else 1
    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_interval == 0:
                frames.append(frame)
            frame_index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decodable video frames found: {video_path}")
    return frames


def _frame_descriptor(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, 256 / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    faces = _face_cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
    largest_face_fraction = 0.0
    if len(faces):
        face_area = max(int(face_width) * int(face_height) for _, _, face_width, face_height in faces)
        largest_face_fraction = face_area / float(gray.shape[0] * gray.shape[1])

    descriptor = np.concatenate(
        [
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            np.asarray([gray_float.mean(), gray_float.std()]),
            np.asarray([gradient.mean(), gradient.std()]),
            np.asarray([min(laplacian_variance / 1000.0, 1.0)]),
            np.asarray([1.0 if len(faces) else 0.0, largest_face_fraction]),
        ]
    )
    if descriptor.shape != (13,):
        raise RuntimeError(f"Unexpected visual descriptor size: {descriptor.shape}")
    return descriptor.astype(np.float32)


def _video_features(video_path: str | Path) -> tuple[np.ndarray, bool]:
    descriptors = np.stack([_frame_descriptor(frame) for frame in _sample_video_frames(video_path)])
    features = np.concatenate([descriptors.mean(axis=0), descriptors.std(axis=0)])
    return features.astype(np.float32), True


def _audio_samples(video_path: str | Path) -> np.ndarray | None:
    try:
        with av.open(str(video_path)) as container:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if not audio_streams:
                return None
            sample_rate = 16000
            max_samples = sample_rate * MAX_AUDIO_SECONDS
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            chunks: list[np.ndarray] = []
            total_samples = 0
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    samples = resampled.to_ndarray().reshape(-1)
                    remaining = max_samples - total_samples
                    if remaining <= 0:
                        break
                    samples = samples[:remaining]
                    chunks.append(samples.astype(np.float32) / 32768.0)
                    total_samples += len(samples)
                if total_samples >= max_samples:
                    break
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise ValueError(f"Could not decode audio from video {video_path}: {exc}") from exc
    if not chunks:
        return None
    waveform = np.concatenate(chunks)
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio contains non-finite samples: {video_path}")
    if float(np.max(np.abs(waveform))) < 1e-6:
        return None
    return waveform


def _audio_features(video_path: str | Path) -> tuple[np.ndarray, bool]:
    waveform = _audio_samples(video_path)
    if waveform is None:
        return np.zeros(AUDIO_FEATURE_COUNT, dtype=np.float32), False

    sample_rate = 16000
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    rms = librosa.feature.rms(y=waveform)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=waveform)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
    stats = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            np.asarray([rms.mean(), rms.std()]),
            np.asarray([zero_crossing_rate.mean(), zero_crossing_rate.std()]),
            np.asarray([spectral_centroid.mean() / sample_rate, spectral_centroid.std() / sample_rate]),
            np.asarray([1.0]),
        ]
    )
    if stats.shape != (AUDIO_FEATURE_COUNT,):
        raise RuntimeError(f"Unexpected audio feature size: {stats.shape}")
    return stats.astype(np.float32), True


def extract_multimodal_features(video_path: str | Path) -> tuple[np.ndarray, dict[str, bool]]:
    """Extract fixed-size visual/audio descriptors and explicit modality-presence flags."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    visual, visual_available = _video_features(path)
    audio, audio_available = _audio_features(path)
    visual_presence = np.asarray([1.0 if visual_available else 0.0], dtype=np.float32)
    features = np.concatenate([visual, audio, visual_presence])
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise RuntimeError("Multimodal feature extraction produced invalid output.")
    return features.astype(np.float32), {
        "visual": visual_available,
        "audio": audio_available,
    }
