from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


def extract_mfcc(audio_path: str | Path, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if librosa is None:
        raise RuntimeError("Librosa is required for audio feature extraction. Install project requirements.")
    if sample_rate <= 0 or n_mfcc <= 0:
        raise ValueError("sample_rate and n_mfcc must be positive integers.")
    try:
        signal, sr = librosa.load(str(audio_file), sr=sample_rate, mono=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Could not decode audio file {audio_file}: {exc}") from exc
    if signal.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_file}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Audio contains non-finite samples: {audio_file}")
    if float(np.max(np.abs(signal))) < 1e-6:
        raise ValueError(f"Audio is silent: {audio_file}")
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


def summarize_audio_features(audio_path: str | Path, sample_rate: int = 16000) -> dict:
    mfcc = extract_mfcc(audio_path, sample_rate)
    return {
        "shape": list(mfcc.shape),
        "mean": float(np.mean(mfcc)),
        "std": float(np.std(mfcc)),
        "max": float(np.max(mfcc)),
    }
