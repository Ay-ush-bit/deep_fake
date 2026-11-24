from __future__ import annotations

import math
import wave
from pathlib import Path

import cv2
import av
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from build_metadata import build_metadata, infer_label_from_path
from app import extract_audio_preview, inspect_video, make_report
from evaluation.evaluate import evaluate_predictions
from evaluation.metrics import binary_metrics
from inference.predict import InferenceNotReadyError, predict_video
from models.baseline_detector import (
    create_baseline_model,
    load_baseline_model,
    model_supports_audio,
)
from models.fusion_model import FusionClassifier
from models.visual_model import VisualDeepfakeModel
from preprocessing.audio_features import extract_mfcc, summarize_audio_features
from preprocessing.extract_frames import extract_frames
from split_dataset import split_dataset
from src.config import get_project_paths
from src.multimodal_features import (
    AUDIO_FEATURE_COUNT,
    FEATURE_COUNT,
    VISUAL_FEATURE_COUNT,
    extract_multimodal_features,
)
from src.utils import valid_media_file
from training.train_detector import train_detector


def _write_tone(path: Path, frequency: int = 440) -> None:
    sample_rate = 16000
    duration = 0.5
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * math.pi * frequency * time)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((signal * 32767).astype(np.int16).tobytes())


def _write_video(path: Path, color: tuple[int, int, int] | None = None) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is not available in this environment.")
    for index in range(12):
        value = index * 10
        frame_color = color if color is not None else (value, value, value)
        frame = np.full((64, 64, 3), frame_color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_audio_video(path: Path) -> None:
    container = av.open(str(path), mode="w")
    video_stream = container.add_stream("mpeg4", rate=10)
    video_stream.width = 64
    video_stream.height = 64
    video_stream.pix_fmt = "yuv420p"
    audio_stream = container.add_stream("aac", rate=16000)
    audio_stream.layout = "mono"
    audio_stream.codec_context.sample_rate = 16000

    for index in range(12):
        image = np.full((64, 64, 3), index * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in video_stream.encode(frame):
            container.mux(packet)
    for packet in video_stream.encode():
        container.mux(packet)

    time = np.arange(16000, dtype=np.float32) / 16000
    samples = (0.2 * np.sin(2 * math.pi * 440 * time)).astype(np.float32).reshape(1, -1)
    audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
    audio_frame.sample_rate = 16000
    for packet in audio_stream.encode(audio_frame):
        container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)
    container.close()


def test_metadata_build_records_labels_validity_and_duplicates(tmp_path):
    raw_dir = tmp_path / "raw"
    real_dir = raw_dir / "real"
    fake_dir = raw_dir / "fake"
    unknown_dir = raw_dir / "notfake"
    for directory in (real_dir, fake_dir, unknown_dir):
        directory.mkdir(parents=True)

    _write_tone(real_dir / "original.wav")
    (fake_dir / "copy.wav").write_bytes((real_dir / "original.wav").read_bytes())
    _write_tone(unknown_dir / "unknown.wav", 880)
    (fake_dir / "corrupt.wav").write_bytes(b"not audio")

    output = tmp_path / "metadata.csv"
    metadata = build_metadata(raw_dir, output)

    assert len(metadata) == 4
    assert output.is_file()
    assert infer_label_from_path(Path("unrelated/fakeproof/clip.mp4")) == "unknown"
    assert set(metadata["label"]) == {"real", "fake", "unknown"}
    assert _source_id_for_test(Path("real/01__clip.mp4")) == _source_id_for_test(
        Path("fake/01_02__clip__variant.mp4")
    )
    duplicate = metadata.loc[metadata["is_duplicate"]].iloc[0]
    assert duplicate["duplicate_of"]
    corrupt = metadata.loc[metadata["file_path"].str.endswith("corrupt.wav")].iloc[0]
    assert not corrupt["is_valid"]


def _source_id_for_test(path: Path) -> str:
    from build_metadata import _source_id

    return _source_id(path)


def test_metadata_requires_existing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_metadata(tmp_path / "missing", tmp_path / "metadata.csv")


def test_grouped_split_has_no_source_leakage(tmp_path):
    rows = []
    for group in range(20):
        rows.append(
            {
                "file_path": f"video_{group}.mp4",
                "source_id": f"source_{group}",
                "label": "real" if group % 2 == 0 else "fake",
                "is_valid": True,
                "is_duplicate": False,
            }
        )
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    output_dir = tmp_path / "splits"
    counts = split_dataset(metadata_path, output_dir)
    splits = {
        name: pd.read_csv(output_dir / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    assert sum(counts.values()) == len(rows)
    assert not (set(splits["train"].source_id) & set(splits["val"].source_id))
    assert not (set(splits["train"].source_id) & set(splits["test"].source_id))
    assert not (set(splits["val"].source_id) & set(splits["test"].source_id))


def test_frame_extraction_from_synthetic_video(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_video(video_path)
    result = extract_frames(video_path, tmp_path / "frames", frames_per_second=1, max_frames=5)
    assert 0 < len(result) <= 5
    assert all(Path(frame_path).is_file() for frame_path in result)


def test_streamlit_analysis_report_never_contains_fake_scores(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_video(video_path)
    video_info = inspect_video(video_path)
    report = make_report(
        "sample.mp4",
        video_path.stat().st_size,
        video_info,
        audio_available=False,
        audio_warning="FFmpeg not installed",
        checkpoints={},
    )
    assert report["inference"]["status"] == "not_available"
    assert report["inference"]["prediction"] is None
    assert report["inference"]["confidence"] is None
    assert len(video_info["frames"]) > 0


def test_frame_extraction_rejects_invalid_arguments(tmp_path):
    with pytest.raises(ValueError):
        extract_frames(tmp_path / "not-used.mp4", tmp_path / "frames", frames_per_second=0)


def test_audio_preprocessing_valid_and_invalid_inputs(tmp_path):
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)
    features = extract_mfcc(audio_path)
    summary = summarize_audio_features(audio_path)
    assert features.shape[0] == 13
    assert np.isfinite(features).all()
    assert summary["shape"] == list(features.shape)

    silent_path = tmp_path / "silent.wav"
    with wave.open(str(silent_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
    with pytest.raises(ValueError, match="silent"):
        extract_mfcc(silent_path)
    with pytest.raises(FileNotFoundError):
        extract_mfcc(tmp_path / "missing.wav")


def test_audio_preview_uses_bundled_decoder_without_ffmpeg_executable(tmp_path):
    source_audio = tmp_path / "source.wav"
    _write_tone(source_audio)
    preview_path = tmp_path / "preview.wav"

    success, warning = extract_audio_preview(source_audio, preview_path, sample_rate=16000)

    assert success, warning
    assert preview_path.is_file()
    with wave.open(str(preview_path), "rb") as preview:
        assert preview.getframerate() == 16000
        assert preview.getnchannels() == 1
        assert preview.getnframes() > 0


def test_audio_preview_extracts_audio_from_video_container(tmp_path):
    video_path = tmp_path / "video-with-audio.mp4"
    _write_audio_video(video_path)
    preview_path = tmp_path / "preview.wav"

    success, warning = extract_audio_preview(video_path, preview_path, sample_rate=16000)

    assert success, warning
    with wave.open(str(preview_path), "rb") as preview:
        assert preview.getframerate() == 16000
        assert preview.getnchannels() == 1
        assert preview.getnframes() > 0


def test_audio_preview_reports_video_without_audio(tmp_path):
    video_path = tmp_path / "silent-video.mp4"
    _write_video(video_path)
    success, warning = extract_audio_preview(video_path, tmp_path / "preview.wav")
    assert not success
    assert "No audio stream" in warning


def test_train_and_run_multimodal_baseline_on_labeled_videos(tmp_path):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    rows = []
    colors = {"real": (20, 40, 220), "fake": (220, 40, 20)}
    for label, folder in (("real", real_dir), ("fake", fake_dir)):
        for split in ("train", "validation"):
            video_path = folder / f"{label}_{split}.mp4"
            _write_video(video_path, color=colors[label])
            rows.append(
                {
                    "file_path": str(video_path),
                    "label": label,
                    "source_id": f"{label}_{split}",
                    "is_valid": True,
                    "is_duplicate": False,
                    "split": split,
                }
            )

    data = pd.DataFrame(rows)
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    data[data["split"] == "train"].drop(columns="split").to_csv(train_path, index=False)
    data[data["split"] == "validation"].drop(columns="split").to_csv(validation_path, index=False)
    checkpoint = tmp_path / "models" / "multimodal_baseline.joblib"

    result = train_detector(train_path, checkpoint, validation_path)
    assert checkpoint.is_file()
    assert result["training_samples"] == 2
    assert result["validation"]["samples"] == 2
    model = load_baseline_model(checkpoint)
    assert not model_supports_audio(model)
    features, available = extract_multimodal_features(Path(rows[0]["file_path"]))
    assert features.shape == (FEATURE_COUNT,)
    assert available == {"visual": True, "audio": False}

    prediction = predict_video(rows[0]["file_path"], checkpoint.parent)
    assert prediction["status"] == "available"
    assert prediction["prediction"] in {"real", "fake"}
    assert 0 <= prediction["fake_probability"] <= 1
    assert "visual" in prediction["modalities_used"]
    assert "audio" not in prediction["modalities_used"]
    assert any("training data contained no audio" in warning for warning in prediction["warnings"])

    audio_video = tmp_path / "audio-video.mp4"
    _write_audio_video(audio_video)
    audio_prediction = predict_video(audio_video, checkpoint.parent)
    assert "audio" not in audio_prediction["modalities_used"]
    assert any("ignores uploaded audio" in warning for warning in audio_prediction["warnings"])


def test_audio_support_is_detected_from_training_features():
    features = np.zeros((4, FEATURE_COUNT), dtype=np.float32)
    audio_presence_index = VISUAL_FEATURE_COUNT + AUDIO_FEATURE_COUNT - 1
    features[:, audio_presence_index] = [0, 1, 0, 1]
    model = create_baseline_model().fit(features, np.array([0, 1, 1, 0]))
    assert model_supports_audio(model)


def test_video_prediction_requires_a_trained_checkpoint(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _write_video(video_path)
    with pytest.raises(InferenceNotReadyError, match="No trained multimodal baseline"):
        predict_video(video_path, tmp_path / "missing-models")


def test_visual_model_logits_shape_and_untrained_safety():
    model = VisualDeepfakeModel()
    logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 2)
    with pytest.raises(RuntimeError, match="no validated trained checkpoint"):
        model.predict_proba(torch.randn(1, 3, 64, 64))
    with pytest.raises(RuntimeError, match="Refusing to save untrained"):
        model.save(Path("unused.pt"))


def test_fusion_requires_training_and_rejects_missing_scores(tmp_path):
    model = FusionClassifier()
    with pytest.raises(NotFittedError):
        model.predict_proba(np.array([[0.2, 0.3, 0.4]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(np.array([[0.2, np.nan, 0.4]]), np.array([1]))

    features = np.array(
        [
            [0.1, 0.1, 0.2],
            [0.2, 0.1, 0.2],
            [0.8, 0.9, 0.8],
            [0.9, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    model.fit(features, np.array([0, 0, 1, 1]))
    checkpoint = model.save(tmp_path / "fusion.npz")
    restored = FusionClassifier()
    restored.load(checkpoint)
    predictions = restored.predict_proba(features)
    assert predictions.shape == (4, 2)
    assert np.allclose(predictions.sum(axis=1), 1.0)


def test_evaluation_handles_single_class_and_validates_inputs(tmp_path):
    metrics = binary_metrics([1, 1], [1, 0], [0.8, 0.2])
    assert metrics["accuracy"] == 0.5
    assert metrics["roc_auc"] is None

    labels = tmp_path / "labels.csv"
    predictions = tmp_path / "predictions.csv"
    labels.write_text("label\nreal\nfake\n", encoding="utf-8")
    predictions.write_text("prediction,score\nreal,0.1\nfake,0.9\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report = evaluate_predictions(labels, predictions, report_path)
    assert report["accuracy"] == 1.0
    assert report_path.is_file()

    predictions.write_text("prediction\nreal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Row count mismatch"):
        evaluate_predictions(labels, predictions, report_path)


def test_invalid_media_upload_is_rejected(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not a video")
    assert not valid_media_file(bad_file)


def test_config_paths_are_resolved(tmp_path):
    paths = get_project_paths(
        {
            "paths": {
                "data_root": "custom-data",
                "raw_dir": "custom-data/raw",
                "processed_dir": "custom-data/processed",
                "metadata_dir": "custom-data/metadata",
                "models_dir": "custom-data/models",
                "reports_dir": "custom-data/reports",
            }
        }
    )
    assert paths["raw_dir"].name == "raw"
    assert paths["models_dir"].name == "models"
