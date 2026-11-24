from __future__ import annotations

import csv
import io
import json
import logging
import tempfile
import wave
from pathlib import Path
from typing import Any

import av
import cv2
import streamlit as st

from inference.predict import InferenceNotReadyError, predict_video
from src.config import ROOT, load_config

CONFIG = load_config()
LOGGER = logging.getLogger(__name__)
SUPPORTED_VIDEO_TYPES = {"mp4", "mov", "avi", "mkv"}
DEFAULT_MAX_UPLOAD_MB = 200


def inspect_video(video_path: Path, frames_per_second: int = 1, max_frames: int = 8) -> dict[str, Any]:
    """Extract a small set of preview frames and report basic video properties."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, round(fps / max(1, frames_per_second))) if fps > 0 else 1

    frames: list[Any] = []
    frame_index = 0
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError("No readable video frames were found.")

    return {
        "frames": frames,
        "fps": float(fps) if fps > 0 else None,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps if fps > 0 and frame_count > 0 else None,
        "width": width,
        "height": height,
    }


def extract_audio_preview(video_path: Path, output_path: Path, sample_rate: int = 16000) -> tuple[bool, str]:
    """Decode up to 60 seconds of mono audio to WAV using PyAV's bundled FFmpeg libraries."""
    if sample_rate <= 0:
        return False, "Audio sample rate must be a positive integer."

    try:
        with av.open(str(video_path)) as container:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if not audio_streams:
                return False, "No audio stream was found in this video."

            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            max_samples = sample_rate * 60
            samples_written = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with wave.open(str(output_path), "wb") as output_wave:
                output_wave.setnchannels(1)
                output_wave.setsampwidth(2)
                output_wave.setframerate(sample_rate)

                for frame in container.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        samples = resampled.to_ndarray().reshape(-1)
                        remaining = max_samples - samples_written
                        if remaining <= 0:
                            break
                        samples = samples[:remaining]
                        output_wave.writeframesraw(samples.tobytes())
                        samples_written += len(samples)
                    if samples_written >= max_samples:
                        break

            if samples_written == 0:
                output_path.unlink(missing_ok=True)
                return False, "The audio stream contained no decodable samples."
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        output_path.unlink(missing_ok=True)
        LOGGER.info("Audio preview extraction failed for %s: %s", video_path.name, exc)
        return False, f"Could not decode an audio preview: {exc}"
    return True, ""


def checkpoint_status(models_dir: Path) -> dict[str, bool]:
    return {
        "Trainable multimodal baseline": (models_dir / "multimodal_baseline.joblib").is_file(),
    }


def make_report(
    filename: str,
    size_bytes: int,
    video_info: dict[str, Any],
    audio_available: bool,
    audio_warning: str,
    checkpoints: dict[str, bool],
    inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inference_result = inference or {
        "status": "not_available",
        "prediction": None,
        "confidence": None,
        "fake_probability": None,
        "reason": (
            "A trained baseline detector is not installed. Use the documented dataset preparation "
            "and training commands before requesting a video prediction."
        ),
    }
    return {
        "file": filename,
        "file_size_bytes": size_bytes,
        "video": {key: value for key, value in video_info.items() if key != "frames"},
        "audio_preview_available": audio_available,
        "audio_warning": audio_warning or None,
        "model_checkpoints": checkpoints,
        "inference": inference_result,
        "notice": "A model score is not a guarantee of authenticity or manipulation.",
    }


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Deepfake Detection",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    max_upload_mb = int(CONFIG.get("streamlit", {}).get("max_upload_size_mb", DEFAULT_MAX_UPLOAD_MB))
    sample_rate = int(CONFIG.get("audio", {}).get("sample_rate", 16000))
    models_dir = ROOT / CONFIG.get("paths", {}).get("models_dir", "data/models")
    models = checkpoint_status(models_dir)

    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {background: #f4f6fb; padding: 1rem; border-radius: 12px;}
        .intro {color: #536079; font-size: 1.05rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🔎 Multimodal Deepfake Detection")
    st.markdown(
        '<p class="intro">Review the video and its audio track, inspect sampled frames, and check model readiness.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis setup")
        st.caption(f"Upload limit: {max_upload_mb} MB")
        st.caption(f"Audio preview sample rate: {sample_rate:,} Hz")
        st.caption("Audio decoding uses bundled libraries; no FFmpeg executable is required.")
        st.subheader("Checkpoint status")
        for name, available in models.items():
            (st.success if available else st.warning)(f"{name}: {'found' if available else 'not found'}")
        st.caption(f"Expected location: `{models_dir}`")
        if models["Trainable multimodal baseline"]:
            st.caption("Checkpoint ready. Upload a video to run the experimental baseline.")
        else:
            st.caption("Train the baseline using labeled videos before running detection.")

    upload = st.file_uploader(
        "Choose a video file",
        type=sorted(SUPPORTED_VIDEO_TYPES),
        help=f"Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_TYPES))}. Maximum size: {max_upload_mb} MB.",
    )
    if upload is None:
        st.info("Upload a video to begin. Your file is processed temporarily and is not added to the dataset.")
        return

    suffix = Path(upload.name).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_VIDEO_TYPES:
        st.error("Unsupported file type. Upload an MP4, MOV, AVI, or MKV video.")
        return

    video_bytes = upload.getvalue()
    if not video_bytes:
        st.error("The uploaded file is empty.")
        return
    if len(video_bytes) > max_upload_mb * 1024 * 1024:
        st.error(f"The file is larger than the configured {max_upload_mb} MB limit.")
        return

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Video preview")
        st.video(video_bytes)
        st.caption(f"{upload.name} · {len(video_bytes) / (1024 * 1024):.2f} MB")

    with st.spinner("Inspecting video and extracting a short audio preview…"):
        try:
            with tempfile.TemporaryDirectory(prefix="deepfake-review-") as temp_dir:
                temp_root = Path(temp_dir)
                video_path = temp_root / f"upload.{suffix}"
                audio_path = temp_root / "audio_preview.wav"
                video_path.write_bytes(video_bytes)

                video_info = inspect_video(video_path)
                audio_ok, audio_warning = extract_audio_preview(video_path, audio_path, sample_rate)
                try:
                    inference = predict_video(video_path, models_dir)
                except InferenceNotReadyError as exc:
                    inference = None
                    inference_reason = str(exc)
                except Exception as exc:
                    LOGGER.exception("Trained detector failed while processing upload")
                    inference = {
                        "status": "error",
                        "prediction": None,
                        "confidence": None,
                        "fake_probability": None,
                        "reason": f"The trained detector could not process this upload: {exc}",
                    }
                    inference_reason = inference["reason"]
                else:
                    inference_reason = ""
                report = make_report(
                    upload.name,
                    len(video_bytes),
                    video_info,
                    audio_ok,
                    audio_warning,
                    models,
                    inference,
                )
                if inference is None:
                    report["inference"]["reason"] = inference_reason

                with right:
                    st.subheader("Video details")
                    metric1, metric2 = st.columns(2)
                    metric1.metric("Resolution", f"{video_info['width']} × {video_info['height']}")
                    duration = video_info["duration_seconds"]
                    metric2.metric("Duration", f"{duration:.1f} s" if duration is not None else "Unknown")
                    st.metric("Sampled frames", len(video_info["frames"]))

                st.subheader("Sampled frames")
                st.image(
                    video_info["frames"],
                    caption=[f"Sample {i + 1}" for i in range(len(video_info["frames"]))],
                    use_column_width=True,
                )

                st.subheader("Audio preview")
                if audio_ok:
                    st.audio(audio_path.read_bytes(), format="audio/wav")
                    st.caption("Preview is limited to the first 60 seconds.")
                else:
                    st.warning(audio_warning)

            st.subheader("Detection status")
            if inference is not None:
                if inference["status"] == "available":
                    st.success(f"Baseline prediction: **{inference['prediction'].upper()}**")
                    score_col, confidence_col = st.columns(2)
                    score_col.metric("Estimated fake probability", f"{inference['fake_probability']:.1%}")
                    confidence_col.metric("Model confidence", f"{inference['confidence']:.1%}")
                    st.caption(f"Modalities used: {', '.join(inference['modalities_used']) or 'none'}")
                    for warning in inference["warnings"]:
                        st.warning(warning)
                    st.caption(f"Model: {inference['model']}")
                else:
                    st.error(inference["reason"])
            else:
                st.warning(
                    "A detector checkpoint is not trained yet. Prepare labeled video data and train the baseline "
                    "before requesting a real prediction."
                )
                st.code(
                    "python build_metadata.py --raw-dir data/raw --output data/metadata/metadata.csv\n"
                    "python split_dataset.py --metadata data/metadata/metadata.csv --output-dir data/processed\n"
                    "python -m training.train_detector --train-metadata data/processed/train.csv "
                    "--validation-metadata data/processed/val.csv",
                    language="powershell",
                )
                st.caption(report["inference"]["reason"])

            report_json = json.dumps(report, indent=2)
            csv_rows = [
                {"field": "file", "value": report["file"]},
                {"field": "file_size_bytes", "value": report["file_size_bytes"]},
                *({"field": f"video.{key}", "value": value} for key, value in report["video"].items()),
                {"field": "audio_preview_available", "value": report["audio_preview_available"]},
                {"field": "inference.status", "value": report["inference"]["status"]},
                {"field": "inference.prediction", "value": report["inference"].get("prediction") or ""},
                {
                    "field": "inference.fake_probability",
                    "value": report["inference"].get("fake_probability", ""),
                },
            ]
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["field", "value"])
            writer.writeheader()
            writer.writerows(csv_rows)

            col_json, col_csv = st.columns(2)
            col_json.download_button(
                "Download JSON report",
                data=report_json,
                file_name="deepfake_analysis_report.json",
                mime="application/json",
                use_container_width=True,
            )
            col_csv.download_button(
                "Download CSV report",
                data=csv_buffer.getvalue(),
                file_name="deepfake_analysis_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(report["notice"])
        except (OSError, ValueError, cv2.error) as exc:
            LOGGER.exception("Uploaded video analysis failed")
            st.error(f"Could not analyze this upload: {exc}")


if __name__ == "__main__":
    main()
