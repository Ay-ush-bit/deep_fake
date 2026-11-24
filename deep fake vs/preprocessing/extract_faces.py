from __future__ import annotations

from pathlib import Path

import cv2


def detect_face_crop(frame, scale: float = 1.2) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    padding = int(max(w, h) * (scale - 1) / 2)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, int(w * scale))
    h = min(frame.shape[0] - y, int(h * scale))
    return x, y, w, h


def crop_face_from_frame(frame, output_path: str | Path) -> str | None:
    face_box = detect_face_crop(frame)
    if face_box is None:
        return None
    x, y, w, h = face_box
    face = frame[y : y + h, x : x + w]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), face)
    return str(out)
