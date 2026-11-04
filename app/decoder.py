"""Video decoder selection with safe software fallback."""
from __future__ import annotations

import cv2

from . import config


def open_capture(source):
    """Open a source using the requested backend when OpenCV supports it."""
    requested = config.DECODER
    candidates = []
    if requested in ("auto", "gstreamer", "nvdec"):
        candidates.append(cv2.CAP_GSTREAMER)
    if requested in ("auto", "ffmpeg") and hasattr(cv2, "CAP_FFMPEG"):
        candidates.append(cv2.CAP_FFMPEG)
    candidates.append(cv2.CAP_ANY)

    for backend in candidates:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            return cap, (
                "gstreamer" if backend == getattr(cv2, "CAP_GSTREAMER", -1)
                else "ffmpeg" if backend == getattr(cv2, "CAP_FFMPEG", -1)
                else "opencv"
            )
        cap.release()
    return None, "unavailable"
