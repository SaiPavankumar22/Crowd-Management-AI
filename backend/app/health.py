"""Camera frame health and image-quality measurements."""
from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np

from . import config


class FrameHealth:
    """Small state machine that distinguishes an empty scene from bad input."""

    def __init__(self):
        self.last_frame_ts = 0.0
        self.last_content_ts = 0.0
        self.last_quality_ts = 0.0
        self.last_digest = None
        self.same_digest_since = None
        self.decode_failures = 0
        self.quality = {
            "brightness": None,
            "contrast": None,
            "blur_score": None,
            "status": "unknown",
        }

    def frame_received(self, frame: np.ndarray, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        self.last_frame_ts = now
        self.decode_failures = 0
        digest = hashlib.blake2b(
            cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA).tobytes(),
            digest_size=8,
        ).hexdigest()
        if digest != self.last_digest:
            self.last_digest = digest
            self.last_content_ts = now
            self.same_digest_since = None
        elif self.same_digest_since is None:
            self.same_digest_since = now

        if now - self.last_quality_ts >= config.QUALITY_SAMPLE_PERIOD_S:
            self.quality = measure_quality(frame)
            self.last_quality_ts = now
        return self.snapshot(now)

    def decode_failed(self) -> dict:
        self.decode_failures += 1
        return self.snapshot(time.time())

    def snapshot(self, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        stale = not self.last_frame_ts or now - self.last_frame_ts > config.FRAME_STALE_S
        frozen = (
            self.last_content_ts > 0
            and now - self.last_content_ts >= config.FROZEN_FRAME_S
        )
        quality_status = self.quality.get("status")
        if stale:
            status = "stale"
        elif frozen:
            status = "frozen"
        elif quality_status in ("dark", "low_contrast", "blurry"):
            status = quality_status
        else:
            status = "healthy"
        return {
            "status": status,
            "last_frame_ts": self.last_frame_ts,
            "frame_age_s": round(max(0.0, now - self.last_frame_ts), 2)
            if self.last_frame_ts else None,
            "decode_failures": self.decode_failures,
            "quality": dict(self.quality),
        }


def measure_quality(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < config.MIN_BRIGHTNESS:
        status = "dark"
    elif contrast < config.MIN_CONTRAST:
        status = "low_contrast"
    elif blur_score < config.MIN_BLUR_SCORE:
        status = "blurry"
    else:
        status = "healthy"
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "status": status,
    }
