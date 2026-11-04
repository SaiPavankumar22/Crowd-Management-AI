"""One shared YOLO model, thread-safe, person class only.

All camera workers call ModelManager.detect(); a single lock serializes
inference so N cameras share one model/GPU safely.
"""
import threading
import time

import numpy as np
import torch

from . import metrics


class ModelManager:
    def __init__(self, model_name: str, fallbacks: list, conf: float, imgsz: int):
        self.requested = [model_name] + list(fallbacks)
        self.conf = conf
        self.imgsz = imgsz
        self.lock = threading.RLock()
        self.ready = threading.Event()
        self.model = None
        self.name = None
        self.load_error = None
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"

    def load(self):
        from ultralytics import YOLO

        last_err = None
        for name in self.requested:
            if self._load_one(name):
                break
            last_err = self.load_error
        if self.model is None:
            self.load_error = str(last_err)
            self.ready.set()
            return

        # Warmup so the first real frame is not slow, and to force the
        # weight download now rather than on the first camera.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.detect(dummy)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"warmup failed: {exc}"
        self.ready.set()

    def _load_one(self, name: str) -> bool:
        from ultralytics import YOLO

        try:
            candidate = YOLO(name)
            self.model = candidate
            self.name = name
            self.load_error = None
            return True
        except Exception as exc:  # noqa: BLE001 - report the model failure
            self.load_error = str(exc)
            return False

    def switch(self, name: str) -> dict:
        """Load a requested model and warm it before replacing the active one."""
        with self.lock:
            if not self._load_one(name):
                raise RuntimeError(self.load_error or "model load failed")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.detect(dummy)
        return self.info()

    def detect(self, frame):
        """Return (boxes_xyxy float32 [n,4], confidences float32 [n])."""
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames):
        """Run one model call for multiple frames and return per-frame results."""
        kwargs = dict(conf=self.conf, classes=[0], imgsz=self.imgsz,
                      device=self.device, verbose=False)
        if self.half:  # GPU only -- omitting it on CPU silences a warning spam
            kwargs["half"] = True
        started = time.perf_counter()
        with self.lock:
            results = self.model.predict(frames, **kwargs)
        metrics.inc("godavari_inference_batches_total")
        metrics.inc("godavari_inference_frames_total", len(frames))
        metrics.inc(
            "godavari_inference_seconds_total",
            time.perf_counter() - started,
        )
        output = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                output.append((
                    np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                ))
                continue
            output.append((
                result.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                result.boxes.conf.detach().cpu().numpy().astype(np.float32),
            ))
        return output

    def info(self) -> dict:
        device = "gpu" if self.device != "cpu" else "cpu"
        return {
            "model": self.name,
            "device": device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "loaded": bool(self.model is not None),
            "load_error": self.load_error,
            "supported_batch_size": 1,
            "available_models": list(dict.fromkeys(self.requested)),
        }
