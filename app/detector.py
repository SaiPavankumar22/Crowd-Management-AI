"""One shared YOLO model, thread-safe, person class only.

All camera workers call ModelManager.detect(); a single lock serializes
inference so N cameras share one model/GPU safely.
"""
import threading

import numpy as np
import torch


class ModelManager:
    def __init__(self, model_name: str, fallbacks: list, conf: float, imgsz: int):
        self.requested = [model_name] + list(fallbacks)
        self.conf = conf
        self.imgsz = imgsz
        self.lock = threading.Lock()
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
            try:
                self.model = YOLO(name)
                self.name = name
                break
            except Exception as exc:  # noqa: BLE001 - report first/best error
                last_err = exc
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

    def detect(self, frame):
        """Return (boxes_xyxy float32 [n,4], confidences float32 [n])."""
        kwargs = dict(conf=self.conf, classes=[0], imgsz=self.imgsz,
                      device=self.device, verbose=False)
        if self.half:  # GPU only -- omitting it on CPU silences a warning spam
            kwargs["half"] = True
        with self.lock:
            results = self.model.predict(frame, **kwargs)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
        return boxes, confs

    def info(self) -> dict:
        device = "gpu" if self.device != "cpu" else "cpu"
        return {
            "model": self.name,
            "device": device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "loaded": bool(self.model is not None),
            "load_error": self.load_error,
        }
