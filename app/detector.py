"""Model manager built on the adapter boundary.

The pipeline never sees Ultralytics objects any more: every call returns a
`DetectionResult` (see `app/adapters.py`). That is what lets a model be
benchmarked, licensed and swapped without touching tracking, zones or alerts.

Switch semantics follow `improvement.md` §"Model registry":
load the replacement, warm it, verify it, then swap atomically; a failed load
leaves the previous model active; every switch is written to the audit log.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch

from . import config, metrics, store
from .adapters import (DetectionResult, ModelSpec, TiledDetector,
                       UltralyticsDetector, spec_for)
from .detail import DetailPass
from .tiling import TilingPolicy


class ModelManager:
    def __init__(self, model_name: str, fallbacks: list, conf: float, imgsz: int):
        self.requested = [model_name] + [m for m in fallbacks if m != model_name]
        self.conf = conf
        self.imgsz = imgsz
        self.default_conf = conf
        self.default_imgsz = imgsz
        self.profile = "balanced" if (conf, imgsz) == (0.30, 640) else "custom"
        self.lock = threading.RLock()
        self.ready = threading.Event()
        self.model = None            # active adapter (kept for API compatibility)
        self.name: str | None = None
        self.load_error: str | None = None
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        self.loaded_at: float | None = None
        self.switch_count = 0
        self.tiling_policy = TilingPolicy(
            enabled=config.TILING != "off",
            rows=config.TILE_ROWS, cols=config.TILE_COLS,
            overlap=config.TILE_OVERLAP,
            min_box_height_px=config.TILE_MIN_BOX_HEIGHT_PX,
        )

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def _build(self, name: str, conf: float | None = None,
               imgsz: int | None = None):
        detector = UltralyticsDetector(
            name, conf=conf if conf is not None else self.conf,
            imgsz=imgsz if imgsz is not None else self.imgsz,
            device=self.device, half=self.half,
        )
        detector.load()
        if config.DETAIL_PASS:
            adapter = DetailPass(
                detector, policy=self.tiling_policy,
                probe_below=config.DETAIL_PROBE_BELOW,
                probe_imgsz=config.DETAIL_PROBE_IMGSZ,
                probe_grid=config.DETAIL_PROBE_GRID,
                detail_imgsz=config.DETAIL_IMGSZ,
                detail_grid=config.DETAIL_GRID,
                overlap=config.TILE_OVERLAP,
                idle_probe_every=config.DETAIL_IDLE_PROBE_EVERY,
                exit_below=config.DETAIL_EXIT_BELOW,
                exit_frames=config.DETAIL_EXIT_FRAMES,
            )
        elif config.TILING != "off":
            adapter = TiledDetector(detector, self.tiling_policy)
        else:
            adapter = detector
        # Warm-up: forces the weight download and the first CUDA kernels now,
        # not on the first real camera frame.
        adapter.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        return adapter

    def load(self):
        last_err = None
        for name in self.requested:
            try:
                adapter = self._build(name)
            except Exception as exc:  # noqa: BLE001 - try the next fallback
                last_err = exc
                self.load_error = str(exc)
                continue
            with self.lock:
                self.model = adapter
                self.name = name
                self.load_error = None
                self.loaded_at = time.time()
            break
        if self.model is None:
            self.load_error = str(last_err) if last_err else "no model could be loaded"
        self.ready.set()

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def detect_batch(self, frames,
                     camera_ids: list[str] | None = None) -> list[DetectionResult]:
        adapter = self.model  # atomic read: an in-flight batch keeps its model
        if adapter is None:
            raise RuntimeError("model not loaded")
        return adapter.detect_batch(frames, camera_ids=camera_ids)

    def detect(self, frame) -> DetectionResult:
        return self.detect_batch([frame])[0]

    # ------------------------------------------------------------------ #
    # runtime management
    # ------------------------------------------------------------------ #
    def switch(self, name: str) -> dict:
        """Load and warm a replacement, then swap it in atomically."""
        with self.lock:
            previous = self.model
            previous_name = self.name
            try:
                adapter = self._build(name)
            except Exception as exc:  # noqa: BLE001 - keep the old model active
                store.add_audit("model.switch.failed", "operator", name,
                                {"error": str(exc), "still_active": previous_name})
                raise RuntimeError(f"model load failed, {previous_name} still active: {exc}")
            self.model = adapter
            self.name = name
            self.loaded_at = time.time()
            self.switch_count += 1
            profile = self.profile
        store.add_audit("model.switch", "operator", name,
                        {"from": previous_name, "to": name, "profile": profile})
        return self.info()

    def set_profile(self, profile: str) -> dict:
        settings = {
            "balanced": (0.30, 640),
            "dense": (0.15, 1280),
            "custom": (self.default_conf, self.default_imgsz),
        }
        if profile not in settings:
            raise ValueError("profile must be balanced, dense, or custom")
        conf, imgsz = settings[profile]
        with self.lock:
            self.conf, self.imgsz, self.profile = conf, imgsz, profile
            adapter = self.model
        if adapter is not None:
            inner = getattr(adapter, "inner", adapter)
            set_profile = getattr(inner, "set_profile", None)
            if callable(set_profile):
                set_profile(conf=conf, imgsz=imgsz)
        store.add_audit("model.profile", "operator", profile,
                        {"conf": conf, "imgsz": imgsz})
        return self.info()

    def set_tiling(self, enabled: bool) -> dict:
        with self.lock:
            self.tiling_policy.enabled = bool(enabled)
        store.add_audit("model.tiling", "operator", str(bool(enabled)), {})
        return self.info()

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        adapter = self.model
        info = adapter.info() if adapter is not None else {}
        device = "gpu" if self.device != "cpu" else "cpu"
        merged = {
            "model": self.name,
            "device": device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "loaded": bool(adapter is not None),
            "load_error": self.load_error,
            "supported_batch_size": self.batch_size_supported(),
            "profile": self.profile,
            "confidence": self.conf,
            "imgsz": self.imgsz,
            "available_models": list(dict.fromkeys(self.requested)),
            "switches": self.switch_count,
            "loaded_at": self.loaded_at,
        }
        merged.update({k: v for k, v in info.items() if k not in merged})
        merged.setdefault("license", spec_for(self.name or "", self.imgsz).license)
        merged.setdefault("backend", spec_for(self.name or "", self.imgsz).backend)
        merged.setdefault("version", spec_for(self.name or "", self.imgsz).version)
        return merged

    @staticmethod
    def batch_size_supported() -> int:
        return max(1, config.BATCH_SIZE)

    def models(self) -> list[dict]:
        out = []
        for name in dict.fromkeys(self.requested):
            spec: ModelSpec = spec_for(name, self.imgsz)
            entry = spec.to_dict()
            entry.update({
                "status": self.status(name),
                "warm": bool(self.model is not None and name == self.name),
            })
            out.append(entry)
        return out

    def status(self, name: str) -> dict:
        active = bool(self.model is not None and name == self.name)
        return {
            "name": name,
            "registered": name in self.requested,
            "active": active,
            "warm": active and self.loaded_at is not None,
            "device": "gpu" if self.device != "cpu" else "cpu",
            "loaded_at": self.loaded_at if active else None,
            "load_error": self.load_error if name == self.name else None,
        }
