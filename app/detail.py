"""Two-tier detail escalation for scenes the detector cannot resolve.

A box detector collapses when people are only ~12 px tall. On the packed-ghat
clip the default pass (imgsz 640, full frame) returns **0.3 people** for a scene
containing hundreds, and nothing in the result says the detector failed: an
empty road produces the same output. That is worse than a wrong number, because
the dashboard certifies it as a trusted count.

Two image-statistics triggers were measured and rejected before this design:

* edge density — measured 0.42-0.89 on every sample clip; it responds to scene
  texture (foliage, road markings, compression noise), not to people;
* temporal motion — measured ~0.000 on the packed-ghat clips, which are
  near-static shots of a standing crowd.

Both would have needed per-site calibration against negatives we do not have.
This module instead uses **detection against detection**, which needs none:

1. run the normal (optionally gated-tiled) full-frame pass;
2. if it found almost nobody, re-run the same frame as a 2x2 tile probe
   (cheap: ~15 ms at 640);
3. only if that probe disagrees strongly, run the expensive 3x3 detail pass at
   high resolution (~58 ms at 1280).

Measured on `gettyimages-683716020-640_adpp.mp4`: full frame **0.3**, 2x2 probe
**4.4**, 3x3 @1280 **135.4**. An empty scene never pays for step 3, and idle
cameras back off from even step 2 after repeated empty probes.

The escalated result is not a true count either: 135 is a *lower bound* for a
scene with hundreds. That is why the mode router degrades it and the UI labels
it `>=` rather than presenting it as fact.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import metrics
from .adapters import (DetectionResult, DetectorAdapter, ModelSpec,
                       merge_tiled)
from .tiling import TilingPolicy, compute_tiles, crop


class DetailPass(DetectorAdapter):
    """Escalating wrapper around one loaded model.

    The model instance is shared: only image size and tile grid change between
    tiers, so no extra VRAM is used and no second set of weights is loaded.
    """

    backend = "ultralytics+detail"

    def __init__(self, core: DetectorAdapter, *, policy: TilingPolicy | None = None,
                 probe_below: int = 6, probe_imgsz: int = 640, probe_grid: int = 2,
                 detail_imgsz: int = 1280, detail_grid: int = 3,
                 overlap: float = 0.15, idle_probe_every: int = 4,
                 exit_below: int = 3, exit_frames: int = 10):
        self.core = core
        self.policy = policy or TilingPolicy(enabled=False)
        self.probe_below = int(probe_below)
        self.probe_imgsz = int(probe_imgsz)
        self.probe_grid = max(2, int(probe_grid))
        self.detail_imgsz = int(detail_imgsz)
        self.detail_grid = max(2, int(detail_grid))
        self.overlap = float(overlap)
        self.idle_probe_every = max(1, int(idle_probe_every))
        self.exit_below = int(exit_below)
        self.exit_frames = max(1, int(exit_frames))
        self._idle: dict[str, int] = {}      # camera -> frames since last probe
        # Sticky escalation: a packed scene's full-frame count hovers around the
        # probe threshold, and without this the same camera would alternate
        # between a trusted "7" and a lower bound of 146 on consecutive frames.
        self._mode: dict[str, str] = {}
        self._below: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def inner(self):            # keeps ModelManager.set_profile working
        return self.core

    def set_profile(self, conf: float | None = None, imgsz: int | None = None):
        self.core.set_profile(conf=conf, imgsz=imgsz)

    @property
    def spec(self) -> ModelSpec:
        return self.core.spec

    def set_profile(self, conf: float | None = None, imgsz: int | None = None):
        setter = getattr(self.core, "set_profile", None)
        if setter is not None:
            setter(conf=conf, imgsz=imgsz)

    def info(self) -> dict:
        info = dict(self.core.info())
        info.update({
            "backend": self.backend,
            "detail_pass": {
                "probe_below": self.probe_below,
                "probe": {"grid": f"{self.probe_grid}x{self.probe_grid}",
                          "imgsz": self.probe_imgsz},
                "detail": {"grid": f"{self.detail_grid}x{self.detail_grid}",
                           "imgsz": self.detail_imgsz},
            },
        })
        return info

    # ------------------------------------------------------------------ #
    def _tiled(self, frame: np.ndarray, grid: int, imgsz: int) -> DetectionResult:
        h, w = frame.shape[:2]
        tiles = compute_tiles(w, h, grid, grid, self.overlap)
        crops = [crop(frame, tile) for tile in tiles]
        started = time.perf_counter()
        results = self.core.detect_batch(crops, imgsz=imgsz)
        elapsed = (time.perf_counter() - started) * 1000.0
        return merge_tiled(results, tiles, frame, self.core.spec, elapsed)

    def _in_detail(self, camera_id: str | None) -> bool:
        if camera_id is None:
            return False
        with self._lock:
            return self._mode.get(camera_id) == "detail"

    def _set_mode(self, camera_id: str | None, mode: str):
        if camera_id is None:
            return
        with self._lock:
            self._mode[camera_id] = mode
            self._below[camera_id] = 0

    def _reset_exit(self, camera_id: str | None):
        if camera_id is None:
            return
        with self._lock:
            self._below[camera_id] = 0

    def _bump_exit(self, camera_id: str | None) -> int:
        if camera_id is None:
            return 0
        with self._lock:
            self._below[camera_id] = self._below.get(camera_id, 0) + 1
            return self._below[camera_id]

    def _probe_allowed(self, camera_id: str | None) -> bool:
        if camera_id is None:
            return True
        with self._lock:
            return self._idle.get(camera_id, self.idle_probe_every) >= self.idle_probe_every

    def _tick(self, camera_id: str | None, *, probed: bool = False,
              found: bool = False):
        """Advance the per-camera probe clock.

        A probe that found people means the camera is in detail mode: probe
        again on the very next frame, otherwise the count would fall back to
        the full-frame "0" every few frames. A probe that found nothing is an
        idle camera: wait `idle_probe_every` frames before paying for it again.
        """
        if camera_id is None:
            return
        with self._lock:
            if found:
                self._idle[camera_id] = self.idle_probe_every
            elif probed:
                self._idle[camera_id] = 0
            else:
                self._idle[camera_id] = self._idle.get(camera_id, 0) + 1

    # ------------------------------------------------------------------ #
    def detect_batch(self, frames: list[np.ndarray], *,
                     imgsz: int | None = None,
                     camera_ids: list[str] | None = None,
                     ) -> list[DetectionResult]:
        primary = self.core.detect_batch(frames)
        ids = list(camera_ids or [None] * len(frames))
        out: list[DetectionResult] = []
        for frame, base, camera_id in zip(frames, primary, ids):
            # Gated tiling on the ordinary path (same rule as TiledDetector).
            if self.policy.enabled and self.policy.should_tile(base.boxes):
                started = time.perf_counter()
                primary_tiled = self._tiled(frame, self.policy.rows,
                                            self.policy.cols, self.core.spec.imgsz)
                base = DetectionResult(
                    primary_tiled.boxes, primary_tiled.confidences,
                    primary_tiled.class_ids,
                    primary_tiled.inference_ms + base.inference_ms,
                    primary_tiled.model_name, primary_tiled.model_version,
                    primary_tiled.image_size, tiled=True,
                    tile_count=primary_tiled.tile_count)

            if self._in_detail(camera_id):
                detail = self._tiled(frame, self.detail_grid, self.detail_imgsz)
                if detail.count > self.exit_below:
                    self._reset_exit(camera_id)
                    out.append(_as_detail(detail, base.count))
                    continue
                # The scene emptied out: leave detail mode deliberately, so one
                # quiet frame cannot drop a packed camera back to primary.
                if self._bump_exit(camera_id) >= self.exit_frames:
                    self._set_mode(camera_id, "primary")
                    out.append(base)
                    continue
                out.append(_as_detail(detail, base.count))
                continue

            if base.count > self.probe_below or not self._probe_allowed(camera_id):
                self._tick(camera_id)
                out.append(base)
                continue

            metrics.inc("godavari_detail_probes_total")
            probe = self._tiled(frame, self.probe_grid, self.probe_imgsz)
            # Disagreement is measured against the primary pass, never against
            # an absolute count: on the packed-ghat clip the probe finds only
            # 4.4 people, which is still four times what the full frame saw and
            # the signal that the scene is being under-read.
            if probe.count < base.count + 2:
                self._tick(camera_id, probed=True)
                out.append(base)
                continue

            metrics.inc("godavari_detail_escalations_total")
            detail = self._tiled(frame, self.detail_grid, self.detail_imgsz)
            best = detail if detail.count >= probe.count else probe
            self._set_mode(camera_id, "detail")
            self._tick(camera_id, probed=True, found=True)
            out.append(_as_detail(best, base.count))
        return out

    def detect(self, frame: np.ndarray,
               camera_id: str | None = None) -> DetectionResult:
        return self.detect_batch([frame], camera_ids=[camera_id])[0]


def _as_detail(result: DetectionResult, primary_count: int) -> DetectionResult:
    """Re-label an escalated result so the router and UI can reason about it."""
    return DetectionResult(
        boxes=result.boxes, confidences=result.confidences,
        class_ids=result.class_ids, inference_ms=result.inference_ms,
        model_name=result.model_name, model_version=result.model_version,
        image_size=result.image_size, tiled=result.tiled,
        tile_count=result.tile_count, tier="detail",
        primary_count=primary_count,
    )
