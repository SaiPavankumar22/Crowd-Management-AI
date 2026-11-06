"""Fair, bounded inference scheduler with per-camera latest-frame slots.

Two problems with a single shared FIFO (`improvement.md` §8):

1. When the queue is full the old code ran inference inline on the camera
   thread, which removes serialisation exactly when the GPU is already the
   bottleneck.
2. One camera producing frames faster than the model can consume them fills the
   queue and starves every other camera.

This scheduler keeps **one pending request per camera**. A newer frame from the
same camera replaces the older one (drop-oldest), cameras are dispatched
round-robin, frames older than `CROWD_MAX_FRAME_AGE_MS` are discarded as stale
instead of being analysed, and nothing ever calls the model outside the
dispatcher thread.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from . import config, metrics


class InferenceTimeout(RuntimeError):
    """The scheduler did not answer in time; drop the frame, keep the feed."""


class StaleFrameError(RuntimeError):
    """The frame aged out before inference; analysing it would be misleading."""


class _Request:
    __slots__ = ("camera_id", "frame", "created", "done", "result", "error")

    def __init__(self, camera_id: str, frame, created: float):
        self.camera_id = camera_id
        self.frame = frame
        self.created = created
        self.done = threading.Event()
        self.result = None
        self.error = None


class InferenceScheduler:
    def __init__(self, model, batch_size: int | None = None,
                 max_frame_age_ms: float | None = None,
                 batch_wait_ms: float | None = None):
        self.model = model
        self.batch_size = max(1, int(batch_size or config.BATCH_SIZE))
        self.batch_wait_ms = float(batch_wait_ms if batch_wait_ms is not None
                                   else config.MAX_BATCH_WAIT_MS)
        self.max_frame_age_ms = (max_frame_age_ms if max_frame_age_ms is not None
                                 else config.MAX_FRAME_AGE_MS)
        self._cond = threading.Condition()
        self._pending: dict[str, _Request] = {}
        self._ready: deque[str] = deque()
        self._stop = threading.Event()
        self._drops = 0
        self._stale = 0
        self._timeouts = 0
        self._frame_ages: deque = deque(maxlen=120)
        self._thread = threading.Thread(target=self._run, name="inference-scheduler",
                                        daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    def infer(self, camera_id: str, frame, timeout: float | None = None):
        """Submit a frame and block until its DetectionResult is ready."""
        if self._stop.is_set():
            raise InferenceTimeout("scheduler stopped")
        request = _Request(camera_id, frame, time.monotonic())
        with self._cond:
            previous = self._pending.get(camera_id)
            if previous is not None:
                # Same camera, newer frame: the older one is now worthless.
                previous.error = StaleFrameError("superseded by a newer frame")
                previous.done.set()
                self._drops += 1
                metrics.inc("godavari_inference_queue_drops_total")
            self._pending[camera_id] = request
            if camera_id not in self._ready:
                self._ready.append(camera_id)
            self._cond.notify()
            metrics.set_gauge("godavari_inference_queue_depth", len(self._pending))

        wait_s = timeout if timeout is not None else max(
            2.0, 8.0 * self.max_frame_age_ms / 1000.0)
        if not request.done.wait(wait_s):
            with self._cond:
                if self._pending.get(camera_id) is request:
                    self._pending.pop(camera_id, None)
                    try:
                        self._ready.remove(camera_id)
                    except ValueError:
                        pass
            self._timeouts += 1
            metrics.inc("godavari_inference_timeouts_total")
            raise InferenceTimeout("inference queue timed out")
        if request.error is not None:
            raise request.error
        return request.result

    # ------------------------------------------------------------------ #
    def _next_batch(self) -> list[_Request]:
        with self._cond:
            while not self._ready and not self._stop.is_set():
                self._cond.wait(timeout=0.2)
            batch_ids = []
            if self._ready:
                batch_ids.append(self._ready.popleft())
            # Short micro-batch window: waiting a few milliseconds lets frames
            # from other cameras share one GPU call. Kept tiny because queueing
            # latency is the thing this scheduler exists to bound.
            deadline = time.monotonic() + self.batch_wait_ms / 1000.0
            while len(batch_ids) < self.batch_size and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not self._ready:
                    self._cond.wait(timeout=remaining)
                while self._ready and len(batch_ids) < self.batch_size:
                    batch_ids.append(self._ready.popleft())
            requests = []
            for cam_id in batch_ids:
                request = self._pending.pop(cam_id, None)
                if request is not None:
                    requests.append(request)
            return requests

    def _run(self):
        while not self._stop.is_set():
            batch = self._next_batch()
            if not batch:
                continue
            now = time.monotonic()
            fresh, stale = [], []
            for request in batch:
                age_ms = (now - request.created) * 1000.0
                self._frame_ages.append(age_ms)
                if age_ms > self.max_frame_age_ms:
                    stale.append(request)
                else:
                    fresh.append(request)
            for request in stale:
                self._stale += 1
                metrics.inc("godavari_inference_stale_frames_total")
                request.error = StaleFrameError(
                    f"frame age {age_ms:.0f}ms exceeds "
                    f"{self.max_frame_age_ms:.0f}ms budget")
                request.done.set()
            if not fresh:
                continue
            try:
                results = self.model.detect_batch(
                    [r.frame for r in fresh],
                    camera_ids=[r.camera_id for r in fresh])
                if len(results) != len(fresh):
                    raise RuntimeError("detector returned a mismatched batch")
                for request, result in zip(fresh, results):
                    request.result = result
            except Exception as exc:  # noqa: BLE001 - surface to the caller
                for request in fresh:
                    request.error = exc
            finally:
                for request in fresh:
                    request.done.set()

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        with self._cond:
            depth = len(self._pending)
        ages = list(self._frame_ages)
        p95 = 0.0
        if ages:
            ordered = sorted(ages)
            p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        return {
            "queue_depth": depth,
            "batch_size": self.batch_size,
            "batch_wait_ms": self.batch_wait_ms,
            "max_frame_age_ms": self.max_frame_age_ms,
            "drops": self._drops,
            "stale_dropped": self._stale,
            "timeouts": self._timeouts,
            "frame_age_p95_ms": round(p95, 1),
            "frame_age_mean_ms": round(sum(ages) / len(ages), 1) if ages else 0.0,
        }

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        for request in list(self._pending.values()):
            request.error = InferenceTimeout("scheduler stopped")
            request.done.set()
        self._pending.clear()
        self._ready.clear()
