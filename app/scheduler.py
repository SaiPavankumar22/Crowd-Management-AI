"""Bounded, fair inference scheduler with optional micro-batching."""
from __future__ import annotations

import queue
import threading
import time

from . import config, metrics


class _Request:
    def __init__(self, camera_id, frame):
        self.camera_id = camera_id
        self.frame = frame
        self.done = threading.Event()
        self.result = None
        self.error = None


class InferenceScheduler:
    def __init__(self, model):
        self.model = model
        self.queue = queue.Queue(maxsize=max(1, config.FRAME_QUEUE_SIZE))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="inference-scheduler", daemon=True
        )
        self._thread.start()

    def infer(self, camera_id, frame):
        request = _Request(camera_id, frame)
        try:
            # A full queue means this camera is already behind. Waiting here
            # preserves correctness while keeping queue age bounded.
            self.queue.put(request, timeout=max(0.1, config.MAX_BATCH_WAIT_MS / 1000))
        except queue.Full:
            metrics.inc("godavari_inference_queue_drops_total")
            return self.model.detect(frame)
        request.done.wait()
        if request.error:
            raise request.error
        return request.result

    def _run(self):
        while not self._stop.is_set():
            try:
                first = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + config.MAX_BATCH_WAIT_MS / 1000
            while len(batch) < max(1, config.BATCH_SIZE):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results = self.model.detect_batch([item.frame for item in batch])
                for item, result in zip(batch, results):
                    item.result = result
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    item.error = exc
            finally:
                for item in batch:
                    item.done.set()

    def stop(self):
        self._stop.set()
