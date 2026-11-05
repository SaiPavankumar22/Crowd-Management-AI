"""Tests for the improvement.md architecture pieces.

Covers the mode router's degradation contract, tiled-inference merging, the
analytics dead-man switch, outbox durability/idempotency and the scheduler's
fairness and staleness rules.
"""
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from app.adapters import DetectionResult
from app.analytics import DEGRADED, HEALTHY, AnalyticsHealth, AnalyticsHealthConfig
from app.router import DENSE, MODERATE, SPARSE, UNKNOWN, ModeRouter, RouterConfig
from app.scheduler import (InferenceScheduler, InferenceTimeout, StaleFrameError)
from app.tiling import (TilingPolicy, compute_tiles, merge_tile_results, nms,
                        offset_boxes)


def make_detection(count: int, box_height: float = 90.0, conf: float = 0.8,
                   low_conf_ratio: float = 0.0) -> DetectionResult:
    boxes = np.tile(np.array([0, 0, 40, box_height], np.float32), (count, 1))
    confidences = np.full(count, conf, np.float32)
    if count and low_conf_ratio:
        low = int(round(count * low_conf_ratio))
        confidences[:low] = 0.2
    return DetectionResult(boxes, confidences, np.zeros(count, np.int32),
                           9.0, "yolo26n.pt", "26-nano", (1280, 720))


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.router = ModeRouter(RouterConfig(hysteresis_frames=3))

    def test_sparse_scene_is_operational_detector(self):
        out = self.router.route(make_detection(2))
        self.assertEqual(out.mode, SPARSE)
        self.assertEqual(out.selected_count, 2)
        self.assertEqual(out.count_source, "detector")
        self.assertFalse(out.degraded)

    def test_dense_scene_never_publishes_a_confident_count(self):
        for _ in range(3):
            out = self.router.route(make_detection(30, box_height=18))
        self.assertEqual(out.mode, DENSE)

        self.assertIsNone(out.selected_count)
        self.assertEqual(out.estimation_mode, "degraded_review")
        self.assertTrue(out.review_required)
        self.assertEqual(out.detector_count, 30)

    def test_detector_and_density_disagreement_requires_review(self):
        from app.adapters import DensityResult

        density = DensityResult(90.0, {}, "advisory", 30.0, "dm-count", "1")
        out = self.router.route(make_detection(20), density=density)
        self.assertTrue(out.degraded)
        self.assertEqual(out.density_count, 90.0)
        # The detector count is retained but flagged low confidence, never summed.
        self.assertEqual(out.selected_count, 20)
        self.assertEqual(out.confidence, "low")

    def test_bad_camera_health_is_unknown_not_safe(self):
        out = self.router.route(make_detection(0), camera_health="frozen")
        self.assertEqual(out.mode, UNKNOWN)
        self.assertIsNone(out.selected_count)
        self.assertTrue(out.degraded)

    def test_recovered_detail_count_is_published_as_a_lower_bound(self):
        """A tiled re-run found the crowd the full-frame pass missed."""
        detail = make_detection(135, box_height=22)
        detail = DetectionResult(detail.boxes, detail.confidences, detail.class_ids,
                                 detail.inference_ms, detail.model_name,
                                 detail.model_version, detail.image_size,
                                 tiled=True, tile_count=9, tier="detail",
                                 primary_count=0)
        out = self.router.route(detail)
        self.assertEqual(out.mode, DENSE)
        # Never a trusted count, and never the full-frame "0" either.
        self.assertIsNone(out.selected_count)
        self.assertEqual(out.detector_count, 135)
        self.assertEqual(out.estimation_mode, "detector_lower_bound")
        self.assertIn("lower bound", out.reason)
        self.assertTrue(out.review_required)

    def test_detail_pass_that_found_nothing_is_not_degraded(self):
        empty = make_detection(0)
        empty = DetectionResult(empty.boxes, empty.confidences, empty.class_ids,
                                empty.inference_ms, empty.model_name,
                                empty.model_version, empty.image_size,
                                tiled=True, tile_count=4, tier="detail",
                                primary_count=0)
        out = self.router.route(empty)
        self.assertEqual(out.mode, SPARSE)
        self.assertFalse(out.degraded)


    def test_mode_switch_needs_hysteresis_but_degrades_fast(self):
        self.router.route(make_detection(2))
        # One moderately busy frame must not flip the mode permanently.
        out = self.router.route(make_detection(6))
        self.assertEqual(out.mode, SPARSE)
        # Dense evidence degrades immediately: never show a confident count
        # while the detector is failing.
        out = self.router.route(make_detection(40, box_height=15))
        self.assertEqual(out.mode, DENSE)


class PerCameraProfileTests(unittest.TestCase):
    def test_scheduler_batches_are_grouped_by_camera_profile(self):
        from app.detector import ModelManager

        class FakeAdapter:
            def __init__(self):
                self.profiles = []
            def set_profile(self, conf=None, imgsz=None):
                self.profiles.append((conf, imgsz))
            def detect_batch(self, frames, *, camera_ids=None):
                return list(zip(camera_ids, [self.profiles[-1]] * len(frames)))

        manager = ModelManager("stub", [], 0.30, 640)
        manager.model = FakeAdapter()
        manager.set_camera_profile("dense-cam", "dense")
        result = manager.detect_batch([object(), object()], camera_ids=["dense-cam", "plain-cam"])
        self.assertEqual(result[0][0], "dense-cam")
        self.assertEqual(result[0][1], (0.15, 1280))
        self.assertEqual(result[1][0], "plain-cam")
        self.assertEqual(result[1][1], (0.30, 640))


class TilingTests(unittest.TestCase):
    def test_grid_covers_frame_with_overlap(self):
        tiles = compute_tiles(1280, 720, 3, 3, overlap=0.15)
        self.assertEqual(len(tiles), 9)
        self.assertEqual(tiles[0].x, 0)
        self.assertLessEqual(tiles[-1].x + tiles[-1].w, 1280)
        self.assertLessEqual(tiles[-1].y + tiles[-1].h, 720)

    def test_offset_and_merge_removes_duplicate_person_on_seam(self):
        tile = compute_tiles(1000, 500, 1, 2, overlap=0.2)[1]
        boxes = np.array([[100, 100, 160, 260]], np.float32)
        shifted = offset_boxes(boxes, tile)
        self.assertGreater(shifted[0, 0], 100)
        doubled = np.concatenate([shifted, shifted + 3.0])
        scores = np.array([0.9, 0.85], np.float32)
        merged_boxes, merged_scores = merge_tile_results(doubled, scores, 0.55)
        self.assertEqual(len(merged_boxes), 1)
        self.assertEqual(merged_scores[0], 0.9)

    def test_gate_tiles_only_when_people_are_small(self):
        policy = TilingPolicy(min_box_height_px=40, min_boxes=3)
        self.assertFalse(policy.should_tile(make_detection(10, 90).boxes))
        self.assertTrue(policy.should_tile(make_detection(10, 25).boxes))
        self.assertFalse(policy.should_tile(make_detection(2, 25).boxes))

    def test_nms_keeps_disjoint_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
        keep = nms(boxes, np.array([0.9, 0.8], np.float32))
        self.assertEqual(len(keep), 2)


class AnalyticsHealthTests(unittest.TestCase):
    def test_no_inference_is_degraded(self):
        health = AnalyticsHealth("cam", AnalyticsHealthConfig(inference_timeout_s=5))
        health.record_inference(10, ts=100)
        self.assertEqual(health.evaluate(now=104)["status"], HEALTHY)
        row = health.evaluate(now=120)
        self.assertEqual(row["status"], DEGRADED)
        self.assertIn("no_inference_for_20s", row["reasons"])

    def test_healthy_camera_count_collapse_is_flagged(self):
        cfg = AnalyticsHealthConfig(collapse_window_s=10, collapse_min_previous=8)
        health = AnalyticsHealth("cam", cfg)
        for i in range(10):
            health.record_inference(20, ts=100 + i)
        for i in range(5):
            health.record_inference(0, ts=115 + i)
        row = health.evaluate(now=130, camera_health="healthy")
        self.assertEqual(row["status"], DEGRADED)
        self.assertIn("count_collapse_suspected", row["reasons"])

    def test_inference_error_is_degraded(self):
        health = AnalyticsHealth("cam")
        health.record_inference(0, error="cuda out of memory")
        self.assertEqual(health.evaluate()["status"], DEGRADED)


class OutboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # config.DATA_DIR is resolved at import time, so point the module
        # attribute at a temp dir and drop the cached connection.
        import app.config as config
        import app.store as store
        self._old_dir = config.DATA_DIR
        config.DATA_DIR = self.tmp.name
        store._conn = None

    def tearDown(self):
        import app.config as config
        import app.store as store
        if store._conn is not None:
            store._conn.close()      # Windows keeps the file locked otherwise
        store._conn = None
        config.DATA_DIR = self._old_dir
        self.tmp.cleanup()

    def _outbox(self, transport):
        from app.outbox import AlertOutbox
        return AlertOutbox(transport=transport, max_attempts=3, poll_s=0.05)

    def test_alert_can_be_closed_and_keeps_operator_attribution(self):
        import app.store as store
        alert = {"id": "close-me", "cam_id": "c1", "cam_name": "Gate",
                 "zone": "North", "level": "WARNING", "message": "busy",
                 "ts": time.time()}
        store.add_alert(alert)
        self.assertTrue(store.close_alert("close-me", "shift-lead"))
        self.assertFalse(store.close_alert("close-me", "shift-lead"))
        row = next(a for a in store.recent_alerts() if a["id"] == "close-me")
        self.assertEqual(row["status"], "closed")
        self.assertEqual(row["closed_by"], "shift-lead")
        self.assertIsNotNone(row["closed_ts"])

    def test_zone_recovery_auto_closes_only_active_capacity_alerts(self):
        import app.store as store
        base = {"cam_id": "c1", "cam_name": "Gate", "zone": "North",
                "zone_id": "z1", "message": "busy", "ts": time.time()}
        for alert_id, level in (("warning", "WARNING"), ("capacity", "OVERCROWDED"),
                                ("other", "INFO")):
            store.add_alert({**base, "id": alert_id, "level": level})
        self.assertEqual(store.close_recovered_zone_alerts("c1", "z1"), 2)
        rows = {a["id"]: a for a in store.recent_alerts()}
        self.assertEqual(rows["warning"]["status"], "closed")
        self.assertEqual(rows["warning"]["closed_by"], "system:recovery")
        self.assertEqual(rows["other"]["status"], "open")

    def test_failed_delivery_is_retried_then_dead_lettered(self):
        class Broken:
            available = True
            calls = 0

            def send(self, payload):
                Broken.calls += 1
                raise RuntimeError("network down")

        outbox = self._outbox(Broken())
        alert = {"id": "a1", "cam_id": "c1", "zone": "Gate", "level": "OVERCROWDED",
                 "message": "over capacity", "ts": time.time()}
        outbox.enqueue(alert)
        # Backoff is real, so advance the clock past each retry window.
        for step in range(3):
            outbox.flush(now=time.time() + step * 600)
        stats = outbox.stats()
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(Broken.calls, 3)

    def test_duplicate_enqueue_is_idempotent(self):
        class Ok:
            available = True
            calls = 0

            def send(self, payload):
                Ok.calls += 1

        outbox = self._outbox(Ok())
        alert = {"id": "dup", "cam_id": "c1", "zone": "Gate", "ts": time.time()}
        outbox.enqueue(alert)
        outbox.enqueue(alert)
        outbox.flush()
        outbox.flush()
        self.assertEqual(Ok.calls, 1)
        self.assertEqual(outbox.stats()["delivered"], 1)

    def test_acknowledgement_is_recorded(self):
        class Ok:
            available = True

            def send(self, payload):
                pass

        outbox = self._outbox(Ok())
        outbox.enqueue({"id": "ackme", "cam_id": "c", "zone": "z", "ts": time.time()})
        row = outbox.ack("ackme", "supervisor")
        self.assertEqual(row["acked_by"], "supervisor")
        self.assertEqual(outbox.stats()["acked"], 1)


class FakeModel:
    """Returns one detection per frame; records concurrency of batch calls."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0
        self.max_batch = 0
        self.lock = threading.Lock()

    def detect_batch(self, frames, *, imgsz=None, camera_ids=None):
        with self.lock:
            self.calls += 1
            self.max_batch = max(self.max_batch, len(frames))
        if self.delay:
            time.sleep(self.delay)
        return [make_detection(1) for _ in frames]


class SchedulerTests(unittest.TestCase):
    def test_batches_multiple_cameras_in_one_call(self):
        model = FakeModel()
        scheduler = InferenceScheduler(model, batch_size=4, max_frame_age_ms=5000,
                                       batch_wait_ms=200)
        # Give the scheduler a moment to be ready before the burst.
        time.sleep(0.05)
        results = []
        threads = [
            threading.Thread(target=lambda i=i: results.append(
                scheduler.infer(f"cam{i}", np.zeros((8, 8, 3), np.uint8))))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        scheduler.stop()
        self.assertEqual(len(results), 3)
        self.assertEqual(model.max_batch, 3)

    def test_stale_frames_are_dropped_not_analysed(self):
        model = FakeModel(delay=0.30)
        scheduler = InferenceScheduler(model, batch_size=1, max_frame_age_ms=50)
        errors = []

        def second_camera():
            try:
                scheduler.infer("cam-b", np.zeros((8, 8, 3), np.uint8), timeout=5)
            except (StaleFrameError, InferenceTimeout) as exc:
                errors.append(exc)

        # cam-a occupies the model for 300ms; cam-b's frame ages out meanwhile.
        first = threading.Thread(
            target=lambda: scheduler.infer("cam-a", np.zeros((8, 8, 3), np.uint8), timeout=5))
        first.start()
        time.sleep(0.05)
        worker = threading.Thread(target=second_camera)
        worker.start()
        first.join(timeout=5)
        worker.join(timeout=5)
        scheduler.stop()
        self.assertTrue(errors, "a frame older than the budget must be dropped")
        self.assertIsInstance(errors[0], StaleFrameError)
        self.assertGreaterEqual(scheduler.stats()["stale_dropped"], 1)

    def test_results_are_detection_results(self):
        scheduler = InferenceScheduler(FakeModel(), batch_size=2, max_frame_age_ms=5000)
        result = scheduler.infer("cam", np.zeros((8, 8, 3), np.uint8))
        scheduler.stop()
        self.assertIsInstance(result, DetectionResult)


if __name__ == "__main__":
    unittest.main()


class _CountingCore:
    """Stands in for the model: the answer depends on how it is cropped.

    Full frame -> 0 people; 2x2 tiles -> 5; 3x3 tiles -> 40. This mirrors the
    measured behaviour of the packed-ghat clip (0.3 / 4.4 / 135).
    """

    FULL_WIDTH = 600

    def __init__(self, per_tile: dict, full_frame: int = 0):
        self.per_tile = per_tile
        self.full_frame = full_frame
        self.batches = 0
        self.tile_batches = 0
        self.spec = type("Spec", (), {"name": "fake.pt", "version": "v0",
                                      "imgsz": 640})()

    def info(self):
        return {"model": "fake.pt", "backend": "fake"}

    def detect_batch(self, frames, *, imgsz=None, camera_ids=None):
        self.batches += 1
        tile_call = bool(frames) and frames[0].shape[1] < self.FULL_WIDTH
        if tile_call:
            self.tile_batches += 1
        out = []
        for frame in frames:
            count = self.per_tile.get(imgsz, 2) if tile_call else self.full_frame
            boxes = np.tile(np.array([0, 0, 10, 20], np.float32), (count, 1))
            out.append(DetectionResult(
                boxes, np.full(count, 0.7, np.float32), np.zeros(count, np.int32),
                5.0, "fake.pt", "v0", (frame.shape[1], frame.shape[0])))
        return out


class DetailPassTests(unittest.TestCase):
    def _frame(self, h=400, w=_CountingCore.FULL_WIDTH):
        return np.zeros((h, w, 3), np.uint8)

    def test_escalates_when_a_tiled_probe_disagrees_with_the_full_frame(self):
        from app.detail import DetailPass
        from app.tiling import TilingPolicy

        core = _CountingCore(per_tile={640: 5, 1280: 40}, full_frame=0)
        pass_ = DetailPass(core, policy=TilingPolicy(enabled=False),
                           probe_below=6, probe_imgsz=640, probe_grid=2,
                           detail_imgsz=1280, detail_grid=3)
        result = pass_.detect(self._frame(), camera_id="cam-1")
        self.assertEqual(result.tier, "detail")
        self.assertEqual(result.primary_count, 0)
        # Two tiled batches: the cheap probe, then the high-resolution pass.
        self.assertEqual(core.tile_batches, 2)
        self.assertGreater(result.count, 0)
        self.assertTrue(result.tiled)

    def test_does_not_escalate_when_the_scene_is_already_resolved(self):
        from app.detail import DetailPass

        core = _CountingCore(per_tile={640: 5, 1280: 40}, full_frame=30)
        pass_ = DetailPass(core, probe_below=6)
        result = pass_.detect(self._frame(), camera_id="cam-1")
        self.assertEqual(result.tier, "primary")
        self.assertEqual(result.count, 30)
        self.assertEqual(core.tile_batches, 0)

    def test_a_dense_camera_stays_in_detail_mode_between_frames(self):
        """No alternating "7" / "≥146" on consecutive frames of one camera."""
        from app.detail import DetailPass

        core = _CountingCore(per_tile={640: 5, 1280: 40}, full_frame=0)
        pass_ = DetailPass(core, probe_below=6, exit_below=3, exit_frames=10)
        first = pass_.detect(self._frame(), camera_id="cam-1")
        self.assertEqual(first.tier, "detail")
        # The full-frame pass now happens to find 7 (> probe_below) on the next
        # frame; the camera must stay in detail mode rather than fall back.
        core.full_frame = 7
        second = pass_.detect(self._frame(), camera_id="cam-1")
        self.assertEqual(second.tier, "detail")

    def test_leaving_detail_mode_needs_sustained_calm(self):
        from app.detail import DetailPass

        core = _CountingCore(per_tile={640: 5, 1280: 40}, full_frame=0)
        pass_ = DetailPass(core, probe_below=6, exit_below=3, exit_frames=3)
        pass_.detect(self._frame(), camera_id="cam-1")
        core.per_tile = {640: 0, 1280: 0}      # crowd dispersed
        tiers = [pass_.detect(self._frame(), camera_id="cam-1").tier
                 for _ in range(4)]
        # Sticky for the first frames, then back to primary.
        self.assertEqual(tiers[0], "detail")
        self.assertEqual(tiers[-1], "primary")

    def test_an_empty_scene_costs_at_most_one_probe_per_idle_window(self):
        from app.detail import DetailPass

        core = _CountingCore(per_tile={640: 0, 1280: 0}, full_frame=0)
        pass_ = DetailPass(core, probe_below=6, idle_probe_every=4)
        for _ in range(8):
            pass_.detect(self._frame(), camera_id="cam-1")
        # 8 frames, one tiled probe per 4 idle frames: never 8 tiled runs.
        self.assertLessEqual(core.tile_batches, 2)
        self.assertEqual(core.batches, 8 + core.tile_batches)
