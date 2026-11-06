"""Per-camera pipeline: capture -> detect -> track -> zone count -> density -> alerts.

Architecture:
- One worker thread per camera keeps N cameras independent (decode pacing,
  tracking, alerts are per-camera state).
- Inference is funneled through the shared ModelManager lock, so 1 model
  serves every camera - memory stays flat as you add cameras.
- Uploaded/sample files play in real time on loop so the dashboard behaves
  exactly like live CCTV. RTSP/webcam are natively live; sample videos are
  inferred at (video fps / TARGET_INFER_FPS) and re-annotated every frame so
  the rendered feed looks smooth but inference stays cheap.
"""
import os
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
import supervision as sv

from . import config, store
from .adapters import (CrowdShieldDensityEstimator, CrowdShieldPointEstimator,
                       NullDensityEstimator, NullPointEstimator,
                       TorchscriptDensityEstimator)
from .alerting import AlertEngine
from .analytics import AnalyticsHealth, AnalyticsHealthConfig
from .calibration import polygon_area_world
from .decoder import open_capture
from .heatmap import HeatmapAccumulator
from .health import FrameHealth
from . import metrics
from .outbox import AlertOutbox
from .router import ModeRouter, RouterConfig
from .scheduler import InferenceScheduler, InferenceTimeout, StaleFrameError
from .zones import SEVERITY, Zone, level_for

BOX_COLOR = (248, 189, 56)  # #38bdf8 (BGR)


def new_camera_id() -> str:
    return uuid.uuid4().hex[:8]


class CameraWorker(threading.Thread):
    daemon = True

    def __init__(self, cam: dict, model, registry):
        super().__init__(name=f"cam-{cam['id']}", daemon=True)
        self.cam = cam
        self.model = model
        self.registry = registry
        self._stop_event = threading.Event()

        self.zones: list[Zone] = [self._zone(z) for z in cam.get("zones", [])]
        self.calibration = cam.get("calibration")
        self.show_detections = bool(cam.get("display", {}).get("detections", True))
        self.engine = AlertEngine(cam["id"], cam["name"],
                                  config.DEFAULT_ALERT_SUSTAIN_S,
                                  config.DEFAULT_ALERT_COOLDOWN_S)
        self.tracker = sv.ByteTrack()
        self.heatmap = HeatmapAccumulator()
        self._track_prev: dict[int, tuple[float, float, float]] = {}
        # Per-zone flow memory so the arrows ease out instead of blinking
        # off on frames where no track happens to be moving.
        self._zone_flow: dict[str, tuple[float, float]] = {}
        self._track_zone: dict[int, tuple[str | None, float]] = {}
        self._zone_transitions: dict[tuple[str, str], int] = {}

        self.latest_jpeg: bytes | None = None
        # A clean copy of the last frame, so a dashboard can show the annotated
        # and the untouched picture of the same camera side by side. The raw
        # JPEG is encoded lazily: cameras nobody watches this way pay nothing.
        self.latest_frame: np.ndarray | None = None
        self._frame_seq = 0
        self._raw_cache_seq = -1
        self._raw_cache: bytes | None = None
        self.metrics: dict = {"status": "starting"}
        self.last_boxes = np.zeros((0, 4), np.float32)
        self._data_lock = threading.Lock()
        self._infer_ts = deque(maxlen=24)
        self._frames_total = 0
        self.total_count = 0.0
        self.frame_health = FrameHealth()
        self.decoder = config.DECODER
        self._last_sample_ts = 0.0
        self._last_heatmap_ts = 0.0
        self.count_lines = list(cam.get("count_lines", []))
        self.line_counts = {line.get("id", line.get("name", "line")): {"entry": 0, "exit": 0}
                            for line in self.count_lines}
        # --- improvement.md: mode router, analytics health, zone versioning ---
        self.zone_version = int(cam.get("zones_version", 0))
        self.router = ModeRouter(RouterConfig(
            dense_box_height_px=config.DENSE_BOX_HEIGHT_PX,
            hysteresis_frames=config.MODE_HYSTERESIS_FRAMES,
        ))
        self.analytics_health = AnalyticsHealth(cam["id"], AnalyticsHealthConfig())
        self._density_result = None
        self._density_error: str | None = None
        self._last_density_ts = 0.0
        self._point_result = None
        self._point_error: str | None = None
        self._last_point_ts = 0.0
        self._line_history: deque = deque(maxlen=600)
        self._route: dict = {}
        self._last_detection = None
        self._was_degraded = False
        self._last_dense_advisory = 0.0

    # ------------------------------------------------------------------ #
    def _zone(self, d: dict) -> Zone:
        return Zone(d["name"], d["polygon"], d.get("capacity", 100),
                    d.get("warn_pct", config.DEFAULT_WARN_PCT), d.get("id"))

    @staticmethod
    def _status_for(cam: dict) -> str:
        return {"rtsp": "starting", "webcam": "starting", "file": "starting"}.get(
            cam["source_type"], "starting")

    def stop(self):
        self._stop_event.set()

    def set_zones(self, zone_dicts: list[dict]):
        zones = [self._zone(z) for z in zone_dicts]
        with self._data_lock:
            self.zones = zones
        self.engine.reset()
        self.heatmap.reset()

    # ------------------------------------------------------------------ #
    def run(self):
        self.model.ready.wait(timeout=300)
        if self.model.load_error or self.model.model is None:
            self.metrics = {"status": "error", "detail": f"model load failed: {self.model.load_error}"}
            return

        is_live = self.cam["source_type"] in ("rtsp", "webcam")
        cap = self._open_capture()
        while cap is None and is_live and not self._stop_event.is_set():
            self.metrics = {"status": "reconnecting", "detail": "waiting for live source"}
            if self._stop_event.wait(2):
                return
            cap = self._open_capture()
        if cap is None:
            self.metrics = {"status": "error",
                            "detail": "could not open source"}
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not is_live:
            fps = min(max(fps, 1.0), 60.0)
        infer_interval = max(1, round(fps / config.TARGET_INFER_FPS))
        frame_period = 0.0 if is_live else 1.0 / fps

        self.metrics = {"status": "running"}
        frame_idx = 0
        while not self._stop_event.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                self.frame_health.decode_failed()
                metrics.inc("godavari_decode_failures_total")
                if not is_live:
                    if config.LOOP_FILE_SOURCES:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    with self._data_lock:
                        self.metrics["status"] = "ended"
                    break
                # Keep retrying live sources until explicitly stopped.
                cap.release()
                recovered = None
                with self._data_lock:
                    self.metrics = {"status": "reconnecting", "detail": "live source disconnected"}
                self.engine.reset()
                while recovered is None and not self._stop_event.is_set():
                    if self._stop_event.wait(2):
                        break
                    recovered = self._open_capture()
                if recovered is None:
                    break
                cap = recovered
                with self._data_lock:
                    self.metrics = {"status": "running", "detail": None}
                continue

            health_snapshot = self.frame_health.frame_received(frame)
            metrics.inc("godavari_frames_total")
            metrics.set_gauge("godavari_camera_frame_age_seconds",
                              health_snapshot.get("frame_age_s") or 0.0,
                              labels={"camera": self.cam["id"]})
            quality = health_snapshot.get("quality") or {}
            score = min(1.0, max(0.0, (float(quality.get("brightness") or 0) / 64.0 +
                                       float(quality.get("contrast") or 0) / 32.0 +
                                       float(quality.get("blur_score") or 0) / 100.0) / 3.0))
            metrics.set_gauge("godavari_camera_quality_score", score,
                              labels={"camera": self.cam["id"]})
            if frame.shape[1] > config.MAX_FRAME_WIDTH:
                scale = config.MAX_FRAME_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (config.MAX_FRAME_WIDTH,
                                           int(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            self._frames_total += 1

            if frame_idx % infer_interval == 0:
                try:
                    self._process(frame)
                except Exception as exc:  # keep the feed alive on bad frames
                    self.metrics["note"] = f"frame error: {exc}"
            self._annotate(frame)
            frame_idx += 1

            if frame_period:
                left = frame_period - (time.time() - t0)
                if left > 0:
                    time.sleep(left)
        cap.release()

    # ------------------------------------------------------------------ #
    def _open_capture(self):
        src: object = config.resolve_media(self.cam["path"])
        if self.cam["source_type"] == "webcam":
            try:
                src = int(str(self.cam["path"]).strip() or "0")
            except ValueError:
                return None
        cap, backend = open_capture(src)
        self.decoder = backend
        if cap is None:
            return None
        if self.cam["source_type"] == "rtsp":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _process(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        now = time.time()
        self._infer_ts.append(now)

        try:
            detection = self.registry.scheduler.infer(self.cam["id"], frame)
        except StaleFrameError:
            # Analysing a 2-second-old frame would show the operator a stale
            # crowd level as if it were current: drop it, keep the feed live.
            metrics.inc("godavari_frames_skipped_total", labels={"camera": self.cam["id"]})
            return
        except InferenceTimeout as exc:
            self.analytics_health.record_inference(0, error=str(exc))
            metrics.inc("godavari_frames_skipped_total", labels={"camera": self.cam["id"]})
            return
        self._last_detection = detection
        boxes, confs = detection.boxes, detection.confidences
        dets = sv.Detections(
            xyxy=boxes, confidence=confs,
            class_id=(detection.class_ids if len(detection.class_ids) == len(boxes)
                      else np.zeros(len(boxes), dtype=int)),
        )
        tracks = self.tracker.update_with_detections(dets)
        track_ids = tracks.tracker_id if len(tracks) else np.zeros(0, int)
        n_tracks = int(np.sum(track_ids >= 0)) if len(track_ids) else 0

        # Foot points (bottom-centre of each box) are the ground truth for
        # "is this person inside this polygon", the heatmap, and flow.
        if len(boxes):
            feet = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, boxes[:, 3]], axis=1)
        else:
            feet = np.zeros((0, 2), np.float32)
        tracked_feet = []
        tracked_ids = []
        if len(tracks) and len(track_ids):
            for i, tid in enumerate(track_ids):
                if int(tid) >= 0:
                    tracked_ids.append(int(tid))
                    x1, _y1, x2, y2 = tracks.xyxy[i]
                    tracked_feet.append(((x1 + x2) * 0.5, y2))
        tracked_feet = np.asarray(tracked_feet, dtype=np.float32).reshape(-1, 2)

        # ---- flow: per-track velocity from the previous inference tick ----
        tpoints = []  # (track_id, foot_x_px, foot_y_px)
        if len(tracks) and len(track_ids):
            for i in range(len(track_ids)):
                tid = int(track_ids[i])
                if tid < 0:
                    continue
                x1, _y1, x2, y2 = tracks.xyxy[i]
                tpoints.append((tid, float((x1 + x2) * 0.5), float(y2)))
        flow_by_track: dict[int, tuple[float, float]] = {}
        prev_positions = {}
        for tid, cx, cy in tpoints:
            nx, ny = cx / w, cy / h
            prev = self._track_prev.get(tid)
            if prev is not None:
                for line in self.count_lines:
                    a, b = line.get("a", [0, 0]), line.get("b", [1, 1])
                    side = lambda p: (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
                    before, after = side(prev[:2]), side((nx, ny))
                    denom = before - after
                    ratio = before / denom if denom else 0
                    ix, iy = prev[0] + ratio * (nx - prev[0]), prev[1] + ratio * (ny - prev[1])
                    seg2 = (b[0]-a[0])**2 + (b[1]-a[1])**2
                    projection = ((ix-a[0])*(b[0]-a[0]) + (iy-a[1])*(b[1]-a[1])) / seg2 if seg2 else -1
                    if before * after < 0 and 0 <= projection <= 1:
                        key = line.get("id", line.get("name", "line"))
                        counts = self.line_counts.setdefault(key, {"entry": 0, "exit": 0})
                        entered = (after > 0) == (line.get("entry_side", "positive") == "positive")
                        counts["entry" if entered else "exit"] += 1
            if prev is not None:
                dt = now - prev[2]
                if 0.04 <= dt <= 5.0:
                    flow_by_track[tid] = ((nx - prev[0]) / dt, (ny - prev[1]) / dt)
            prev_positions[tid] = (nx, ny, now)
        self._track_prev = prev_positions

        if len(feet):
            self.heatmap.add_points(feet / np.array([w, h], np.float32))
        else:
            self.heatmap.add_points(feet)

        with self._data_lock:
            zones = list(self.zones)

        # ---- dense estimators: independent, rate-limited advisory paths ----
        # They are never summed with detector boxes.  The router chooses a
        # single promoted source only after a camera-specific validation gate.
        if now - self._last_point_ts >= config.POINT_PERIOD_S:
            self._last_point_ts = now
            try:
                self._point_result = self.registry.points.estimate(
                    frame, [z.to_dict() for z in zones])
                self._point_error = None
            except RuntimeError as exc:
                self._point_result = None
                self._point_error = str(exc)
            metrics.set_gauge("godavari_head_point_estimate",
                              self._point_result.total_count if self._point_result else 0.0,
                              labels={"camera": self.cam["id"]})
        if now - self._last_density_ts >= config.DENSITY_PERIOD_S:
            self._last_density_ts = now
            try:
                self._density_result = self.registry.density.estimate(
                    frame, [z.to_dict() for z in zones])
                self._density_error = None
            except RuntimeError as exc:
                self._density_result = None
                self._density_error = str(exc)
            metrics.set_gauge("godavari_density_estimate",
                              self._density_result.total_count if self._density_result else 0.0,
                              labels={"camera": self.cam["id"]})

        # ---- confidence and mode router (before alerts, so an alert records
        # the estimation mode of the frame that raised it) -------------------
        # Track continuity = share of detections that got a stable identity.
        # When it collapses, boxes are unreliable and the count must be marked
        # degraded instead of presented as fact.
        track_continuity = (n_tracks / len(boxes)) if len(boxes) else 1.0
        with self._data_lock:
            camera_health = self.frame_health.snapshot().get("status")
        promoted_source = self.registry.calibrated_count_source(self.cam["id"])
        # A camera-wide density total cannot safely drive zone capacities.  It
        # remains a useful advisory estimate, but promotion waits until the
        # backend exposes the density map needed to integrate each zone.
        if (promoted_source == "density" and zones
                and (self._density_result is None
                     or self._density_result.density_map is None)):
            promoted_source = "none"
        if config.MODE_ROUTER:
            route = self.router.route(
                detection,
                track_continuity=track_continuity,
                density=self._density_result,
                density_error=self._density_error,
                points=self._point_result,
                point_error=self._point_error,
                camera_health=camera_health,
                calibrated_density_enabled=config.CALIBRATED_DENSITY_ENABLED,
                calibrated_count_source=promoted_source,
            )
        else:
            route = None
        self._route = route.to_dict() if route is not None else {}
        metrics.set_gauge("godavari_detector_count", len(boxes),
                          labels={"camera": self.cam["id"]})
        self.analytics_health.record_inference(len(boxes), now)
        analytics = self.analytics_health.evaluate(now=now, worker_status="running",
                                                   camera_health=camera_health)

        for i, tid in enumerate(tracked_ids):
            point = tracked_feet[i]
            current_zone = None
            for zone in zones:
                poly = (np.asarray(zone.polygon, np.float32) * [w, h]).astype(np.int32)
                if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0:
                    current_zone = zone.id
                    break
            previous = self._track_zone.get(tid)
            if current_zone is not None:
                if previous and previous[0] and previous[0] != current_zone:
                    key = (previous[0], current_zone)
                    self._zone_transitions[key] = self._zone_transitions.get(key, 0) + 1
                self._track_zone[tid] = (current_zone, now)
        # Drop stale tracker IDs so a recycled ByteTrack ID cannot bridge an old route.
        self._track_zone = {tid: val for tid, val in self._track_zone.items()
                            if now - val[1] <= 30}
        degraded_count = bool(self._route.get("degraded"))
        if degraded_count and not self._was_degraded:
            # Do not carry sustain timers across a spell of untrusted counting.
            self.engine.reset()
        self._was_degraded = degraded_count

        zone_results = []
        for zone in zones:
            poly = (np.asarray(zone.polygon, np.float32) * [w, h]).astype(np.int32)
            area = cv2.contourArea(poly)
            inside = 0
            # Occupancy is based on unique active track IDs rather than raw
            # detections, so repeated detections do not inflate the count.
            for p in tracked_feet:
                if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0:
                    inside += 1
            vx = vy = 0.0
            moving = 0
            for tid, cx, cy in tpoints:
                if cv2.pointPolygonTest(poly, (cx, cy), False) < 0:
                    continue
                flow = flow_by_track.get(tid)
                if flow is not None:
                    vx += flow[0]
                    vy += flow[1]
                    moving += 1
            prev_vx, prev_vy = self._zone_flow.get(zone.id, (0.0, 0.0))
            if moving:
                vx /= moving
                vy /= moving
                # light temporal smoothing keeps the arrow direction stable
                vx = 0.65 * vx + 0.35 * prev_vx
                vy = 0.65 * vy + 0.35 * prev_vy
            else:
                # nobody moving right now: fade the previous vector out
                vx, vy = prev_vx * 0.88, prev_vy * 0.88
                if abs(vx) + abs(vy) < 0.003:
                    vx = vy = 0.0
            self._zone_flow[zone.id] = (vx, vy)
            area_m2 = None
            if self.calibration:
                world_poly = np.asarray(zone.polygon, np.float32) * [w, h]
                area_m2 = polygon_area_world(world_poly, self.calibration)
            # If a dense estimator has passed the camera's promotion gate,
            # apply its own per-zone estimate.  Raw head points are scaled by
            # the model's corrected total; density cells already integrate to
            # people.  Otherwise keep the track count as the detector source.
            zone_count = inside
            source = self._route.get("count_source")
            if source == "head_points_validated" and self._point_result is not None:
                raw = max(1, self._point_result.raw_count)
                point_count = self._point_result.zone_counts.get(zone.id)
                if point_count is not None:
                    zone_count = float(point_count) * self._point_result.total_count / raw
            elif source == "density_validated" and self._density_result is not None:
                density_count = self._density_result.zone_counts.get(zone.id)
                if density_count is not None:
                    zone_count = float(density_count)
            zone.apply_count(zone_count, area, area_m2)
            metrics.set_gauge("godavari_zone_occupancy_ratio",
                              zone.occupancy_pct / 100.0,
                              labels={"camera": self.cam["id"], "zone": zone.id})
            # A lower-bound count cannot justify a capacity breach: in degraded
            # modes the number is incomplete by construction, so the alert is
            # replaced by one advisory (see below) instead of a false breach.
            if degraded_count:
                alerts = []
            else:
                alerts = self.engine.update(zone)
                recovered = (zone.occupancy_pct < zone.warn_pct * 0.8
                             and zone.raw_count < zone.capacity * 0.9)
                if recovered:
                    self.registry.close_recovered_alerts(self.cam["id"], zone.id)
            for alert in alerts:
                # Every alert carries the geometry/calibration/model versions it
                # fired under, so an after-event audit can reconstruct it.
                alert["zone_id"] = zone.id
                alert["context"] = {
                    "zones_version": self.zone_version,
                    "calibration_version": (self.calibration or {}).get("version"),
                    "estimation_mode": self._route.get("estimation_mode"),
                    "count_source": self._route.get("count_source"),
                    "detector_count": self._route.get("detector_count"),
                    "density_count": self._route.get("density_count"),
                    "head_point_count": self._route.get("head_point_count"),
                    "model": (self._last_detection.model_name if self._last_detection else None),
                    "model_version": (self._last_detection.model_version if self._last_detection else None),
                }
                evidence_path = self._save_alert_evidence(alert["id"], frame)
                if evidence_path:
                    alert["context"]["evidence"] = f"/api/alerts/{alert['id']}/evidence"
                store.add_alert(alert)
                self.registry.push_alert(alert)
                self.registry.dispatch_alert(alert)
            zone_results.append({
                "id": zone.id, "name": zone.name, "capacity": zone.capacity,
                "count": int(round(zone.count)), "occupancy_pct": round(zone.occupancy_pct, 1),
                "density_per_10k": round(zone.density_per_10k, 1),
                "density_m2": round(zone.density_m2, 2)
                if zone.density_m2 is not None else None,
                "detector_count": inside,
                "head_point_count": (round(self._point_result.zone_counts.get(zone.id, 0), 1)
                                     if self._point_result is not None else None),
                "density_count": (round(self._density_result.zone_counts.get(zone.id, 0), 1)
                                  if self._density_result is not None else None),
                "level": zone.level,
                "flow": [round(vx, 4), round(vy, 4)], "moving": moving,
            })

        for zr in zone_results:
            zr["count_source"] = self._route.get("count_source")
            zr["degraded"] = bool(self._route.get("degraded"))

        if degraded_count:
            self._maybe_dense_advisory(now, len(boxes), zones)

        total = int(len(boxes))
        selected_total = self._route.get("selected_count")
        operational_total = int(selected_total) if selected_total is not None else total
        self.total_count = total if self.total_count == 0 else self.total_count * 0.6 + total * 0.4
        cam_cap = sum(z.capacity for z in zones)
        cam_occ = 100.0 * (operational_total / cam_cap) if cam_cap else 0.0
        if zones:
            worst = max(zones, key=lambda z: SEVERITY[z.level])
            cam_level = worst.level
        else:
            cam_level = level_for(cam_occ, config.DEFAULT_WARN_PCT)

        with self._data_lock:
            self.last_boxes = boxes
            self.metrics = {
                "status": "running",
                "ts": time.time(),
                "infer_fps": self._rolling_infer_fps(),
                "frames": self._frames_total,
                "count": total,
                "operational_count": operational_total,
                "tracked_count": n_tracks,
                "raw_detection_count": total,
                "count_smooth": int(round(self.total_count)),
                "active_tracks": n_tracks,
                "zones_defined": len(zones),
                "capacity": cam_cap,
                "occupancy_pct": round(cam_occ, 1) if cam_cap else None,
                "level": cam_level,
                "zones": zone_results,
                "frame_size": [w, h],
                "heatmap": self.heatmap.encode(),
                "health": self.frame_health.snapshot(),
                "decoder": self.decoder,
                # --- improvement.md: explicit estimation state ------------
                "estimation_mode": self._route.get("estimation_mode", "detector"),
                "mode": self._route.get("mode"),
                "selected_count": selected_total if self._route else total,
                "density_count": self._route.get("density_count"),
                "head_point_count": self._route.get("head_point_count"),
                "count_source": self._route.get("count_source", "detector"),
                "count_confidence": self._route.get("confidence", "medium"),
                "degraded": bool(self._route.get("degraded")),
                "review_required": bool(self._route.get("review_required")),
                "degraded_reason": self._route.get("reason"),
                # how the number was produced: plain pass or recovered detail tier
                "tier": self._route.get("tier", "primary"),
                "primary_count": self._route.get("primary_count"),
                # Mean of two agreeing dense estimators, shown as ≈N. Still advisory.
                "consensus_count": self._route.get("consensus_count"),
                "estimator_spread_pct": self._route.get("estimator_spread_pct"),
                "route_detail": self._route,
                "analytics_health": analytics,
                "detector": (detection.to_dict() if detection is not None else None),
                "head_point_model": (self._point_result.model_name if self._point_result else None),
                "head_point_model_version": (self._point_result.model_version if self._point_result else None),
                "density_model": (self._density_result.model_name if self._density_result else None),
                "density_model_version": (self._density_result.model_version if self._density_result else None),
                # Both dense estimators report their own latency and zone
                # counts, so an operator can see what the advisory cost and
                # where it put people, not just the camera-wide total.
                "estimators": {
                    "head_points": (None if self._point_result is None else {
                        "count": round(self._point_result.total_count, 1),
                        "raw_count": self._point_result.raw_count,
                        "zone_counts": self._point_result.zone_counts,
                        "inference_ms": round(self._point_result.inference_ms, 1),
                        "confidence": self._point_result.confidence,
                    }),
                    "density": (None if self._density_result is None else {
                        "count": round(self._density_result.total_count, 1),
                        "zone_counts": self._density_result.zone_counts,
                        "inference_ms": round(self._density_result.inference_ms, 1),
                        "confidence": self._density_result.confidence,
                    }),
                    "head_point_error": self._point_error,
                    "density_error": self._density_error,
                },
                "zone_version": self.zone_version,
                "throughput": self._throughput(now),
                "count_lines": [{**line, **self.line_counts.get(line.get("id", line.get("name", "line")), {})}
                                for line in self.count_lines],
            }
        if now - self._last_sample_ts >= 15:
            store.add_sample(self.cam["id"], now, total,
                             round(cam_occ, 1) if cam_cap else None, zone_results,
                             [{**line, **self.line_counts.get(line.get("id", line.get("name", "line")), {})}
                              for line in self.count_lines],
                             [{"from": a, "to": b, "count": count}
                              for (a, b), count in self._zone_transitions.items()])
            self._last_sample_ts = now
        if now - self._last_heatmap_ts >= 300:
            store.add_heatmap(self.cam["id"], now, self.metrics.get("heatmap") or {})
            self._last_heatmap_ts = now

    @staticmethod
    def _save_alert_evidence(alert_id: str, frame: np.ndarray) -> str | None:
        """Persist one compressed frame for a rare alert, outside SQLite."""
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(alert_id)):
            return None
        try:
            os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
            ok, encoded = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                return None
            path = os.path.join(config.EVIDENCE_DIR, f"{alert_id}.jpg")
            tmp = path + ".tmp"
            with open(tmp, "wb") as output:
                output.write(encoded.tobytes())
            os.replace(tmp, path)
            return path
        except OSError:
            metrics.inc("godavari_alert_evidence_write_failures_total")
            return None

    def _maybe_dense_advisory(self, now: float, total: int, zones) -> None:
        """One honest alert while counting is untrusted, instead of false breaches.

        In a packed scene the recovered count is a lower bound, so a capacity
        percentage computed from it is meaningless in both directions. The
        operator gets told that counting is unreliable and that they must look,
        which is the only safe instruction available.
        """
        if now - self._last_dense_advisory < config.DEFAULT_ALERT_COOLDOWN_S:
            return
        self._last_dense_advisory = now
        detail = self._route.get("detail") or {}
        primary = detail.get("primary_count")
        reason = self._route.get("reason") or "counting unreliable"
        consensus = self._route.get("consensus_count")
        head_points = self._route.get("head_point_count")
        density = self._route.get("density_count")
        if consensus is not None and head_points and density:
            # Two independent models agree: an operator can act on ~N with the
            # spread visible, instead of only being told the detector is low.
            spread = self._route.get("estimator_spread_pct") or 0.0
            message = (f"Dense crowd: about {consensus:.0f} people — PET {head_points:.0f} "
                       f"and DM-Count {density:.0f} agree within {spread:.0f}%; body "
                       f"detector saw {total}. Estimate, not a validated count — "
                       "manual observation required.")
        else:
            message = (f"Dense crowd: {total} people detected is a lower bound, not a "
                       f"count ({reason})")
            if primary is not None:
                message += f"; full-frame pass saw {primary}. Manual observation required."
            else:
                message += ". Manual observation required."
        alert = {
            "id": uuid.uuid4().hex,
            "cam_id": self.cam["id"],
            "cam_name": self.cam.get("name"),
            "zone": zones[0].name if zones else "Camera",
            "level": "WARNING",
            "message": message,
            "ts": time.time(),
            "zone_id": zones[0].id if zones else None,
            "context": {
                "zones_version": self.zone_version,
                "estimation_mode": self._route.get("estimation_mode"),
                "count_source": self._route.get("count_source"),
                "detector_count": self._route.get("detector_count"),
                "primary_count": primary,
                "tier": detail.get("tier"),
                "model": (self._last_detection.model_name if self._last_detection else None),
                "model_version": (self._last_detection.model_version if self._last_detection else None),
            },
        }
        store.add_alert(alert)
        self.registry.push_alert(alert)
        self.registry.dispatch_alert(alert)

    def _throughput(self, now: float) -> dict:
        """People per minute crossing each configured line.

        For ghat management the rate of inflow matters more than the
        instantaneous head-count: this is the number that predicts a surge.
        """
        snapshot = {k: dict(v) for k, v in self.line_counts.items()}
        self._line_history.append((now, snapshot))
        window = [h for h in self._line_history if now - h[0] <= 60.0]
        if len(window) < 2:
            return {}
        t0, first = window[0]
        t1, last = window[-1]
        minutes = max((t1 - t0) / 60.0, 1e-6)
        out = {}
        for key, values in last.items():
            prior = first.get(key, {})
            entry = max(0, values.get("entry", 0) - prior.get("entry", 0))
            exit_ = max(0, values.get("exit", 0) - prior.get("exit", 0))
            out[key] = {"entry_per_min": round(entry / minutes, 1),
                        "exit_per_min": round(exit_ / minutes, 1)}
        return out

    def _rolling_infer_fps(self) -> float:
        ts = list(self._infer_ts)
        if len(ts) < 2:
            return 0.0
        span = ts[-1] - ts[0]
        return round((len(ts) - 1) / span, 1) if span > 0 else 0.0

    # ------------------------------------------------------------------ #
    def _annotate(self, frame: np.ndarray):
        """Burn detection boxes into the JPEG only.

        Zones, heatmap, flow arrows, counts and the HUD are drawn by the
        dashboard's overlay canvas: they stay vector-crisp at any size, can
        be toggled by the operator, and cost the worker nothing.
        """
        with self._data_lock:
            boxes = self.last_boxes
            # Copy BEFORE annotating: cv2.rectangle draws in place, so a copy
            # taken afterwards would carry the boxes into the "clean" view.
            self.latest_frame = frame.copy()
            self._frame_seq += 1

        if self.show_detections:
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        if ok:
            with self._data_lock:
                self.latest_jpeg = buf.tobytes()

    def raw_jpeg(self) -> bytes | None:
        """JPEG of the last frame without detection boxes, for split views."""
        with self._data_lock:
            frame, seq = self.latest_frame, self._frame_seq
            if frame is not None and self._raw_cache_seq == seq:
                return self._raw_cache
            if frame is None:
                return None
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        if not ok:
            return None
        data = buf.tobytes()
        with self._data_lock:
            self._raw_cache, self._raw_cache_seq = data, seq
        return data


def build_density_estimator():
    """Density is opt-in: an unvalidated checkpoint must not silently count."""
    if config.DENSITY_BACKEND == "crowdshield":
        try:
            return CrowdShieldDensityEstimator(validated=config.DENSITY_VALIDATED)
        except Exception as exc:  # noqa: BLE001 - detector operation continues
            store.add_audit("density.load.failed", "system", "crowdshield",
                            {"error": str(exc)})
            return NullDensityEstimator()
    if not config.DENSITY_MODEL:
        return NullDensityEstimator()
    try:
        return TorchscriptDensityEstimator(config.DENSITY_MODEL,
                                          validated=config.DENSITY_VALIDATED)
    except Exception as exc:  # noqa: BLE001 - never block startup on density
        store.add_audit("density.load.failed", "system", config.DENSITY_MODEL,
                        {"error": str(exc)})
        return NullDensityEstimator()


def build_point_estimator():
    """Load the optional PET path without making detector startup depend on it."""
    if config.POINT_BACKEND != "crowdshield":
        return NullPointEstimator()
    try:
        return CrowdShieldPointEstimator(validated=config.POINT_MODEL_VALIDATED)
    except Exception as exc:  # noqa: BLE001 - dense inference is an optional path
        store.add_audit("points.load.failed", "system", "crowdshield",
                        {"error": str(exc)})
        return NullPointEstimator()


class CameraRegistry:
    def __init__(self, model):
        self.model = model
        self.scheduler = InferenceScheduler(model, batch_size=config.BATCH_SIZE,
                                           max_frame_age_ms=config.MAX_FRAME_AGE_MS)
        self._lock = threading.Lock()
        self.cameras: dict[str, dict] = {}
        self.workers: dict[str, CameraWorker] = {}
        self.alerts: deque = deque(maxlen=200)
        self.density = build_density_estimator()
        self.points = build_point_estimator()
        # Durable outbox: alerts survive restarts and long network outages.
        self.outbox = AlertOutbox()
        self.outbox.start()
        # Read the saved cameras immediately. Waiting until the model has loaded
        # meant an API call in those first seconds (add/delete/zone edit) saved a
        # registry that had not been restored yet, wiping cameras.json.
        self._load_config()

    def calibrated_count_source(self, cam_id: str) -> str:
        """Return an active, unexpired promotion only for this camera."""
        with self._lock:
            camera = self.cameras.get(cam_id) or {}
            policy = camera.get("estimation_policy") or {}
            validation = policy.get("validation") or {}
            source = str(policy.get("selected_source") or "none")
        if source not in {"head_point", "density"}:
            return "none"
        if not validation.get("approved") or float(validation.get("expires_at") or 0) < time.time():
            return "none"
        if source == "head_point" and not self.points.validated:
            return "none"
        if source == "density" and not self.density.validated:
            return "none"
        return source

    def validation_summary(self, cam_id: str) -> dict:
        with self._lock:
            camera = self.cameras.get(cam_id)
            if camera is None:
                raise KeyError(cam_id)
            policy = dict(camera.get("estimation_policy") or {})
            samples = list(policy.get("samples") or [])
            validation = dict(policy.get("validation") or {})
        absolute = [abs(float(s["model_count"]) - float(s["manual_count"]))
                    for s in samples]
        relative = [error / max(1.0, float(s["manual_count"]))
                    for error, s in zip(absolute, samples)]
        return {
            "selected_source": policy.get("selected_source", "none"),
            "samples": len(samples),
            "mae": round(sum(absolute) / len(absolute), 2) if absolute else None,
            "mape": round(sum(relative) / len(relative), 4) if relative else None,
            "approved": bool(validation.get("approved")),
            "approved_by": validation.get("approved_by"),
            "approved_at": validation.get("approved_at"),
            "expires_at": validation.get("expires_at"),
            "active_source": self.calibrated_count_source(cam_id),
            "calibration_version": (camera.get("calibration") or {}).get("version"),
        }

    def add_validation_sample(self, cam_id: str, source: str,
                              manual_count: float, model_count: float,
                              recorded_by: str, note: str = "") -> dict:
        if source not in {"head_point", "density"}:
            raise ValueError("source must be head_point or density")
        if min(manual_count, model_count) < 0:
            raise ValueError("counts must be zero or greater")
        with self._lock:
            camera = self.cameras.get(cam_id)
            if camera is None:
                raise KeyError(cam_id)
            policy = camera.setdefault("estimation_policy", {})
            samples = policy.setdefault("samples", [])
            samples.append({"source": source, "manual_count": float(manual_count),
                            "model_count": float(model_count), "recorded_at": time.time(),
                            "recorded_by": recorded_by[:60], "note": note[:240]})
            # A new observation makes a prior sign-off stale until reviewed.
            policy["validation"] = {"approved": False}
            if len(samples) > 500:
                del samples[:-500]
        self.save()
        summary = self.validation_summary(cam_id)
        store.add_audit("estimation.validation.sample", recorded_by, cam_id,
                        {"source": source, "manual_count": manual_count,
                         "model_count": model_count, "samples": summary["samples"]})
        return summary

    def promote_estimation_source(self, cam_id: str, source: str,
                                  approved_by: str) -> dict:
        if source not in {"head_point", "density"}:
            raise ValueError("source must be head_point or density")
        with self._lock:
            camera = self.cameras.get(cam_id)
            if camera is None:
                raise KeyError(cam_id)
            if not camera.get("calibration"):
                raise ValueError("ground-plane calibration is required before promotion")
            policy = camera.setdefault("estimation_policy", {})
            samples = [s for s in policy.get("samples", []) if s.get("source") == source]
        if len(samples) < config.VALIDATION_MIN_SAMPLES:
            raise ValueError(f"{config.VALIDATION_MIN_SAMPLES} validation samples are required")
        errors = [abs(float(s["model_count"]) - float(s["manual_count"])) /
                  max(1.0, float(s["manual_count"])) for s in samples]
        mape = sum(errors) / len(errors)
        if mape > config.VALIDATION_MAX_MAPE:
            raise ValueError(f"validation MAPE {mape:.1%} exceeds {config.VALIDATION_MAX_MAPE:.1%}")
        estimator = self.points if source == "head_point" else self.density
        if not estimator.validated:
            raise ValueError(f"{source} checkpoint is not enabled as validated on this node")
        now = time.time()
        with self._lock:
            policy = camera.setdefault("estimation_policy", {})
            policy["selected_source"] = source
            policy["validation"] = {
                "approved": True, "approved_by": approved_by[:60], "approved_at": now,
                "expires_at": now + config.VALIDATION_TTL_DAYS * 86400,
                "calibration_version": camera["calibration"].get("version"),
                "sample_count": len(samples), "mape": round(mape, 4),
            }
        self.save()
        summary = self.validation_summary(cam_id)
        store.add_audit("estimation.validation.promote", approved_by, cam_id,
                        {"source": source, **summary})
        return summary

    def revoke_estimation_source(self, cam_id: str, by: str, reason: str = "") -> dict:
        with self._lock:
            camera = self.cameras.get(cam_id)
            if camera is None:
                raise KeyError(cam_id)
            policy = camera.setdefault("estimation_policy", {})
            policy["selected_source"] = "none"
            policy["validation"] = {"approved": False, "revoked_by": by[:60],
                                    "revoked_at": time.time(), "reason": reason[:240]}
        self.save()
        summary = self.validation_summary(cam_id)
        store.add_audit("estimation.validation.revoke", by, cam_id, {"reason": reason[:240]})
        return summary

    # lifecycle ---------------------------------------------------------
    def add(self, cfg: dict, persist: bool = True):
        with self._lock:
            self.cameras[cfg["id"]] = cfg
            if cfg.get("profile"):
                self.model.set_camera_profile(cfg["id"], cfg["profile"])
            worker = CameraWorker(cfg, self.model, self)
            self.workers[cfg["id"]] = worker
        worker.start()
        if persist:
            self.save()

    def remove(self, cam_id: str):
        with self._lock:
            worker = self.workers.pop(cam_id, None)
            camera = self.cameras.pop(cam_id, None)
        if worker:
            worker.stop()
            worker.join(timeout=4)
        try:
            self.model.set_camera_profile(cam_id, "global")
        except (AttributeError, ValueError):
            pass
        self.save()
        from .mongo_sync import mirror
        mirror.delete("cameras", cam_id)
        uploaded_file_deleted = False
        if camera and camera.get("source_type") == "file" and camera.get("path"):
            source_path = os.path.realpath(config.resolve_media(camera["path"]))
            uploads_root = os.path.realpath(config.UPLOADS_DIR)
            try:
                is_managed_upload = os.path.commonpath((uploads_root, source_path)) == uploads_root
            except ValueError:
                is_managed_upload = False
            with self._lock:
                remaining_paths = {os.path.realpath(c.get("path", ""))
                                   for c in self.cameras.values() if c.get("path")}
                if (is_managed_upload and source_path not in remaining_paths
                        and os.path.isfile(source_path)):
                    os.remove(source_path)
                    uploaded_file_deleted = True
        from . import store
        store.add_audit("camera.remove", "operator", cam_id,
                        {"name": camera.get("name"),
                         "uploaded_file_deleted": uploaded_file_deleted} if camera else {})

    def set_zones(self, cam_id: str, zone_dicts: list[dict]):
        cleaned = []
        for d in zone_dicts:
            cleaned.append({
                "id": d["id"], "name": d["name"], "polygon": d["polygon"],
                "capacity": d["capacity"], "warn_pct": d.get("warn_pct", config.DEFAULT_WARN_PCT),
            })
        with self._lock:
            cam = self.cameras.get(cam_id)
            if cam is None:
                raise KeyError(cam_id)
            # Geometry version: a zone redraw invalidates comparisons with
            # history, so every alert records which version it fired under.
            cam["zones"] = cleaned
            cam["zones_version"] = int(cam.get("zones_version", 0)) + 1
            cam["zones_updated_at"] = time.time()
            version = cam["zones_version"]
            worker = self.workers.get(cam_id)
        if worker:
            worker.zone_version = version
            worker.set_zones(cleaned)
            worker.router.reset()
        self.save()
        store.add_audit("zones.update", "operator", cam_id,
                        {"zones_version": version, "zone_count": len(cleaned)})

    def set_count_lines(self, cam_id: str, lines: list[dict]):
        with self._lock:
            cam = self.cameras.get(cam_id)
            if cam is None:
                raise KeyError(cam_id)
            cam["count_lines"] = lines
            worker = self.workers.get(cam_id)
        if worker:
            worker.count_lines = list(lines)
            worker.line_counts = {line["id"]: {"entry": 0, "exit": 0} for line in lines}
        self.save()

    def set_display(self, cam_id: str, display: dict):
        with self._lock:
            cam = self.cameras.get(cam_id)
            if cam is None:
                raise KeyError(cam_id)
            cam["display"] = {"detections": bool(display.get("detections", True))}
            worker = self.workers.get(cam_id)
        if worker:
            worker.show_detections = cam["display"]["detections"]
        self.save()

    def set_camera_profile(self, cam_id: str, profile: str):
        with self._lock:
            cam = self.cameras.get(cam_id)
            if cam is None:
                raise KeyError(cam_id)
            cam["profile"] = profile
            self.model.set_camera_profile(cam_id, profile)
        self.save()
        store.add_audit("camera.profile", "operator", cam_id, {"profile": profile})

    def get(self, cam_id: str) -> dict:
        with self._lock:
            return self.cameras.get(cam_id)

    # read models ---------------------------------------------------------
    def snapshot(self) -> list[dict]:
        out = []
        with self._lock:
            items = list(self.cameras.items())
        for cam_id, cam in items:
            worker = self.workers.get(cam_id)
            metrics_snapshot = worker.metrics if worker else None
            status = (metrics_snapshot.get("status", "stopped")
                      if metrics_snapshot else "stopped")
            health_status = (metrics_snapshot or {}).get("health", {}).get("status")
            if status == "running" and health_status in {
                "stale", "frozen", "dark", "low_contrast", "blurry"
            }:
                status = "degraded"
            out.append({
                "id": cam_id,
                "name": cam["name"],
                "source_type": cam["source_type"],
                "zones": cam.get("zones", []),
                "zones_version": int(cam.get("zones_version", 0)),
                "count_lines": cam.get("count_lines", []),
                "display": cam.get("display", {"detections": True}),
                "profile": cam.get("profile", "global"),
                "estimation_policy": self.validation_summary(cam_id),
                "status": status,
                "detail": (metrics_snapshot.get("detail") if metrics_snapshot else None),
                "metrics": metrics_snapshot,
            })
        return out

    def push_alert(self, alert: dict):
        self.alerts.append(alert)

    def close_recovered_alerts(self, cam_id: str, zone_id: str):
        closed = store.close_recovered_zone_alerts(cam_id, zone_id)
        if not closed:
            return
        now = time.time()
        for alert in self.alerts:
            if (alert.get("cam_id") == cam_id and alert.get("zone_id") == zone_id
                    and alert.get("status", "open") == "open"
                    and alert.get("level") in ("WARNING", "OVERCROWDED")):
                alert.update(status="closed", closed_by="system:recovery", closed_ts=now)
        metrics.inc("godavari_alerts_auto_closed_total", closed)
        store.add_audit("alert.auto_close", "system:recovery", zone_id,
                        {"camera_id": cam_id, "count": closed})

    def dispatch_alert(self, alert: dict):
        self.outbox.enqueue(alert)

    def analytics_snapshot(self) -> dict:
        """Analytics health (model-side dead-man switch) for the dashboard."""
        out = {}
        for cam_id, worker in list(self.workers.items()):
            out[cam_id] = worker.analytics_health.evaluate(
                worker_status=(worker.metrics or {}).get("status", "running"),
                camera_health=(worker.metrics or {}).get("health", {}).get("status"))
        return out

    def recent_alerts(self, limit: int = 60) -> list[dict]:
        data = list(self.alerts)[-limit:]
        return list(reversed(data))

    # persistence ---------------------------------------------------------
    def _config_path(self) -> str:
        return os.path.join(config.DATA_DIR, "cameras.json")

    def save(self):
        import json

        with self._lock:
            cams = [dict(c) for c in self.cameras.values()]
        tmp = self._config_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cams, fh, indent=2)
        os.replace(tmp, self._config_path())
        from .mongo_sync import mirror_camera
        for cam in cams:
            mirror_camera(cam)

    def _load_config(self):
        """Restore saved cameras into memory (no workers yet)."""
        import json

        path = self._config_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                cams = json.load(fh)
        except Exception:
            return
        if not isinstance(cams, list):
            return
        with self._lock:
            for cam in cams:
                if isinstance(cam, dict) and cam.get("id"):
                    self.cameras.setdefault(cam["id"], cam)

    def load_and_start(self):
        """Start a worker for every restored camera whose media is available."""
        with self._lock:
            cams = list(self.cameras.values())
        for cam in cams:
            try:
                exists = (cam.get("source_type") in ("rtsp", "webcam")
                          or os.path.exists(
                              config.resolve_media(cam.get("path") or "")))
            except Exception:
                exists = False
            if exists and cam["id"] not in self.workers:
                self.add(cam, persist=False)
