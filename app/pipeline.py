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
from .alerting import AlertEngine
from .calibration import polygon_area_world
from .decoder import open_capture
from .delivery import AlertDispatcher
from .heatmap import HeatmapAccumulator
from .health import FrameHealth
from . import metrics
from .scheduler import InferenceScheduler
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

        cap = self._open_capture()
        if cap is None:
            self.metrics = {"status": "error",
                            "detail": f"could not open source: {self.cam.get('path')!r}"}
            return

        is_live = self.cam["source_type"] in ("rtsp", "webcam")
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
                # live stream hiccup -> try to reconnect for ~12s
                cap.release()
                recovered = None
                for _ in range(6):
                    if self._stop_event.is_set():
                        break
                    time.sleep(2)
                    recovered = self._open_capture()
                    if recovered is not None:
                        break
                if recovered is None:
                    with self._data_lock:
                        self.metrics = {"status": "error", "detail": "stream lost"}
                    break
                cap = recovered
                continue

            self.frame_health.frame_received(frame)
            metrics.inc("godavari_frames_total")
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

        boxes, confs = self.registry.scheduler.infer(self.cam["id"], frame)
        dets = sv.Detections(
            xyxy=boxes, confidence=confs,
            class_id=np.zeros(len(boxes), dtype=int),
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
            zone.apply_count(inside, area, area_m2)
            for alert in self.engine.update(zone):
                store.add_alert(alert)
                self.registry.push_alert(alert)
                self.registry.dispatch_alert(alert)
            zone_results.append({
                "id": zone.id, "name": zone.name, "capacity": zone.capacity,
                "count": int(round(zone.count)), "occupancy_pct": round(zone.occupancy_pct, 1),
                "density_per_10k": round(zone.density_per_10k, 1),
                "density_m2": round(zone.density_m2, 2)
                if zone.density_m2 is not None else None,
                "level": zone.level,
                "flow": [round(vx, 4), round(vy, 4)], "moving": moving,
            })

        total = int(len(boxes))
        self.total_count = total if self.total_count == 0 else self.total_count * 0.6 + total * 0.4
        cam_cap = sum(z.capacity for z in zones)
        cam_occ = 100.0 * (total / cam_cap) if cam_cap else 0.0
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
                "estimation_mode": "detector",
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

        if self.show_detections:
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        if ok:
            with self._data_lock:
                self.latest_jpeg = buf.tobytes()


class CameraRegistry:
    def __init__(self, model):
        self.model = model
        self.scheduler = InferenceScheduler(model)
        self._lock = threading.Lock()
        self.cameras: dict[str, dict] = {}
        self.workers: dict[str, CameraWorker] = {}
        self.alerts: deque = deque(maxlen=200)
        self.dispatcher = AlertDispatcher()
        # Read the saved cameras immediately. Waiting until the model has loaded
        # meant an API call in those first seconds (add/delete/zone edit) saved a
        # registry that had not been restored yet, wiping cameras.json.
        self._load_config()

    # lifecycle ---------------------------------------------------------
    def add(self, cfg: dict, persist: bool = True):
        with self._lock:
            self.cameras[cfg["id"]] = cfg
            worker = CameraWorker(cfg, self.model, self)
            self.workers[cfg["id"]] = worker
        worker.start()
        if persist:
            self.save()

    def remove(self, cam_id: str):
        with self._lock:
            worker = self.workers.pop(cam_id, None)
            self.cameras.pop(cam_id, None)
        if worker:
            worker.stop()
            worker.join(timeout=4)
        self.save()

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
            cam["zones"] = cleaned
            worker = self.workers.get(cam_id)
        if worker:
            worker.set_zones(cleaned)
        self.save()

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
                "count_lines": cam.get("count_lines", []),
                "display": cam.get("display", {"detections": True}),
                "status": status,
                "detail": (metrics_snapshot.get("detail") if metrics_snapshot else None),
                "metrics": metrics_snapshot,
            })
        return out

    def push_alert(self, alert: dict):
        self.alerts.append(alert)

    def dispatch_alert(self, alert: dict):
        self.dispatcher.dispatch(alert)

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
