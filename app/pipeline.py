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
from .heatmap import HeatmapAccumulator
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
        self.engine = AlertEngine(cam["id"], cam["name"],
                                  config.DEFAULT_ALERT_SUSTAIN_S,
                                  config.DEFAULT_ALERT_COOLDOWN_S)
        self.tracker = sv.ByteTrack()
        self.heatmap = HeatmapAccumulator()
        self._track_prev: dict[int, tuple[float, float, float]] = {}
        # Per-zone flow memory so the arrows ease out instead of blinking
        # off on frames where no track happens to be moving.
        self._zone_flow: dict[str, tuple[float, float]] = {}

        self.latest_jpeg: bytes | None = None
        self.metrics: dict = {"status": "starting"}
        self.last_boxes = np.zeros((0, 4), np.float32)
        self._data_lock = threading.Lock()
        self._infer_ts = deque(maxlen=24)
        self._frames_total = 0
        self.total_count = 0.0

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
        cap = cv2.VideoCapture(src)
        if self.cam["source_type"] == "rtsp":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _process(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        now = time.time()
        self._infer_ts.append(now)

        boxes, confs = self.model.detect(frame)
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
        zone_results = []
        for zone in zones:
            poly = (np.asarray(zone.polygon, np.float32) * [w, h]).astype(np.int32)
            area = cv2.contourArea(poly)
            inside = 0
            for p in feet:
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
            zone.apply_count(inside, area)
            for alert in self.engine.update(zone):
                store.add_alert(alert)
                self.registry.push_alert(alert)
            zone_results.append({
                "id": zone.id, "name": zone.name, "capacity": zone.capacity,
                "count": int(round(zone.count)), "occupancy_pct": round(zone.occupancy_pct, 1),
                "density_per_10k": round(zone.density_per_10k, 1), "level": zone.level,
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
                "count_smooth": int(round(self.total_count)),
                "active_tracks": n_tracks,
                "zones_defined": len(zones),
                "capacity": cam_cap,
                "occupancy_pct": round(cam_occ, 1) if cam_cap else None,
                "level": cam_level,
                "zones": zone_results,
                "frame_size": [w, h],
                "heatmap": self.heatmap.encode(),
            }

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
        self._lock = threading.Lock()
        self.cameras: dict[str, dict] = {}
        self.workers: dict[str, CameraWorker] = {}
        self.alerts: deque = deque(maxlen=200)
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
            status = (worker.metrics.get("status", "stopped") if worker else "stopped")
            out.append({
                "id": cam_id,
                "name": cam["name"],
                "source_type": cam["source_type"],
                "zones": cam.get("zones", []),
                "status": status,
                "detail": (worker.metrics.get("detail") if worker else None),
                "metrics": (worker.metrics if worker else None),
            })
        return out

    def push_alert(self, alert: dict):
        self.alerts.append(alert)

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
