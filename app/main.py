"""FastAPI app: REST API, UploadFile/stream upload, MJPEG streams, WebSocket push."""
import asyncio
import os
import time
import urllib.parse

from fastapi import (FastAPI, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, ingest, metrics
from .mongo_sync import mirror as mongo_mirror, mirror_camera
from .calibration import validate_calibration
from .detector import ModelManager
from .pipeline import CameraRegistry, new_camera_id

model = ModelManager(config.MODEL_NAME, config.FALLBACK_MODELS,
                     config.CONF_THRESHOLD, config.IMGSZ)
registry = CameraRegistry(model)

app = FastAPI(title="Godavari Crowd Monitor")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

_ws_clients: set[WebSocket] = set()
START_TS = time.time()


@app.on_event("startup")
async def startup():
    mongo_mirror.start()
    for camera in list(registry.cameras.values()):
        mirror_camera(camera)
    asyncio.create_task(_bootstrap())


async def _bootstrap():
    await asyncio.to_thread(model.load)
    await asyncio.to_thread(registry.load_and_start)
    await broadcaster()


async def broadcaster():
    """Push a full metrics snapshot to every dashboard ~twice a second."""
    while True:
        await asyncio.sleep(config.WS_PERIOD_S)
        if not _ws_clients:
            continue
        payload = {
            "type": "snapshot",
            "ts": time.time(),
            "system": system_info(),
            "cameras": registry.snapshot(),
            "alerts": registry.recent_alerts(60),
        }
        dead = []
        for ws in list(_ws_clients):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 - client went away
                dead.append(ws)
        for ws in dead:
            _ws_clients.discard(ws)


def system_info() -> dict:
    info = model.info()
    snap = registry.snapshot()
    info["cameras_total"] = len(snap)
    info["cameras_running"] = sum(1 for c in snap if c["status"] == "running")
    info["samples"] = len(ingest.list_samples())
    info["uptime_s"] = int(time.time() - START_TS)
    info["ws_clients"] = len(_ws_clients)
    info["scheduler"] = registry.scheduler.stats()
    info["outbox"] = registry.outbox.stats()
    info["density"] = registry.density.info()
    info["mongodb"] = mongo_mirror.status()
    return info


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


@app.get("/")
async def index():
    return FileResponse(os.path.join(config.STATIC_DIR, "index.html"))


# --------------------------------------------------------------- cameras
@app.post("/api/cameras")
async def add_camera(request: Request):
    form = await request.form()

    def cam_name(fallback: str) -> str:
        name = (form.get("name") or "").strip()
        return name or fallback

    loop = str(form.get("loop", "true")).lower() != "false"
    upload: UploadFile | None = form.get("file")
    if upload is not None and upload.filename:
        try:
            path = await asyncio.to_thread(ingest.save_upload, upload)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        cfg = make_cfg(cam_name(os.path.splitext(os.path.basename(upload.filename))[0]),
                       "file", path, loop)
    elif form.get("sample"):
        try:
            path = ingest.sample_path(str(form.get("sample")))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        cfg = make_cfg(cam_name(os.path.splitext(os.path.basename(path))[0]),
                       "file", path, loop)
    elif form.get("source"):
        src = str(form.get("source")).strip()
        low = src.lower()
        if low.startswith("rtsp://") or low.startswith("rtsps://") or \
           low.startswith("http://") or low.startswith("https://"):
            cfg = make_cfg(cam_name(src[:40]), "rtsp", src, True)
        elif src.isdigit():
            cfg = make_cfg(cam_name(f"Webcam {src}"), "webcam", src, False)
        else:
            raise HTTPException(400, "source must be rtsp://... or a webcam index")
    else:
        raise HTTPException(400, "provide a video file, sample name, or source url")

    registry.add(cfg)
    return cfg


def make_cfg(name: str, source_type: str, path: str, loop: bool) -> dict:
    cfg = {"id": new_camera_id(), "name": name, "source_type": source_type,
           "path": path, "zones": [], "created": time.time()}
    if source_type == "file":
        cfg["loop"] = loop
    return cfg


@app.get("/api/cameras")
async def list_cameras():
    return registry.snapshot()


@app.get("/api/cameras/{cam_id}/diagnostics")
async def camera_diagnostics(cam_id: str):
    worker = registry.workers.get(cam_id)
    if worker is None:
        raise HTTPException(404, "camera not running")
    metrics_snapshot = dict(worker.metrics)
    return {
        "camera_id": cam_id,
        "status": metrics_snapshot.get("status"),
        "health": metrics_snapshot.get("health"),
        "decoder": getattr(worker, "decoder", "unknown"),
        "frames": metrics_snapshot.get("frames", 0),
        "infer_fps": metrics_snapshot.get("infer_fps", 0),
        "frame_size": metrics_snapshot.get("frame_size"),
        "model": model.info(),
    }


@app.get("/api/cameras/{cam_id}")
async def get_camera(cam_id: str):
    for cam in registry.snapshot():
        if cam["id"] == cam_id:
            return cam
    raise HTTPException(404, "camera not found")


@app.put("/api/cameras/{cam_id}/profile")
async def set_camera_profile(cam_id: str, request: Request):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    try:
        body = await request.json()
        profile = str(body.get("profile", "")).strip().lower()
        registry.set_camera_profile(cam_id, profile)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(400, str(exc) or "body must contain a profile")
    return {"ok": True, "camera_id": cam_id, "profile": profile}


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    registry.remove(cam_id)
    return {"ok": True}


@app.put("/api/cameras/{cam_id}/zones")
async def put_zones(cam_id: str, request: Request):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON: {\"zones\": [...]}")
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be {\"zones\": [...]}")
    zones = body.get("zones")
    if not isinstance(zones, list):
        raise HTTPException(400, "body must be {\"zones\": [...]}")
    cleaned = []
    from .zones import clean_zone_dict
    for d in zones:
        try:
            cleaned.append(clean_zone_dict(d, config.DEFAULT_WARN_PCT))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"invalid zone: {exc}")
    registry.set_zones(cam_id, cleaned)
    return {"ok": True, "zones": cleaned}


@app.put("/api/cameras/{cam_id}/count-lines")
async def put_count_lines(cam_id: str, request: Request):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    try:
        body = await request.json()
        lines = body.get("lines")
        if not isinstance(lines, list) or len(lines) > 20:
            raise ValueError("lines must be a list of at most 20 virtual lines")
        cleaned = []
        for i, line in enumerate(lines):
            if not isinstance(line, dict): raise ValueError("each line must be an object")
            pts = [line.get("a"), line.get("b")]
            if any(not isinstance(p, list) or len(p) != 2 or any(float(v) < 0 or float(v) > 1 for v in p) for p in pts):
                raise ValueError("line endpoints must be normalized [x,y] pairs")
            cleaned.append({"id": str(line.get("id") or f"line-{i+1}"),
                            "name": str(line.get("name") or f"Line {i+1}")[:60],
                            "a": [float(v) for v in pts[0]], "b": [float(v) for v in pts[1]],
                            "entry_side": "negative" if line.get("entry_side") == "negative" else "positive"})
        registry.set_count_lines(cam_id, cleaned)
        return {"ok": True, "lines": cleaned}
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(400, f"invalid count lines: {exc}")


@app.put("/api/cameras/{cam_id}/display")
async def put_camera_display(cam_id: str, request: Request):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    try:
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("detections"), bool):
            raise ValueError("detections must be a boolean")
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(400, f"invalid display settings: {exc}")
    registry.set_display(cam_id, {"detections": body["detections"]})
    return {"ok": True, "display": {"detections": body["detections"]}}


# --------------------------------------------------------------- samples
@app.get("/api/samples")
async def list_samples():
    return ingest.list_samples()


# --------------------------------------------------------------- alerts
@app.get("/api/alerts")
async def get_alerts(limit: int = 100):
    limit = max(1, min(limit, 500))
    return store_recent_alerts(limit)


@app.get("/api/alerts/{alert_id}/evidence")
async def alert_evidence(alert_id: str):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alert_id):
        raise HTTPException(404, "evidence not found")
    from pathlib import Path
    evidence = Path(config.EVIDENCE_DIR) / f"{alert_id}.jpg"
    if not evidence.is_file():
        raise HTTPException(404, "evidence not found")
    return FileResponse(evidence, media_type="image/jpeg",
                        filename=f"alert-{alert_id}.jpg")


def store_recent_alerts(limit: int):
    from . import store
    return store.recent_alerts(limit)


# --------------------------------------------------------------- video
@app.get("/api/cameras/{cam_id}/stream")
async def cam_stream(cam_id: str, raw: int = 0):
    """MJPEG stream. `raw=1` serves the same frames without detection boxes,
    so a split view can show the annotated and the clean picture together."""
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")

    def frames():
        period = 1.0 / max(1, config.STREAM_FPS)
        while registry.get(cam_id) is not None:
            worker = registry.workers.get(cam_id)
            jpeg = None
            if worker is not None:
                jpeg = worker.raw_jpeg() if raw else worker.latest_jpeg
            if jpeg is None:
                time.sleep(0.2)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(period)

    return StreamingResponse(frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cameras/{cam_id}/snapshot")
async def cam_snapshot(cam_id: str):
    worker = registry.workers.get(cam_id)
    jpeg = worker.latest_jpeg if worker else None
    if jpeg is None:
        raise HTTPException(503, "no frame yet")
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")


# --------------------------------------------------------------- system
@app.get("/api/system")
async def system_endpoint():
    return system_info()


@app.get("/api/models")
async def list_models():
    """Model registry: backend, version, licence and warm-up state per model."""
    return {
        "active": model.name,
        "available": list(dict.fromkeys(model.requested)),
        "device": model.info()["device"],
        "models": model.models(),
        "density": registry.density.info(),
        "tiling": model.tiling_policy.__dict__,
    }


@app.get("/api/models/{name}/status")
async def model_status(name: str):
    if name not in dict.fromkeys(model.requested):
        raise HTTPException(404, "model not registered")
    return {**model.status(name), "spec": next(
        (m for m in model.models() if m["name"] == name), {})}


@app.put("/api/models")
async def switch_model(request: Request):
    try:
        body = await request.json()
        name = str(body.get("name", "")).strip()
    except Exception:
        raise HTTPException(400, "body must be JSON: {\"name\": \"model.pt\"}")
    if not name or name not in model.requested:
        raise HTTPException(400, "model must be one of the configured available models")
    try:
        return await asyncio.to_thread(model.switch, name)
    except Exception as exc:
        raise HTTPException(503, f"model switch failed: {exc}")


@app.put("/api/detector-profile")
async def set_detector_profile(request: Request):
    try:
        body = await request.json()
        profile = str(body.get("profile", "")).strip().lower()
    except Exception:
        raise HTTPException(400, "body must be JSON: {\"profile\": \"balanced|dense\"}")
    try:
        return await asyncio.to_thread(model.set_profile, profile)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/health")
async def health_endpoint():
    cameras = registry.snapshot()
    unhealthy = [
        c["id"] for c in cameras
        if c.get("metrics", {}).get("health", {}).get("status")
        in {"stale", "frozen", "dark", "low_contrast", "blurry"}
    ]
    analytics = registry.analytics_snapshot()
    analytics_degraded = [cid for cid, row in analytics.items()
                          if row["status"] == "degraded"]
    return {
        "status": "degraded" if (unhealthy or analytics_degraded) else "healthy",
        "cameras": len(cameras),
        "unhealthy_cameras": unhealthy,
        "analytics_degraded_cameras": analytics_degraded,
        "model_loaded": bool(model.model is not None),
        "scheduler": registry.scheduler.stats(),
        "outbox": registry.outbox.stats(),
    }


@app.get("/metrics")
async def prometheus_metrics():
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")


@app.put("/api/cameras/{cam_id}/calibration")
async def put_calibration(cam_id: str, request: Request):
    cam = registry.get(cam_id)
    if cam is None:
        raise HTTPException(404, "camera not found")
    try:
        data = await request.json()
        calibration = validate_calibration(data)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"invalid calibration: {exc}")
    # Calibration is versioned and attributed, so an alert can always be traced
    # back to the geometry that produced it (improvement.md §5).
    previous = cam.get("calibration") or {}
    calibration["version"] = int(previous.get("version", 0)) + 1
    calibration["validated_by"] = str((data or {}).get("validated_by") or "operator")[:60]
    calibration["validated_at"] = time.time()
    cam["calibration"] = calibration
    worker = registry.workers.get(cam_id)
    if worker:
        worker.calibration = calibration
    registry.save()
    from . import store
    store.add_audit("calibration.update", calibration["validated_by"], cam_id,
                    {"version": calibration["version"],
                     "inliers": calibration.get("inliers")})
    return {"ok": True, "calibration": calibration}


@app.post("/api/alerts/{alert_id}/ack")
async def ack_alert(alert_id: str, request: Request):
    """Operator acknowledgement. Stops escalation, keeps the audit record."""
    by = "operator"
    try:
        body = await request.json()
        by = str(body.get("by") or by)[:60]
    except Exception:
        pass
    row = registry.outbox.ack(alert_id, by)
    from . import store
    store.add_audit("alert.ack", by, alert_id, row or {"outbox": "absent"})
    if row is None:
        return {"ok": True, "id": alert_id, "acked_by": by, "outbox": "not_queued"}
    return {"ok": True, **row}


@app.post("/api/alerts/{alert_id}/close")
async def close_alert(alert_id: str, request: Request):
    """Close an alert and retain who closed it for the audit trail."""
    by = "operator"
    try:
        body = await request.json()
        by = str(body.get("by") or by)[:60]
    except Exception:
        pass
    from . import store
    if not store.close_alert(alert_id, by):
        if not any(alert["id"] == alert_id for alert in store.recent_alerts(500)):
            raise HTTPException(404, "alert not found")
        return {"ok": True, "id": alert_id, "status": "closed"}
    store.add_audit("alert.close", by, alert_id, {})
    return {"ok": True, "id": alert_id, "status": "closed", "closed_by": by}


@app.post("/api/alerts/{alert_id}/escalate")
async def escalate_alert(alert_id: str):
    ok = registry.outbox.escalate(alert_id)
    from . import store
    store.add_audit("alert.escalate", "operator", alert_id, {})
    return {"ok": ok, "id": alert_id}


@app.get("/api/outbox")
async def outbox_status():
    return registry.outbox.stats()


@app.get("/api/audit")
async def audit_log(limit: int = 100):
    from . import store
    return store.recent_audit(limit)


@app.get("/api/analytics/health")
async def analytics_health():
    """Model-side health: no-inference, count collapse and drift per camera."""
    data = registry.analytics_snapshot()
    degraded = [cid for cid, row in data.items() if row["status"] == "degraded"]
    return {"status": "degraded" if degraded else "healthy",
            "degraded_cameras": degraded, "cameras": data}


@app.get("/api/alerts/export")
async def export_alerts(limit: int = 500):
    rows = store_recent_alerts(limit)
    lines = ["id,ts,cam_name,zone,level,message"]
    for r in rows:
        lines.append(",".join([
            r["id"],
            str(r["ts"]),
            '"' + (r["cam_name"] or "").replace('"', '""') + '"',
            '"' + str(r["zone"]).replace('"', '""') + '"',
            r["level"],
            '"' + str(r["message"]).replace('"', '""') + '"',
        ]))
    return Response("\n".join(lines), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=alerts.csv"})


@app.get("/api/analytics")
async def analytics(cam_id: str | None = None, hours: int = 24):
    """Event analytics from 15-second samples. Forecasts are linear trend estimates."""
    from . import store
    now = time.time()
    hours = max(1, min(hours, 24 * 90))
    rows = store.history(now - hours * 3600, cam_id)
    prior_rows = store.history(now - 2 * hours * 3600, cam_id)
    prior_rows = [r for r in prior_rows if r["ts"] < now - hours * 3600]
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["cam_id"], []).append(row)
    forecasts = {}
    growth_alerts = []
    for key, series in grouped.items():
        points = [(r["ts"], r["count"]) for r in series[-240:]]
        slope = 0.0
        if len(points) >= 2:
            x0 = points[0][0]
            xs = [(p[0] - x0) / 60 for p in points]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(p[1] for p in points) / len(points)
            denom = sum((x - mean_x) ** 2 for x in xs)
            slope = sum((x - mean_x) * (p[1] - mean_y) for x, p in zip(xs, points)) / denom if denom else 0
        current = points[-1][1] if points else 0
        forecasts[key] = {"count": current, "growth_per_min": round(slope, 2),
                          "forecast": {str(m): max(0, round(current + slope * m)) for m in (5, 10, 15, 30)},
                          "confidence": "low" if len(points) < 8 else "trend only"}
        recent_points = [(r["ts"], r["count"]) for r in series if r["ts"] >= now - 300]
        growth_rate = 0.0
        if len(recent_points) >= 4:
            rx = [p[0] / 60 for p in recent_points]
            mx = sum(rx) / len(rx)
            my = sum(p[1] for p in recent_points) / len(recent_points)
            denom = sum((x - mx) ** 2 for x in rx)
            growth_rate = sum((x - mx) * (p[1] - my) for x, p in zip(rx, recent_points)) / denom if denom else 0.0
        if growth_rate >= 5:
            camera = registry.get(key)
            growth_alerts.append({"cam_id": key, "cam_name": camera.get("name", key) if camera else key,
                                  "growth_per_min": round(growth_rate, 2), "count": current,
                                  "level": "WARNING", "basis": "5-minute sampled trend; confirm on live video"})
    # Zone pressure/flow indicators are explicitly heuristics for operator review.
    latest = {}
    queues = []
    transitions = []
    for key, series in grouped.items():
        latest[key] = series[-1]["zones"] if series else []
        if len(series) > 1:
            def transition_map(row):
                return {(item["from"], item["to"]): item["count"]
                        for item in row.get("transitions", [])}
            first_t, last_t = transition_map(series[0]), transition_map(series[-1])
            names = {z["id"]: z["name"] for z in latest[key]}
            transitions.extend({"cam_id": key, "from_id": a, "to_id": b,
                                "from": names.get(a, a), "to": names.get(b, b),
                                "count": max(0, last_t.get((a, b), 0) - first_t.get((a, b), 0))}
                               for a, b in set(first_t) | set(last_t)
                               if last_t.get((a, b), 0) - first_t.get((a, b), 0) > 0)
        queue_zones = [z for z in latest[key] if "queue" in z.get("name", "").lower()]
        if queue_zones and series:
            recent = [r for r in series if r["ts"] >= now - 300]
            def exit_total(row):
                return sum(line.get("exit", 0) for line in row.get("lines", [])
                           if "exit" in line.get("name", "").lower())
            served = max(0, exit_total(recent[-1]) - exit_total(recent[0])) if len(recent) > 1 else 0
            elapsed_min = (recent[-1]["ts"] - recent[0]["ts"]) / 60 if len(recent) > 1 else 0
            rate = served / elapsed_min if elapsed_min > 0 else 0
            for zone in queue_zones:
                queues.append({"cam_id": key, "zone": zone["name"], "length": zone["count"],
                               "wait_minutes": round(zone["count"] / rate, 1) if rate > 0 else None,
                               "basis": "queue-zone count / recent exit-line throughput" if rate > 0 else "waiting for exit-line history"})
    alerts = store.recent_alerts(500)
    comparison = (f"Previous {hours}h: average {sum(r['count'] for r in prior_rows) / len(prior_rows):.1f} people across {len(prior_rows)} samples"
                  if prior_rows else f"Previous {hours}h comparison needs more recorded history.")
    camera_status = registry.snapshot()
    performance = {"cameras": len(camera_status),
                   "running": sum(c["status"] == "running" for c in camera_status),
                   "degraded": sum(c["status"] == "degraded" for c in camera_status),
                   "mean_infer_fps": round(sum((c.get("metrics") or {}).get("infer_fps", 0) for c in camera_status) / max(1, len(camera_status)), 2),
                   "sample_coverage": len(rows)}
    return {"window_hours": hours, "samples": len(rows), "forecasts": forecasts, "growth_alerts": growth_alerts, "queues": queues,
            "zones": latest, "transitions": transitions, "performance": performance,
            "bottlenecks": [{"cam_id": cid, **z} for cid, zs in latest.items() for z in zs
                            if z.get("occupancy_pct", 0) >= 75 and z.get("moving", 0) < 3],
            "alerts": [a for a in alerts if a["ts"] >= now - hours * 3600],
            "comparison": comparison}


@app.get("/api/heatmaps")
async def heatmap_history(cam_id: str, hours: int = 24, limit: int = 500):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    from . import store
    hours = max(1, min(hours, 24 * 90))
    return store.heatmap_history(time.time() - hours * 3600, cam_id, limit)


@app.get("/api/reports/daily.pdf")
@app.get("/api/reports/event.pdf")
async def daily_pdf(request: Request, hours: int = 24):
    from . import store
    from datetime import datetime, timezone
    is_daily = request.url.path.endswith("/daily.pdf")
    hours = 24 if is_daily else max(1, min(hours, 24 * 7))
    rows = store.history(time.time() - hours * 3600)
    alerts = store.recent_alerts(500)
    alerts = [a for a in alerts if a["ts"] >= time.time() - hours * 3600]
    report_kind = "Daily" if is_daily else "Event"
    lines = [f"Godavari Crowd Monitor - {report_kind} Operations Report ({hours}h)",
             datetime.now(timezone.utc).strftime("UTC %Y-%m-%d"),
             f"Samples captured: {len(rows)}", f"Alerts: {len(alerts)}", "",
             "Camera summary (latest sampled count):"]
    latest = {}
    for row in rows:
        latest[row["cam_id"]] = row
    lines.extend([f"{cid}: {r['count']} people" for cid, r in latest.items()] or ["No analytics samples recorded."])
    lines.append("Entry/exit lines (latest tracked totals):")
    lines.extend([f"{cid} · {line.get('name', 'Line')}: {line.get('entry', 0)} in / {line.get('exit', 0)} out"
                  for cid, r in latest.items() for line in r.get("lines", [])] or ["No virtual count lines configured."])
    lines.append("Zone transitions during report window:")
    first_by_cam = {}
    for row in rows:
        first_by_cam.setdefault(row["cam_id"], row)
    move_totals = {}
    for cid, last in latest.items():
        first = first_by_cam[cid]
        first_counts = {(t["from"], t["to"]): t["count"] for t in first.get("transitions", [])}
        for t in last.get("transitions", []):
            delta = max(0, t["count"] - first_counts.get((t["from"], t["to"]), 0))
            if delta:
                move_totals[(cid, t["from"], t["to"])] = delta
    lines.extend([f"{cid}: {a} -> {b}: {n}" for (cid, a, b), n in move_totals.items()] or ["No zone transitions recorded."])
    lines.extend(["", "Recent alerts:"])
    lines.extend([f"{a['level']} | {a['cam_name']} | {a['zone']} | {a['message']}" for a in alerts[-20:]] or ["No alerts in this period."])
    # Small dependency-free PDF writer. Split long reports over pages instead
    # of silently dropping everything after the first 52 lines.
    esc_pdf = lambda s: str(s).encode("latin-1", "replace").decode("latin-1").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    page_lines = [lines[i:i + 52] for i in range(0, len(lines), 52)] or [[]]
    page_ids = [4 + i * 2 for i in range(len(page_lines))]
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    for page_id, chunk in zip(page_ids, page_lines):
        content_id = page_id + 1
        content = "BT /F1 10 Tf 48 790 Td 13 TL " + " ".join(
            f"({esc_pdf(line[:110])}) Tj T*" for line in chunk) + " ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        objects.append(
            f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}\nendstream")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf)); pdf.extend(f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1"))
    xref = len(pdf); pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]: pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return Response(bytes(pdf), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=godavari-{report_kind.lower()}-{hours}h-report.pdf"})
