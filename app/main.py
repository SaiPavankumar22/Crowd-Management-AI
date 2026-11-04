"""FastAPI app: REST API, UploadFile/stream upload, MJPEG streams, WebSocket push."""
import asyncio
import os
import time
import urllib.parse

from fastapi import (FastAPI, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, ingest
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


@app.get("/api/cameras/{cam_id}")
async def get_camera(cam_id: str):
    for cam in registry.snapshot():
        if cam["id"] == cam_id:
            return cam
    raise HTTPException(404, "camera not found")


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


# --------------------------------------------------------------- samples
@app.get("/api/samples")
async def list_samples():
    return ingest.list_samples()


# --------------------------------------------------------------- alerts
@app.get("/api/alerts")
async def get_alerts(limit: int = 100):
    limit = max(1, min(limit, 500))
    return store_recent_alerts(limit)


def store_recent_alerts(limit: int):
    from . import store
    return store.recent_alerts(limit)


# --------------------------------------------------------------- video
@app.get("/api/cameras/{cam_id}/stream")
async def cam_stream(cam_id: str):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")

    def frames():
        period = 1.0 / max(1, config.STREAM_FPS)
        while registry.get(cam_id) is not None:
            worker = registry.workers.get(cam_id)
            jpeg = worker.latest_jpeg if worker else None
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
    from fastapi.responses import Response
    return Response("\n".join(lines), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=alerts.csv"})
