"""Cameras: CRUD, zones, gate lines, calibration and dense-count validation."""
import asyncio
import math
import os
import time

from fastapi import APIRouter, HTTPException, Request, UploadFile

from .. import config, ingest
from ..calibration import validate_calibration
from ..privacy import source_display_name
from ..pipeline import new_camera_id
from ..state import model, registry

router = APIRouter(tags=["cameras"])

@router.post("/api/cameras")
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
        except ingest.UploadTooLarge as exc:
            raise HTTPException(413, str(exc))
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
            cfg = make_cfg(cam_name(source_display_name(src)), "rtsp", src, True)
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
           "path": path, "zones": [], "created": time.time(),
           "estimation_policy": {"selected_source": "none", "samples": [],
                                 "validation": {"approved": False}}}
    if source_type == "file":
        cfg["loop"] = loop
    return cfg

@router.get("/api/cameras")
async def list_cameras():
    return registry.snapshot()

@router.get("/api/cameras/{cam_id}/diagnostics")
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

@router.get("/api/cameras/{cam_id}")
async def get_camera(cam_id: str):
    for cam in registry.snapshot():
        if cam["id"] == cam_id:
            return cam
    raise HTTPException(404, "camera not found")

@router.put("/api/cameras/{cam_id}/profile")
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

@router.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    registry.remove(cam_id)
    return {"ok": True}

@router.put("/api/cameras/{cam_id}/zones")
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
    from ..zones import clean_zone_dict
    for d in zones:
        try:
            cleaned.append(clean_zone_dict(d, config.DEFAULT_WARN_PCT))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"invalid zone: {exc}")
    registry.set_zones(cam_id, cleaned)
    return {"ok": True, "zones": cleaned}

@router.put("/api/cameras/{cam_id}/count-lines")
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
            if any(not isinstance(p, list) or len(p) != 2 or any(not math.isfinite(float(v)) or float(v) < 0 or float(v) > 1 for v in p) for p in pts):
                raise ValueError("line endpoints must be normalized [x,y] pairs")
            if pts[0] == pts[1]:
                raise ValueError("line endpoints must differ")
            cleaned.append({"id": str(line.get("id") or f"line-{i+1}"),
                            "name": str(line.get("name") or f"Line {i+1}")[:60],
                            "a": [float(v) for v in pts[0]], "b": [float(v) for v in pts[1]],
                            "entry_side": "negative" if line.get("entry_side") == "negative" else "positive"})
        registry.set_count_lines(cam_id, cleaned)
        return {"ok": True, "lines": cleaned}
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(400, f"invalid count lines: {exc}")

@router.put("/api/cameras/{cam_id}/display")
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

@router.get("/api/cameras/{cam_id}/estimation-policy")
async def get_estimation_policy(cam_id: str):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    return registry.validation_summary(cam_id)

@router.post("/api/cameras/{cam_id}/validation-samples")
async def add_validation_sample(cam_id: str, request: Request):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("body must be an object")
        source = str(body.get("source") or "").strip().lower()
        manual_count = float(body["manual_count"])
        model_count = float(body["model_count"])
        if not math.isfinite(manual_count) or not math.isfinite(model_count):
            raise ValueError("counts must be finite")
        summary = registry.add_validation_sample(
            cam_id, source, manual_count, model_count,
            str(body.get("by") or "operator"), str(body.get("note") or ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"invalid validation sample: {exc}")
    return {"ok": True, **summary}

@router.put("/api/cameras/{cam_id}/estimation-policy")
async def put_estimation_policy(cam_id: str, request: Request):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("body must be an object")
        source = str(body.get("selected_source") or "none").strip().lower()
        by = str(body.get("by") or "operator")
        if source == "none":
            summary = registry.revoke_estimation_source(cam_id, by,
                                                         str(body.get("reason") or ""))
        else:
            summary = registry.promote_estimation_source(cam_id, source, by)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"invalid estimation policy: {exc}")
    return {"ok": True, **summary}

@router.put("/api/cameras/{cam_id}/calibration")
async def put_calibration(cam_id: str, request: Request):
    cam = registry.get(cam_id)
    if cam is None:
        raise HTTPException(404, "camera not found")
    try:
        data = await request.json()
        calibration = validate_calibration(data)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(400, f"invalid calibration: {exc}")
    # Calibration is versioned and attributed, so an alert can always be traced
    # back to the geometry that produced it (improvement.md §5).
    previous = cam.get("calibration") or {}
    calibration["version"] = int(previous.get("version", 0)) + 1
    calibration["validated_by"] = str((data or {}).get("validated_by") or "operator")[:60]
    calibration["validated_at"] = time.time()
    cam["calibration"] = calibration
    # A new homography changes zone area and people/m².  Any prior promotion
    # was validated against the old geometry and must be reviewed again.
    policy = cam.setdefault("estimation_policy", {})
    if policy.get("selected_source") not in (None, "none"):
        policy["selected_source"] = "none"
        policy["validation"] = {"approved": False, "revoked_by": calibration["validated_by"],
                                "revoked_at": calibration["validated_at"],
                                "reason": "calibration changed"}
    worker = registry.workers.get(cam_id)
    if worker:
        worker.calibration = calibration
    registry.save()
    from .. import store
    store.add_audit("calibration.update", calibration["validated_by"], cam_id,
                    {"version": calibration["version"],
                     "inliers": calibration.get("inliers")})
    return {"ok": True, "calibration": calibration}


@router.get("/api/samples")
async def list_samples():
    return ingest.list_samples()
