"""Detector registry, camera health and Prometheus metrics."""
import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from .. import metrics
from ..state import model, registry, system_info

router = APIRouter(tags=["engine"])

@router.get("/api/system")
async def system_endpoint():
    return system_info()

@router.get("/api/models")
async def list_models():
    """Model registry: backend, version, licence and warm-up state per model."""
    return {
        "active": model.name,
        "available": list(dict.fromkeys(model.requested)),
        "device": model.info()["device"],
        "models": model.models(),
        "density": registry.density.info(),
        "head_points": registry.points.info(),
        "tiling": model.tiling_policy.__dict__,
    }

@router.get("/api/models/{name}/status")
async def model_status(name: str):
    if name not in dict.fromkeys(model.requested):
        raise HTTPException(404, "model not registered")
    return {**model.status(name), "spec": next(
        (m for m in model.models() if m["name"] == name), {})}

@router.put("/api/models")
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

@router.put("/api/detector-profile")
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

@router.get("/api/health")
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

@router.get("/metrics")
async def prometheus_metrics():
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")

@router.get("/api/analytics/health")
async def analytics_health():
    """Model-side health: no-inference, count collapse and drift per camera."""
    data = registry.analytics_snapshot()
    degraded = [cid for cid, row in data.items() if row["status"] == "degraded"]
    return {"status": "degraded" if degraded else "healthy",
            "degraded_cameras": degraded, "cameras": data}
