"""Shared runtime objects."""
from __future__ import annotations

import asyncio
import time

from . import config, ingest
from .detector import ModelManager
from .mongo_sync import mirror as mongo_mirror
from .pipeline import CameraRegistry

model = ModelManager(config.MODEL_NAME, config.FALLBACK_MODELS,
                     config.CONF_THRESHOLD, config.IMGSZ)
registry = CameraRegistry(model)
START_TS = time.time()
ws_clients: set = set()

async def broadcaster():
    """Push a full metrics snapshot to every dashboard ~twice a second."""
    while True:
        await asyncio.sleep(config.WS_PERIOD_S)
        if not ws_clients:
            continue
        payload = {
            "type": "snapshot",
            "ts": time.time(),
            "system": system_info(),
            "cameras": registry.snapshot(),
            "alerts": registry.recent_alerts(60),
        }
        dead = []
        for ws in list(ws_clients):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 - client went away
                dead.append(ws)
        for ws in dead:
            ws_clients.discard(ws)

def system_info() -> dict:
    info = model.info()
    snap = registry.snapshot()
    info["cameras_total"] = len(snap)
    info["cameras_running"] = sum(1 for c in snap if c["status"] == "running")
    info["samples"] = len(ingest.list_samples())
    info["uptime_s"] = int(time.time() - START_TS)
    info["ws_clients"] = len(ws_clients)
    info["scheduler"] = registry.scheduler.stats()
    info["outbox"] = registry.outbox.stats()
    info["density"] = registry.density.info()
    info["head_points"] = registry.points.info()
    info["mongodb"] = mongo_mirror.status()
    return info
