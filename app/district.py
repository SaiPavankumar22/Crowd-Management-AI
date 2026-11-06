"""Edge -> district aggregation (plan §8).

Each camera node keeps running the pipeline locally. When
`CROWD_DISTRICT_PUSH_URL` is set it posts a signed summary of its cameras and
alerts on a fixed period; a district node accepts those summaries on
`/api/district/ingest` and serves the merged picture on
`/api/district/overview`.

There is no central tier: one district node is the ceiling of what this
repository runs, and nothing here is required for single-node operation — with
no push URL the thread never starts and ingest answers 503 until a shared
secret exists.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.request

from fastapi import APIRouter, HTTPException, Request

from . import config, metrics
from .state import model, registry

router = APIRouter(tags=["district"])

_lock = threading.Lock()
_nodes: dict[str, dict] = {}


def _signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _authorised(body: bytes, signature: str) -> bool:
    if not config.DISTRICT_SECRET:
        return False
    return hmac.compare_digest(_signature(body, config.DISTRICT_SECRET), signature or "")


# ------------------------------------------------------------------ ingest
@router.post("/api/district/ingest")
async def ingest(request: Request):
    """Accept one node snapshot. Signed with the district's shared secret."""
    if not config.DISTRICT_SECRET:
        raise HTTPException(503, "district ingest is not configured")
    body = await request.body()
    if not _authorised(body, request.headers.get("x-godavari-signature", "")):
        raise HTTPException(401, "invalid district signature")
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(400, "body must be JSON")
    node = str(payload.get("node") or "").strip()[:60]
    if not node:
        raise HTTPException(400, "node id is required")
    with _lock:
        _nodes[node] = {**payload, "node": node, "received_ts": time.time()}
    metrics.inc("godavari_district_ingest_total")
    return {"ok": True, "node": node, "nodes": len(_nodes)}


@router.get("/api/district/overview")
async def overview():
    """Merged picture of every node that pushed recently."""
    now = time.time()
    ttl = config.DISTRICT_NODE_TTL_S
    with _lock:
        nodes = sorted(_nodes.values(), key=lambda row: row["node"])
    rows = []
    totals = {"cameras": 0, "people": 0, "overcrowded": 0, "alerts": 0}
    for node in nodes:
        cameras = node.get("cameras") or []
        people = sum(int(c.get("count") or 0) for c in cameras)
        over = sum(1 for c in cameras if str(c.get("level")) == "OVERCROWDED")
        alerts = len(node.get("alerts") or [])
        totals["cameras"] += len(cameras)
        totals["people"] += people
        totals["overcrowded"] += over
        totals["alerts"] += alerts
        rows.append({
            "node": node["node"],
            "age_s": round(now - node["received_ts"], 1),
            "stale": now - node["received_ts"] > ttl,
            "cameras": len(cameras),
            "people": people,
            "overcrowded": over,
            "alerts": alerts,
            "device": node.get("device"),
        })
    return {
        "district": config.DISTRICT_NODE_ID or "local",
        "nodes": rows,
        "totals": totals,
        "stale_nodes": [row["node"] for row in rows if row["stale"]],
        "node_ttl_s": ttl,
    }


# ----------------------------------------------------------------- publish
def summary() -> dict:
    """The payload this node publishes to its district."""
    cameras = []
    for cam in registry.snapshot():
        metrics_ = cam.get("metrics") or {}
        zones = cam.get("zones") or []
        cameras.append({
            "id": cam["id"],
            "name": cam.get("name"),
            "count": metrics_.get("count"),
            "level": max((z.get("metrics", {}).get("level") for z in zones),
                         key=lambda lv: {"OK": 0, "BUSY": 1, "WARNING": 2,
                                         "OVERCROWDED": 3}.get(lv, 0), default=None),
            "status": cam.get("status"),
        })
    alerts = [{"id": a.get("id"), "level": a.get("level"), "ts": a.get("ts")}
              for a in registry.recent_alerts(20)]
    return {"node": config.DISTRICT_NODE_ID, "ts": time.time(),
            "device": model.info().get("device"), "cameras": cameras,
            "alerts": alerts}


class DistrictPublisher(threading.Thread):
    """Push this node's summary to the district on a fixed period."""

    def __init__(self, url: str | None = None, secret: str | None = None,
                 period_s: float | None = None):
        super().__init__(name="district-push", daemon=True)
        self.url = url or config.DISTRICT_PUSH_URL
        self.secret = secret or config.DISTRICT_SECRET
        self.period_s = period_s or config.DISTRICT_PUSH_PERIOD_S
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.secret)

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.wait(self.period_s):
            try:
                self.push_once()
            except Exception:  # noqa: BLE001 - a district outage must not stop counting
                metrics.inc("godavari_district_push_failures_total")

    def push_once(self) -> None:
        body = json.dumps(summary()).encode()
        request = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "godavari-district/1.0",
                     "X-Godavari-Signature": _signature(body, self.secret)})
        with urllib.request.urlopen(request, timeout=config.WEBHOOK_TIMEOUT_S) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"district returned HTTP {response.status}")
        metrics.inc("godavari_district_push_total")


publisher = DistrictPublisher()
