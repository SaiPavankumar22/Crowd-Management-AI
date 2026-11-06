"""Alerts: stream, evidence, acknowledgement, escalation and the outbox."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .. import config, store
from ..state import registry

router = APIRouter(tags=["alerts"])

@router.get("/api/alerts")
async def get_alerts(limit: int = 100):
    limit = max(1, min(limit, 500))
    return store_recent_alerts(limit)

@router.get("/api/alerts/{alert_id}/evidence")
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
    return store.recent_alerts(limit)

@router.post("/api/alerts/{alert_id}/ack")
async def ack_alert(alert_id: str, request: Request):
    """Operator acknowledgement. Stops escalation, keeps the audit record."""
    by = "operator"
    try:
        body = await request.json()
        if isinstance(body, dict):
            by = str(body.get("by") or by)[:60]
    except Exception:
        pass
    row = registry.outbox.ack(alert_id, by)
    store.add_audit("alert.ack", by, alert_id, row or {"outbox": "absent"})
    if row is None:
        return {"ok": True, "id": alert_id, "acked_by": by, "outbox": "not_queued"}
    return {"ok": True, **row}

@router.post("/api/alerts/{alert_id}/close")
async def close_alert(alert_id: str, request: Request):
    """Close an alert and retain who closed it for the audit trail."""
    by = "operator"
    try:
        body = await request.json()
        if isinstance(body, dict):
            by = str(body.get("by") or by)[:60]
    except Exception:
        pass
    existing = store.get_alert(alert_id)
    if existing is None:
        raise HTTPException(404, "alert not found")
    if existing.get("status") == "closed":
        return {"ok": True, "id": alert_id, "status": "closed"}
    if not (existing.get("ack_by") or existing.get("ack_ts")):
        raise HTTPException(409, "alert must be acknowledged before it can be closed")
    if not store.close_alert(alert_id, by):
        return {"ok": True, "id": alert_id, "status": "closed"}
    store.add_audit("alert.close", by, alert_id, {})
    return {"ok": True, "id": alert_id, "status": "closed", "closed_by": by}

@router.post("/api/alerts/{alert_id}/escalate")
async def escalate_alert(alert_id: str):
    ok = registry.outbox.escalate(alert_id)
    store.add_audit("alert.escalate", "operator", alert_id, {})
    return {"ok": ok, "id": alert_id}

@router.get("/api/outbox")
async def outbox_status():
    return registry.outbox.stats()

@router.get("/api/audit")
async def audit_log(limit: int = 100):
    return store.recent_audit(limit)

@router.get("/api/alerts/export")
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
