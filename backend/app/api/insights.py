"""Event analytics, heatmap history and printable operations reports."""
import time

from fastapi import APIRouter, HTTPException, Request, Response

from ..state import registry

router = APIRouter(tags=["insights"])

@router.get("/api/analytics")
async def analytics(cam_id: str | None = None, hours: int = 24):
    """Event analytics from 15-second samples. Forecasts are linear trend estimates."""
    from .. import store
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

@router.get("/api/heatmaps")
async def heatmap_history(cam_id: str, hours: int = 24, limit: int = 500):
    if registry.get(cam_id) is None:
        raise HTTPException(404, "camera not found")
    from .. import store
    hours = max(1, min(hours, 24 * 90))
    return store.heatmap_history(time.time() - hours * 3600, cam_id, limit)

@router.get("/api/reports/daily.pdf")
@router.get("/api/reports/event.pdf")
async def daily_pdf(request: Request, hours: int = 24):
    from .. import store
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
