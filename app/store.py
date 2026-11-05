"""SQLite alert history and coarse analytics for event review."""
import os
import sqlite3
import threading

from . import config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        _conn = sqlite3.connect(
            os.path.join(config.DATA_DIR, "crowd.db"), check_same_thread=False
        )
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY, cam_id TEXT, cam_name TEXT,
                zone TEXT, level TEXT, message TEXT, ts REAL)"""
        )
        _conn.execute("""CREATE TABLE IF NOT EXISTS samples (
            cam_id TEXT NOT NULL, ts REAL NOT NULL, count INTEGER NOT NULL,
            occupancy REAL, zones TEXT NOT NULL, lines TEXT NOT NULL DEFAULT '[]',
            transitions TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(cam_id, ts))""")
        columns = {row[1] for row in _conn.execute("PRAGMA table_info(samples)")}
        for name, definition in (("lines", "TEXT NOT NULL DEFAULT '[]'"),
                                 ("transitions", "TEXT NOT NULL DEFAULT '[]'")):
            if name not in columns:
                _conn.execute(f"ALTER TABLE samples ADD COLUMN {name} {definition}")
        _conn.execute("""CREATE TABLE IF NOT EXISTS heatmaps (
            cam_id TEXT NOT NULL, ts REAL NOT NULL, grid TEXT NOT NULL,
            PRIMARY KEY(cam_id, ts))""")
        # Durable alert outbox: an alert is persisted before it is delivered, so
        # a restart or a long outage cannot lose it (improvement.md §3).
        _conn.execute("""CREATE TABLE IF NOT EXISTS alert_outbox (
            id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0, created_ts REAL NOT NULL,
            next_attempt_ts REAL NOT NULL, last_error TEXT, delivered_ts REAL,
            acked_by TEXT, acked_ts REAL, escalated INTEGER NOT NULL DEFAULT 0)""")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_due "
                      "ON alert_outbox(state, next_attempt_ts)")
        _conn.execute("""CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, actor TEXT,
            action TEXT NOT NULL, target TEXT, detail TEXT)""")
        alert_columns = {row[1] for row in _conn.execute("PRAGMA table_info(alerts)")}
        for name, definition in (("zone_id", "TEXT"), ("ack_by", "TEXT"),
                                 ("ack_ts", "REAL"), ("context", "TEXT")):
            if name not in alert_columns:
                _conn.execute(f"ALTER TABLE alerts ADD COLUMN {name} {definition}")
        _conn.commit()
    return _conn


def add_alert(alert: dict):
    import json
    with _lock:
        conn = _db()
        conn.execute(
            "INSERT OR REPLACE INTO alerts (id,cam_id,cam_name,zone,level,message,ts,"
            "zone_id,context) VALUES (?,?,?,?,?,?,?,?,?)",
            (alert["id"], alert["cam_id"], alert["cam_name"], alert["zone"],
             alert["level"], alert["message"], alert["ts"],
             alert.get("zone_id"), json.dumps(alert.get("context") or {})),
        )
        conn.commit()


def recent_alerts(limit: int = 100) -> list[dict]:
    import json
    with _lock:
        conn = _db()
        rows = conn.execute(
            "SELECT id, cam_id, cam_name, zone, level, message, ts, zone_id, "
            "context, ack_by FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        try:
            context = json.loads(r[8] or "{}")
        except Exception:
            context = {}
        out.append({"id": r[0], "cam_id": r[1], "cam_name": r[2], "zone": r[3],
                    "level": r[4], "message": r[5], "ts": r[6],
                    "zone_id": r[7], "context": context, "ack_by": r[9]})
    return out


def ack_alert(alert_id: str, by: str) -> None:
    with _lock:
        conn = _db()
        conn.execute("UPDATE alerts SET ack_by=?, ack_ts=? WHERE id=?",
                     (by, __import__("time").time(), alert_id))
        conn.commit()


# --------------------------------------------------------------------- outbox
def outbox_enqueue(key: str, payload: dict, ts: float) -> bool:
    """Insert once per idempotency key; duplicates are ignored."""
    import json
    with _lock:
        conn = _db()
        cursor = conn.execute(
            "INSERT OR IGNORE INTO alert_outbox "
            "(id,payload,state,attempts,created_ts,next_attempt_ts) "
            "VALUES (?,?, 'pending', 0, ?, ?)",
            (key, json.dumps(payload), ts, ts),
        )
        conn.commit()
        return cursor.rowcount > 0


def outbox_due(now: float, limit: int = 20) -> list[dict]:
    import json
    with _lock:
        rows = _db().execute(
            "SELECT id, payload, attempts FROM alert_outbox "
            "WHERE state='pending' AND next_attempt_ts<=? "
            "ORDER BY next_attempt_ts LIMIT ?", (now, max(1, limit))
        ).fetchall()
    return [{"id": r[0], "payload": json.loads(r[1]), "attempts": r[2]} for r in rows]


def outbox_mark(key: str, state: str, attempts: int, error: str | None,
                next_attempt_ts: float | None) -> None:
    import time
    with _lock:
        conn = _db()
        conn.execute(
            "UPDATE alert_outbox SET state=?, attempts=?, last_error=?, "
            "next_attempt_ts=COALESCE(?, next_attempt_ts), delivered_ts=? WHERE id=?",
            (state, attempts, error, next_attempt_ts,
             time.time() if state == "delivered" else None, key),
        )
        conn.commit()


def outbox_ack(key: str, by: str) -> dict | None:
    import time
    with _lock:
        conn = _db()
        row = conn.execute("SELECT id, state, attempts FROM alert_outbox WHERE id=?",
                           (key,)).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE alert_outbox SET acked_by=?, acked_ts=?, "
                     "state=CASE WHEN state='failed' THEN 'acked' ELSE state END "
                     "WHERE id=?", (by, time.time(), key))
        conn.commit()
    return {"id": row[0], "state": row[1], "attempts": row[2], "acked_by": by}


def outbox_escalate(key: str) -> bool:
    with _lock:
        conn = _db()
        cursor = conn.execute("UPDATE alert_outbox SET escalated=1 WHERE id=?", (key,))
        conn.commit()
        return cursor.rowcount > 0


def outbox_stats() -> dict:
    with _lock:
        conn = _db()
        rows = conn.execute(
            "SELECT state, COUNT(*) FROM alert_outbox GROUP BY state").fetchall()
        oldest = conn.execute(
            "SELECT MIN(created_ts) FROM alert_outbox WHERE state='pending'"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM alert_outbox WHERE state='failed'").fetchone()[0]
        acked = conn.execute(
            "SELECT COUNT(*) FROM alert_outbox WHERE acked_ts IS NOT NULL").fetchone()[0]
    counts = {state: count for state, count in rows}
    return {"pending": counts.get("pending", 0),
            "delivered": counts.get("delivered", 0),
            "failed": failed, "acked": acked,
            "oldest_pending_ts": oldest}


# ---------------------------------------------------------------------- audit
def add_audit(action: str, actor: str = "system", target: str | None = None,
              detail: dict | None = None) -> None:
    import json
    import time
    with _lock:
        conn = _db()
        conn.execute("INSERT INTO audit_log (ts,actor,action,target,detail) "
                     "VALUES (?,?,?,?,?)",
                     (time.time(), actor, action, target,
                      json.dumps(detail or {})))
        conn.commit()


def recent_audit(limit: int = 100) -> list[dict]:
    import json
    with _lock:
        rows = _db().execute(
            "SELECT ts, actor, action, target, detail FROM audit_log "
            "ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
    return [{"ts": r[0], "actor": r[1], "action": r[2], "target": r[3],
             "detail": json.loads(r[4] or "{}")} for r in rows]


def add_sample(cam_id: str, ts: float, count: int, occupancy, zones: list,
               lines: list | None = None, transitions: list | None = None):
    import json
    with _lock:
        conn = _db()
        conn.execute("INSERT OR REPLACE INTO samples (cam_id,ts,count,occupancy,zones,lines,transitions) VALUES (?,?,?,?,?,?,?)",
                     (cam_id, ts, count, occupancy, json.dumps(zones), json.dumps(lines or []), json.dumps(transitions or [])))
        # Retain 90 days; samples are intentionally coarse (one per 15 seconds).
        conn.execute("DELETE FROM samples WHERE ts < ?", (ts - 90 * 86400,))
        conn.commit()


def history(since: float, cam_id: str | None = None) -> list[dict]:
    import json
    with _lock:
        conn = _db()
        if cam_id:
            rows = conn.execute("SELECT cam_id,ts,count,occupancy,zones,lines,transitions FROM samples WHERE ts>=? AND cam_id=? ORDER BY ts", (since, cam_id)).fetchall()
        else:
            rows = conn.execute("SELECT cam_id,ts,count,occupancy,zones,lines,transitions FROM samples WHERE ts>=? ORDER BY ts", (since,)).fetchall()
    return [{"cam_id": r[0], "ts": r[1], "count": r[2], "occupancy": r[3], "zones": json.loads(r[4]), "lines": json.loads(r[5]), "transitions": json.loads(r[6])} for r in rows]


def add_heatmap(cam_id: str, ts: float, grid: dict):
    import json
    with _lock:
        conn = _db()
        conn.execute("INSERT OR REPLACE INTO heatmaps VALUES (?,?,?)", (cam_id, ts, json.dumps(grid)))
        conn.execute("DELETE FROM heatmaps WHERE ts < ?", (ts - 90 * 86400,))
        conn.commit()


def heatmap_history(since: float, cam_id: str, limit: int = 500) -> list[dict]:
    import json
    with _lock:
        rows = _db().execute("SELECT ts,grid FROM heatmaps WHERE cam_id=? AND ts>=? ORDER BY ts DESC LIMIT ?",
                             (cam_id, since, max(1, min(limit, 2000)))).fetchall()
    return [{"ts": r[0], "heatmap": json.loads(r[1])} for r in reversed(rows)]
