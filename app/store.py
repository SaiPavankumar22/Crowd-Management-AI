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
        _conn.commit()
    return _conn


def add_alert(alert: dict):
    with _lock:
        conn = _db()
        conn.execute(
            "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?)",
            (alert["id"], alert["cam_id"], alert["cam_name"], alert["zone"],
             alert["level"], alert["message"], alert["ts"]),
        )
        conn.commit()


def recent_alerts(limit: int = 100) -> list[dict]:
    with _lock:
        conn = _db()
        rows = conn.execute(
            "SELECT id, cam_id, cam_name, zone, level, message, ts "
            "FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {"id": r[0], "cam_id": r[1], "cam_name": r[2], "zone": r[3],
         "level": r[4], "message": r[5], "ts": r[6]}
        for r in rows
    ]


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
