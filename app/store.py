"""SQLite alert history so operators can audit alerts after the fact."""
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
