"""Best-effort MongoDB Atlas archive; SQLite and local media stay canonical.

Writes are queued off the camera and request threads. Mongo outages never stop
camera processing or lose the local SQLite copy. Camera documents deliberately
omit source paths because RTSP URLs can contain credentials and uploaded video
files remain on the local volume.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from . import config
from .privacy import redact_source_credentials
from . import store

log = logging.getLogger(__name__)


class MongoMirror:
    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "disabled" if not config.MONGODB_URL else "connecting"
        self._error: str | None = None
        self._last_success: float | None = None
        self._dropped = 0

    def start(self) -> None:
        if not config.MONGODB_URL:
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="mongo-mirror", daemon=True)
            self._thread.start()
        # Persist a startup ping so status reflects an actual Atlas handshake
        # even when no application data has changed since the previous boot.
        store.mongo_enqueue("ping", "", {})
        self._wake.set()

    def _enqueue(self, op: str, collection: str, value: dict[str, Any] | str) -> None:
        if not config.MONGODB_URL:
            return
        self.start()
        store.mongo_enqueue(op, collection, redact_source_credentials(value))
        self._wake.set()

    def replace(self, collection: str, document: dict[str, Any]) -> None:
        if "_id" not in document:
            raise ValueError("Mongo mirror document requires _id")
        self._enqueue("replace", collection, document)

    def delete(self, collection: str, key: str) -> None:
        self._enqueue("delete", collection, key)

    def _run(self) -> None:
        client = None
        database = None
        try:
            from pymongo import MongoClient
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = type(exc).__name__
            return
        while True:
            rows = store.mongo_due(time.time(), limit=1)
            if not rows:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            row = rows[0]
            op, collection, value = row["op"], row["collection"], row["value"]
            try:
                if client is None:
                    client = MongoClient(
                        config.MONGODB_URL, serverSelectionTimeoutMS=3000,
                        connectTimeoutMS=3000, socketTimeoutMS=5000,
                        appname="GodavariCrowdMonitor")
                    database = client[config.MONGODB_DB_NAME]
                    client.admin.command("ping")
                if op == "ping":
                    pass
                elif op == "replace":
                    database[collection].replace_one({"_id": value["_id"]}, value, upsert=True)
                else:
                    database[collection].delete_one({"_id": value})
                store.mongo_complete(row["id"])
                with self._lock:
                    self._state, self._error = "connected", None
                    self._last_success = time.time()
            except Exception as exc:  # local SQLite remains authoritative and retryable.
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                client = database = None
                delay = min(3600, 2 ** min(row["attempts"], 12))
                store.mongo_retry(row["id"], row["attempts"] + 1,
                                  type(exc).__name__, time.time() + delay)
                with self._lock:
                    self._state, self._error = "error", type(exc).__name__
                log.warning("Mongo archive write failed (%s); durable SQLite retry queued",
                            type(exc).__name__)

    def status(self) -> dict[str, Any]:
        with self._lock:
            state, error, last_success = self._state, self._error, self._last_success
        pending = store.mongo_pending()
        with self._lock:
            return {
                "configured": bool(config.MONGODB_URL),
                "state": state,
                "database": config.MONGODB_DB_NAME if config.MONGODB_URL else None,
                "pending": pending,
                "dropped": 0,
                "last_success_ts": last_success,
                "last_error": error,
            }


mirror = MongoMirror()


def camera_document(camera: dict[str, Any]) -> dict[str, Any] | None:
    """Prepare camera settings without source credentials or local file paths."""
    camera_id = str(camera.get("id") or "")
    if not camera_id:
        return None
    doc = {key: value for key, value in camera.items()
           if key in {"name", "source_type", "zones", "zones_version", "count_lines",
                      "display", "profile", "created", "calibration", "calibration_version",
                      "warn_pct", "capacity"}}
    doc = redact_source_credentials(doc)
    doc.update({"_id": camera_id, "id": camera_id,
                "source_config_stored_locally": True})
    return doc


def mirror_camera(camera: dict[str, Any]) -> None:
    """Archive camera settings without source credentials or local file paths."""
    doc = camera_document(camera)
    if doc:
        mirror.replace("cameras", doc)
