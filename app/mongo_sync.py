"""Best-effort MongoDB Atlas archive; SQLite and local media stay canonical.

Writes are queued off the camera and request threads. Mongo outages never stop
camera processing or lose the local SQLite copy. Camera documents deliberately
omit source paths because RTSP URLs can contain credentials and uploaded video
files remain on the local volume.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Any

from . import config

log = logging.getLogger(__name__)


class MongoMirror:
    def __init__(self):
        self._queue: queue.Queue[tuple[str, str, dict[str, Any] | str]] = queue.Queue(maxsize=5000)
        self._lock = threading.Lock()
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
        try:
            self._queue.put_nowait(("ping", "", {}))
        except queue.Full:
            pass

    def _enqueue(self, op: str, collection: str, value: dict[str, Any] | str) -> None:
        if not config.MONGODB_URL:
            return
        self.start()
        try:
            self._queue.put_nowait((op, collection, value))
        except queue.Full:
            with self._lock:
                self._dropped += 1
                self._error = "archive queue full; local SQLite copy retained"

    def replace(self, collection: str, document: dict[str, Any]) -> None:
        if "_id" not in document:
            raise ValueError("Mongo mirror document requires _id")
        self._enqueue("replace", collection, document)

    def delete(self, collection: str, key: str) -> None:
        self._enqueue("delete", collection, key)

    def _run(self) -> None:
        from pymongo import MongoClient

        client = None
        database = None
        while True:
            op, collection, value = self._queue.get()
            try:
                for attempt in range(3):
                    try:
                        if client is None:
                            client = MongoClient(
                                config.MONGODB_URL,
                                serverSelectionTimeoutMS=3000,
                                connectTimeoutMS=3000,
                                socketTimeoutMS=5000,
                                appname="GodavariCrowdMonitor",
                            )
                            database = client[config.MONGODB_DB_NAME]
                            client.admin.command("ping")
                        if op == "ping":
                            pass
                        elif op == "replace":
                            collection_obj = database[collection]
                            collection_obj.replace_one({"_id": value["_id"]}, value,
                                                       upsert=True)
                        else:
                            collection_obj = database[collection]
                            collection_obj.delete_one({"_id": value})
                        with self._lock:
                            self._state = "connected"
                            self._error = None
                            self._last_success = time.time()
                        break
                    except Exception as exc:  # Mongo is optional; SQLite remains authoritative.
                        if client is not None:
                            try:
                                client.close()
                            except Exception:
                                pass
                        client = database = None
                        with self._lock:
                            self._state = "error"
                            self._error = type(exc).__name__
                        if attempt < 2:
                            time.sleep(0.5 * (attempt + 1))
                else:
                    log.warning("Mongo archive write failed (%s); SQLite copy retained",
                                self._error or "unknown error")
            finally:
                self._queue.task_done()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "configured": bool(config.MONGODB_URL),
                "state": self._state,
                "database": config.MONGODB_DB_NAME if config.MONGODB_URL else None,
                "pending": self._queue.qsize(),
                "dropped": self._dropped,
                "last_success_ts": self._last_success,
                "last_error": self._error,
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
    if isinstance(doc.get("name"), str):
        doc["name"] = re.sub(r"(?i)(rtsps?://)[^/@]+@", r"\1[redacted]@", doc["name"])
    doc.update({"_id": camera_id, "id": camera_id,
                "source_config_stored_locally": True})
    return doc


def mirror_camera(camera: dict[str, Any]) -> None:
    """Archive camera settings without source credentials or local file paths."""
    doc = camera_document(camera)
    if doc:
        mirror.replace("cameras", doc)
