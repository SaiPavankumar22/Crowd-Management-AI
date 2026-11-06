"""Durable alert outbox."""
from __future__ import annotations

import hashlib
import hmac
import json
import random
import threading
import time
import urllib.request

from . import config, metrics, store

PENDING = "pending"
DELIVERED = "delivered"
FAILED = "failed"
ACKED = "acked"

class WebhookTransport:
    """HMAC-signed JSON POST to ALERT_WEBHOOK_URL."""

    def __init__(self, url: str | None = None, secret: str | None = None,
                 timeout: float | None = None):
        self.url = url or config.WEBHOOK_URL
        self.secret = secret or config.WEBHOOK_SECRET
        self.timeout = timeout or config.WEBHOOK_TIMEOUT_S

    @property
    def available(self) -> bool:
        return bool(self.url)

    def send(self, payload: dict) -> None:
        if not self.url:
            raise RuntimeError("no webhook configured")
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "User-Agent": "godavari-alerts/2.0"}
        if self.secret:
            headers["X-Godavari-Signature"] = hmac.new(
                self.secret.encode(), body, hashlib.sha256).hexdigest()
        request = urllib.request.Request(self.url, data=body, headers=headers,
                                         method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"webhook returned HTTP {response.status}")

class AlertOutbox:
    def __init__(self, transport=None, *, max_attempts: int | None = None,
                 poll_s: float = 1.0, backoff_base: float = 2.0,
                 backoff_cap: float = 120.0):
        self.transport = transport or WebhookTransport()
        self.max_attempts = max_attempts or config.WEBHOOK_RETRIES
        self.poll_s = max(0.1, poll_s)
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.transport, "available", True))

    # lifecycle ---------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="alert-outbox",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    # api ---------------------------------------------------------------
    def enqueue(self, alert: dict) -> str | None:
        """Persist an alert for delivery. Returns the idempotency key."""
        if not self.enabled:
            return None
        key = str(alert.get("id") or hashlib.sha1(
            f"{alert.get('cam_id')}|{alert.get('zone')}|{alert.get('ts')}".encode()
        ).hexdigest()[:16])
        payload = {
            "event": "crowd_alert",
            "schema_version": "1.1",
            "alert": {**alert, "id": key},
        }
        store.outbox_enqueue(key, payload, alert.get("ts") or time.time())
        metrics.inc("godavari_alert_outbox_enqueued_total")
        self._wake.set()
        return key

    def ack(self, alert_id: str, by: str = "operator") -> dict | None:
        row = store.outbox_ack(alert_id, by)
        store.ack_alert(alert_id, by)
        return row

    def escalate(self, alert_id: str) -> bool:
        return store.outbox_escalate(alert_id)

    def stats(self) -> dict:
        data = store.outbox_stats()
        data["enabled"] = self.enabled
        data["transport"] = type(self.transport).__name__
        data["max_attempts"] = self.max_attempts
        return data

    def flush(self, limit: int = 20, now: float | None = None) -> int:
        """Deliver due alerts synchronously (used by tests and shutdown)."""
        return self._deliver_due(now if now is not None else time.time(), limit)

    # internals ---------------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            try:
                delivered = self._deliver_due(time.time(), limit=20)
            except Exception:  # noqa: BLE001 - the outbox must never die
                delivered = 0
                metrics.inc("godavari_alert_outbox_errors_total")
            self._wake.wait(timeout=self.poll_s if not delivered else 0.05)
            self._wake.clear()

    def _deliver_due(self, now: float, limit: int) -> int:
        rows = store.outbox_due(now, limit)
        delivered = 0
        for row in rows:
            try:
                self.transport.send(row["payload"])
            except Exception as exc:  # noqa: BLE001 - retry or dead-letter
                attempts = row["attempts"] + 1
                if attempts >= self.max_attempts:
                    store.outbox_mark(row["id"], FAILED, attempts,
                                      str(exc)[:300], None)
                    metrics.inc("godavari_alert_delivery_failures_total")
                else:
                    delay = min(self.backoff_cap,
                                self.backoff_base ** attempts) + random.uniform(0, 0.5)
                    store.outbox_mark(row["id"], PENDING, attempts,
                                      str(exc)[:300], now + delay)
                    metrics.inc("godavari_alert_delivery_retries_total")
            else:
                store.outbox_mark(row["id"], DELIVERED, row["attempts"],
                                  None, None)
                metrics.inc("godavari_alert_deliveries_total")
                delivered += 1
        return delivered
