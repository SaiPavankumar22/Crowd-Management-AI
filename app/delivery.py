"""Non-blocking alert delivery sinks."""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.request

from . import config, metrics


class AlertDispatcher:
    def __init__(self):
        self.enabled = bool(config.WEBHOOK_URL)

    def dispatch(self, alert: dict):
        if not self.enabled:
            return
        threading.Thread(
            target=self._send_with_retries,
            args=(dict(alert),),
            name="alert-webhook",
            daemon=True,
        ).start()

    def _send_with_retries(self, alert: dict):
        body = json.dumps({"event": "crowd_alert", "alert": alert}).encode()
        signature = ""
        if config.WEBHOOK_SECRET:
            signature = hmac.new(
                config.WEBHOOK_SECRET.encode(), body, hashlib.sha256
            ).hexdigest()
        for attempt in range(max(1, config.WEBHOOK_RETRIES)):
            try:
                request = urllib.request.Request(
                    config.WEBHOOK_URL,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "godavari-alerts/1.0",
                        **({"X-Godavari-Signature": signature} if signature else {}),
                    },
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=config.WEBHOOK_TIMEOUT_S
                ) as response:
                    if 200 <= response.status < 300:
                        metrics.inc("godavari_alert_deliveries_total")
                        return
                    raise RuntimeError(f"webhook returned HTTP {response.status}")
            except Exception:  # noqa: BLE001 - delivery must not stop cameras
                if attempt + 1 < config.WEBHOOK_RETRIES:
                    time.sleep(min(30.0, 2**attempt))
        metrics.inc("godavari_alert_delivery_failures_total")
