"""Analytics health: a dead-man's switch for the model, not just the camera."""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field

DEGRADED = "degraded"
HEALTHY = "healthy"
UNKNOWN = "unknown"

@dataclass
class AnalyticsHealthConfig:
    # No successful inference for this long (while the worker claims running)
    # means the analytics path is dead even if frames still arrive.
    inference_timeout_s: float = 12.0
    # A previously busy zone dropping to zero for this long is suspicious.
    collapse_window_s: float = 45.0
    collapse_min_previous: int = 8
    # Compare the recent median against the same-length prior window.
    drift_window: int = 60
    drift_high_ratio: float = 3.0
    drift_low_ratio: float = 0.33

@dataclass
class AnalyticsHealth:
    camera_id: str
    config: AnalyticsHealthConfig = field(default_factory=AnalyticsHealthConfig)
    last_ok_ts: float = 0.0
    last_count: int = 0
    last_nonzero_ts: float = 0.0
    consecutive_empty: int = 0
    counts: deque = field(default_factory=lambda: deque(maxlen=240))
    status: str = UNKNOWN
    reasons: list[str] = field(default_factory=list)
    inference_errors: int = 0

    # ------------------------------------------------------------------ #
    def record_inference(self, count: int, ts: float | None = None,
                         error: str | None = None) -> None:
        ts = ts if ts is not None else time.time()
        if error:
            self.inference_errors += 1
            self.reasons = [f"inference_error: {error}"]
            self.status = DEGRADED
            return
        self.last_ok_ts = ts
        self.last_count = int(count)
        self.counts.append(int(count))
        if count > 0:
            self.last_nonzero_ts = ts
            self.consecutive_empty = 0
        else:
            self.consecutive_empty += 1

    def evaluate(self, now: float | None = None, worker_status: str = "running",
                 camera_health: str | None = None) -> dict:
        now = now if now is not None else time.time()
        reasons: list[str] = []
        cfg = self.config

        if worker_status not in ("running",):
            reasons.append(f"worker_{worker_status}")
        if self.last_ok_ts == 0.0:
            reasons.append("no_inference_yet")
        elif now - self.last_ok_ts > cfg.inference_timeout_s:
            reasons.append(f"no_inference_for_{now - self.last_ok_ts:.0f}s")

        if (self.consecutive_empty
                and self.last_nonzero_ts
                and now - self.last_nonzero_ts > cfg.collapse_window_s
                and self.counts
                and max(self.counts) >= cfg.collapse_min_previous
                and camera_health in (None, "healthy")):
            # The camera looks fine but the crowd "vanished": likely a model or
            # threshold failure, not an empty ghat. Demand operator review.
            reasons.append("count_collapse_suspected")

        drift = self._drift()
        if drift is not None and drift >= cfg.drift_high_ratio:
            reasons.append(f"count_drift_up_{drift:.1f}x")
        if drift is not None and 0 < drift <= cfg.drift_low_ratio:
            reasons.append(f"count_drift_down_{drift:.1f}x")

        if not self.last_ok_ts and worker_status == "starting":
            status = UNKNOWN
        elif reasons:
            status = DEGRADED
        else:
            status = HEALTHY
        self.status = status
        self.reasons = reasons
        return {
            "camera_id": self.camera_id,
            "status": status,
            "reasons": list(reasons),
            "last_inference_age_s": (round(now - self.last_ok_ts, 1)
                                     if self.last_ok_ts else None),
            "consecutive_empty": self.consecutive_empty,
            "drift_ratio": round(drift, 2) if drift is not None else None,
            "inference_errors": self.inference_errors,
        }

    def _drift(self) -> float | None:
        """Recent median vs the same-length window before it."""
        n = self.config.drift_window
        if len(self.counts) < 2 * n:
            return None
        recent = list(self.counts)[-n:]
        prior = list(self.counts)[-2 * n:-n]
        prior_median = statistics.median(prior)
        if prior_median <= 0:
            return None
        return statistics.median(recent) / prior_median
