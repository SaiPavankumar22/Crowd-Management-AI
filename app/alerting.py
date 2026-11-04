"""Alert engine.

Design choices that kill false alarms in practice:
- WARNING fires once on a rising edge over warn_pct and re-arms below it
  (hysteresis) instead of spamming every frame.
- OVERCROWDED (>= 100% capacity) must persist for `sustain_s` seconds before
  an alert fires; while a zone stays over capacity it re-alerts at most once
  per `cooldown_s`.
"""
import time
import uuid


class AlertEngine:
    def __init__(self, cam_id: str, cam_name: str, sustain_s: float, cooldown_s: float):
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.sustain_s = max(0.0, sustain_s)
        self.cooldown_s = max(1.0, cooldown_s)
        self._state: dict[str, dict] = {}

    def reset(self):
        self._state.clear()

    def update(self, zone, ts: float | None = None) -> list[dict]:
        now = ts if ts is not None else time.time()
        s = self._state.setdefault(zone.id, {
            "armed_warn": True, "over_since": None,
            "last_alert": 0.0, "last_warn": 0.0,
        })
        fired: list[dict] = []
        occ = zone.occupancy_pct

        # --- WARNING: rising edge over warn_pct, but re-warn at most once
        # per cooldown, so a flickering count cannot spam the feed ---
        if occ >= zone.warn_pct:
            if s["armed_warn"] and now - s["last_warn"] >= self.cooldown_s:
                fired.append(self._alert(zone, "WARNING",
                    f"{zone.name} at {occ:.0f}% of capacity (warning level {zone.warn_pct:.0f}%)"))
                s["armed_warn"] = False
                s["last_warn"] = now
        elif occ < zone.warn_pct * 0.8:
            s["armed_warn"] = True

        # --- OVERCROWDED: capacity breach measured on the raw (unsmoothed)
        # count so a single noisy frame can neither start nor break the
        # sustained-over-capacity timer; re-alerts gated by cooldown ---
        over_raw = zone.raw_count >= zone.capacity
        if over_raw:
            if s["over_since"] is None:
                s["over_since"] = now
            if (now - s["over_since"] >= self.sustain_s
                    and now - s["last_alert"] >= self.cooldown_s):
                fired.append(self._alert(zone, "OVERCROWDED",
                    f"{zone.name} OVER CAPACITY: {zone.raw_count} people vs capacity {zone.capacity} "
                    f"({zone.occupancy_pct:.0f}%) sustained for {self.sustain_s:.0f}s"))
                s["last_alert"] = now
        else:
            s["over_since"] = None
            if zone.raw_count < zone.capacity * 0.9:
                s["last_alert"] = 0.0

        return fired

    def _alert(self, zone, level: str, message: str) -> dict:
        return {
            "id": uuid.uuid4().hex[:10],
            "cam_id": self.cam_id,
            "cam_name": self.cam_name,
            "zone": zone.name,
            "level": level,
            "message": message,
            "ts": time.time(),
        }
