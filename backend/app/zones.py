"""Zones: a named polygon region of a camera with a capacity."""
import uuid
import math

LEVEL_COLORS = {
    "OK": "#22c55e",
    "BUSY": "#eab308",
    "WARNING": "#f97316",
    "OVERCROWDED": "#ef4444",
}
SEVERITY = {"OK": 0, "BUSY": 1, "WARNING": 2, "OVERCROWDED": 3}

# Crowd levels on a zone's occupancy % of configured capacity:
#   < ~2/3 warn  -> OK | BUSY,  >= warn -> WARNING,  >= 100 -> OVERCROWDED
def level_for(occupancy_pct: float, warn_pct: float = 75.0) -> str:
    if occupancy_pct >= 100.0:
        return "OVERCROWDED"
    if occupancy_pct >= warn_pct:
        return "WARNING"
    if occupancy_pct >= (warn_pct * 2.0 / 3.0):
        return "BUSY"
    return "OK"

class Zone:
    def __init__(self, name: str, polygon, capacity: int = 100,
                 warn_pct: float = 75.0, zone_id: str | None = None):
        self.id = zone_id or uuid.uuid4().hex[:8]
        self.name = name
        self.polygon = [[float(x), float(y)] for x, y in polygon]
        self.capacity = max(1, int(capacity))
        self.warn_pct = float(warn_pct)
        # runtime metrics
        self.count = 0.0          # EMA-smoothed person count inside the zone
        self.raw_count = 0
        self.occupancy_pct = 0.0
        self.density_per_10k = 0.0  # persons per 10,000 px^2 of zone area
        self.density_m2 = None
        self.level = "OK"
        self._first = True

    def apply_count(self, raw_count: int, area_px: float, area_m2: float | None = None):
        if self._first:
            self.count = float(raw_count)
            self._first = False
        else:
            self.count = self.count * 0.6 + raw_count * 0.4
        self.raw_count = raw_count
        self.occupancy_pct = 100.0 * self.count / max(1, self.capacity)
        self.density_per_10k = raw_count / (max(1.0, area_px) / 10000.0)
        self.density_m2 = (
            raw_count / area_m2 if area_m2 is not None and area_m2 > 0 else None
        )
        self.level = level_for(self.occupancy_pct, self.warn_pct)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "polygon": self.polygon,
            "capacity": self.capacity,
            "warn_pct": self.warn_pct,
        }

    def to_dict_with_metrics(self) -> dict:
        d = self.to_dict()
        d["metrics"] = {
            "count": int(round(self.count)),
            "raw_count": self.raw_count,
            "occupancy_pct": round(self.occupancy_pct, 1),
            "density_per_10k": round(self.density_per_10k, 1),
            "density_m2": round(self.density_m2, 2) if self.density_m2 is not None else None,
            "level": self.level,
        }
        return d

def validate_polygon(polygon) -> list:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("polygon needs at least 3 [x, y] points")
    pts = []
    for p in polygon:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("polygon points must be [x, y] pairs")
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("polygon coordinates must be normalized 0..1")
        pts.append([x, y])
    return pts

def clean_zone_dict(d: dict, warn_pct_default: float = 75.0) -> dict:
    if not isinstance(d, dict) or not d.get("name"):
        raise ValueError("zone needs a name")
    warn_pct = float(d.get("warn_pct", warn_pct_default))
    if not math.isfinite(warn_pct) or not 0 < warn_pct <= 100:
        raise ValueError("warn_pct must be finite and between 0 and 100")
    return {
        "id": d.get("id") or uuid.uuid4().hex[:8],
        "name": str(d["name"])[:60],
        "polygon": validate_polygon(d.get("polygon")),
        "capacity": max(1, int(d.get("capacity", 100))),
        "warn_pct": warn_pct,
    }
