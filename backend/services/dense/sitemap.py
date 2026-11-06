"""A monitored site as zones on a real map: areas, densities, risk and evacuation routes.

Pure functions, no UI. The Map page draws what these return.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import REPO_ROOT
from .geo import adjacency, centroid, haversine_m, ring_area_m2
from .risk import judge, plan_optimal, plan_relocations, risk_level, route, thresholds_from_area

SITES_DIR = REPO_ROOT / "sites"
DEMO_SITE = SITES_DIR / "kumbh_sangam.json"


def load_site(path: Path = DEMO_SITE) -> dict:
    """Read a site file: a name, a map centre and zones with (lon, lat) rings."""
    site = json.loads(Path(path).read_text(encoding="utf-8"))
    if not site.get("zones"):
        raise ValueError(f"{path} has no zones")
    return site


# How a simulated surge builds and eases, as a share of each zone's peak density.
DEFAULT_TIMELINE = [
    {"label": "T-90 min", "factor": 0.3},
    {"label": "T-60 min", "factor": 0.5},
    {"label": "T-30 min", "factor": 0.75},
    {"label": "Peak", "factor": 1.0},
    {"label": "T+30 min", "factor": 0.6},
]


def timeline(site: dict) -> list[dict]:
    """The steps of a site's simulated surge (its own, or the default build-up)."""
    return site.get("timeline") or DEFAULT_TIMELINE


def scenario_counts(site: dict, factor: float = 1.0) -> dict:
    """People per zone in the site's simulated scenario: density x mapped area x factor."""
    return {z: round(info["scenario_density"] * factor * ring_area_m2(info["ring"]))
            for z, info in site["zones"].items() if "scenario_density" in info}


def to_geojson(view: dict) -> dict:
    """The assessed zones as a GeoJSON FeatureCollection, for GIS tools."""
    features = []
    for z, r in view["zones"].items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [r["ring"]]},
            "properties": {
                "zone": z, "name": r["name"], "area_m2": round(r["area_m2"], 1),
                "people": r["count"],
                "per_m2": None if r["per_m2"] is None else round(r["per_m2"], 3),
                "risk": r["risk"], "connected_to": view["adjacency"][z],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def list_sites() -> list[dict]:
    """Every bundled demo site, with its file path, demo first."""
    paths = [DEMO_SITE] + sorted(p for p in SITES_DIR.glob("*.json") if p != DEMO_SITE)
    return [dict(load_site(p), path=str(p)) for p in paths]


def assess(site: dict, counts: dict, preset: dict, touch_m: float | None = None,
           planner: str = "optimal", shares: dict | None = None) -> dict:
    """Everything the map shows, from zone shapes, people counts and a density standard.

    Returns {"zones": {zone: row}, "adjacency": graph, "plans": [...], "distance_m": {...}} where each
    row has the zone's name, ring, centre, area_m2, count, per_m2, risk and
    thresholds, and each plan from plan_relocations gains a "path" of zones to
    walk through. Zones without a count are kept for drawing, with count None.
    `shares` is for counts from photos: how much of each zone they show, None if unknown.
    """
    rings = {z: info["ring"] for z, info in site["zones"].items()}
    # A site can say how far apart zones may be and still connect (streets between them).
    graph = adjacency(rings, touch_m if touch_m is not None else site.get("touch_m", 30.0))
    rows = {}
    for z, info in site["zones"].items():
        area = ring_area_m2(info["ring"])
        thresholds = thresholds_from_area(area, preset)
        count = counts.get(z)
        risk = None if count is None else risk_level(count, thresholds)
        share = 1.0
        if shares is not None and count is not None:
            share = shares.get(z)
            risk, thresholds = judge(count, thresholds, share)
        rows[z] = {
            "name": info.get("name", f"Zone {z}"),
            "ring": info["ring"],
            "centre": centroid(info["ring"]),
            "area_m2": area,
            "thresholds": thresholds,
            "count": count,
            "per_m2": None if count is None else count / (area * (share or 1.0)),
            "risk": risk,
        }
    counted = {z: r for z, r in rows.items() if r["count"] is not None}
    distance = {(a, b): haversine_m(rows[a]["centre"], rows[b]["centre"])
                for a in graph for b in graph[a]}
    if planner == "simple":
        plans = plan_relocations(counted, adjacency=graph)
        for p in plans:
            p["path"] = route(p["high"], p["target"], graph) if p["target"] else []
    else:
        plans = plan_optimal(counted, graph, distance)
    return {"zones": rows, "adjacency": graph, "plans": plans, "distance_m": distance}


# Weidmann (1993): how fast people walk at a given density. Free walking is 1.34 m/s, and the crowd stops at 5.4/m².
FREE_SPEED, GAMMA, JAM_DENSITY = 1.34, 1.913, 5.4


def walking_speed(per_m2: float) -> float:
    """Average walking speed in m/s at `per_m2` people per square metre, after Weidmann's fundamental diagram."""
    if per_m2 <= 0:
        return FREE_SPEED
    if per_m2 >= JAM_DENSITY:
        return 0.0
    return FREE_SPEED * (1 - math.exp(-GAMMA * (1 / per_m2 - 1 / JAM_DENSITY)))


def walk_minutes(path: list[str], distance_m: dict, density: dict) -> float | None:
    """Minutes to walk a zone-to-zone path, each step at the speed the crowd allows between its two zones
    (`density`: zone -> people per m², unknown zones count as empty), or None if the start is packed too tightly to
    move. Queueing at exits and narrow gates is not included, so it is a lower bound."""
    if walking_speed(density.get(path[0]) or 0) < 0.05:
        return None
    seconds = 0.0
    for a, b in zip(path, path[1:]):
        metres = float(distance_m.get((a, b), distance_m.get((b, a), 0.0)))
        speed = walking_speed(((density.get(a) or 0) + (density.get(b) or 0)) / 2)
        seconds += metres / max(speed, 0.05)
    return seconds / 60


def site_from_drawings(features: list, names: dict, letters: dict) -> dict:
    """Turn shapes drawn on the map (GeoJSON features) into a site dict.

    `letters` maps a feature's index to the zone letter it was assigned, and
    `names` maps that letter to a display name. Unassigned shapes are skipped.
    """
    zones = {}
    for i, feature in enumerate(features):
        letter = letters.get(i)
        geometry = feature.get("geometry") or {}
        if not letter or geometry.get("type") != "Polygon":
            continue
        zones[letter] = {"name": names.get(letter) or f"Zone {letter}",
                         "ring": geometry["coordinates"][0]}
    if not zones:
        raise ValueError("no drawn polygon was assigned to a zone")
    lons = [p[0] for z in zones.values() for p in z["ring"]]
    lats = [p[1] for z in zones.values() for p in z["ring"]]
    return {"name": "My site", "short": "My site",
            "centre": [(min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2],
            "zoom": 17, "zones": zones}
