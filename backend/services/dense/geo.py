"""Zones drawn on a real map: their ground area in m², their centres, and which zones touch.

Polygons come in as GeoJSON rings of (longitude, latitude) pairs. The area uses
the spherical-excess formula from Chamberlain & Duquette, "Some algorithms for
polygons on a sphere" (JPL, 2007), on a sphere of radius 6,378,137 m. Leaflet.draw
and Turf use the same formula, so the area shown here matches what the map's own
drawing tools would report. For zones a few hundred metres across the error
against the WGS84 ellipsoid is well under 1%, far below the error in any count.

Distances between zones are worked out in a flat local frame (metres east and
north of the site's centre), which is accurate to centimetres over a few km.
"""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


def _open_ring(ring) -> np.ndarray:
    """(N, 2) lon/lat array without the repeated closing point."""
    pts = np.asarray(ring, float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("a ring is a list of (longitude, latitude) pairs")
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("a polygon needs at least 3 corners")
    return pts


def ring_area_m2(ring) -> float:
    """Ground area enclosed by one ring of (lon, lat) corners, in square metres."""
    pts = np.radians(_open_ring(ring))
    lon, lat = pts[:, 0], pts[:, 1]
    # sum over corners of (lon[i+1] - lon[i-1]) * sin(lat[i])
    total = np.sum((np.roll(lon, -1) - np.roll(lon, 1)) * np.sin(lat))
    return float(abs(total) * EARTH_RADIUS_M ** 2 / 2)


def polygon_area_m2(rings) -> float:
    """Area of a GeoJSON polygon (outer ring first, then any holes), in square metres."""
    outer, *holes = rings
    return max(0.0, ring_area_m2(outer) - sum(ring_area_m2(h) for h in holes))


def haversine_m(a, b) -> float:
    """Great-circle distance in metres between two (lon, lat) points."""
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def to_local_m(points, origin) -> np.ndarray:
    """(lon, lat) points to metres east and north of `origin` (an equirectangular frame)."""
    pts = np.asarray(points, float)
    lon0, lat0 = origin
    x = np.radians(pts[:, 0] - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = np.radians(pts[:, 1] - lat0) * EARTH_RADIUS_M
    return np.c_[x, y]


def centroid(ring) -> tuple[float, float]:
    """Area-weighted centre of a ring, as (lon, lat): where a zone's label belongs."""
    pts = _open_ring(ring)
    origin = pts.mean(axis=0)
    xy = to_local_m(pts, origin)
    x, y = xy[:, 0], xy[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:  # degenerate sliver: fall back to the corner average
        return float(origin[0]), float(origin[1])
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * a)
    lon = origin[0] + math.degrees(cx / (EARTH_RADIUS_M * math.cos(math.radians(origin[1]))))
    lat = origin[1] + math.degrees(cy / EARTH_RADIUS_M)
    return float(lon), float(lat)


def _point_in_polygon(p, poly) -> bool:
    """Ray casting in the local frame."""
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _segment_distance(p, a, b) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return (orient(a, b, c) != orient(a, b, d)) and (orient(c, d, a) != orient(c, d, b))


def gap_m(ring_a, ring_b) -> float:
    """Shortest walking-free distance between two zones in metres; 0 if they touch or overlap."""
    pa, pb = _open_ring(ring_a), _open_ring(ring_b)
    origin = np.vstack([pa, pb]).mean(axis=0)
    a, b = to_local_m(pa, origin), to_local_m(pb, origin)
    if _point_in_polygon(a[0], b) or _point_in_polygon(b[0], a):
        return 0.0
    edges_a = list(zip(a, np.roll(a, -1, axis=0)))
    edges_b = list(zip(b, np.roll(b, -1, axis=0)))
    if any(_segments_cross(p, q, r, s) for p, q in edges_a for r, s in edges_b):
        return 0.0
    return min(min(_segment_distance(p, r, s) for p in a for r, s in edges_b),
               min(_segment_distance(p, r, s) for p in b for r, s in edges_a))


def adjacency(zones: dict, touch_m: float = 30.0) -> dict:
    """Zone -> sorted neighbours, where two zones are neighbours if they lie within touch_m.

    `zones` maps a zone name to its outer ring. This is the walking graph the
    relocation planner uses, so people are only ever sent to a zone they can reach.
    """
    names = sorted(zones)
    graph = {n: [] for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if gap_m(zones[a], zones[b]) <= touch_m:
                graph[a].append(b)
                graph[b].append(a)
    return graph
