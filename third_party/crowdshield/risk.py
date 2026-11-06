"""Risk tiers, area-based thresholds and relocation planning. Pure functions, no UI."""

from __future__ import annotations

import math
from collections import deque

COMBINE_MODES = ("sum", "max")
UNKNOWN = "Unknown"  # a count that can't be judged, because how much of the zone it covers isn't known


def risk_level(count: int, thresholds: dict) -> str:
    """Return Safe / Moderate / High Risk / Critical for a people count."""
    if count < thresholds.get("moderate", 30):
        return "Safe"
    if count < thresholds.get("high", 60):
        return "Moderate"
    if count < thresholds.get("critical", 90):
        return "High Risk"
    return "Critical"


def thresholds_from_area(area_m2: float, preset: dict, coverage: float = 1.0) -> dict:
    """People-count thresholds for a zone: area x density limit x camera coverage.

    Coverage below 1 means the cameras see only part of the zone, so detected
    counts run low; scaling the thresholds by the same factor keeps tiers honest.
    """
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    moderate = max(1, math.floor(area_m2 * preset["moderate"] * coverage))
    high = max(moderate + 1, math.floor(area_m2 * preset["high"] * coverage))
    critical = max(high + 1, math.floor(area_m2 * preset["critical"] * coverage))
    return {"moderate": moderate, "high": high, "critical": critical}


def judge(count: int, thresholds: dict, share: float | None) -> tuple[str, dict]:
    """(risk, limits) for a count from a view of `share` of a zone, against limits for the whole zone.

    Limits shrink with the share seen. If the share isn't known, a low count proves nothing: the rest of
    the zone could be packed, so it is Unknown, never Safe. A count already past the whole zone's limits
    stands, since seeing more of the zone could only add people.
    """
    if share is None:
        risk = risk_level(count, thresholds)
        return (risk if risk in ("High Risk", "Critical") else UNKNOWN), thresholds
    limits = {k: max(1, math.floor(v * share)) for k, v in thresholds.items()}
    return risk_level(count, limits), limits


def combine_counts(counts: list[int], mode: str = "sum") -> int:
    """Merge per-photo counts for one zone.

    "sum" fits photos of different parts of the zone; "max" fits photos of the
    same area from several cameras, where summing would count people twice.
    """
    if mode not in COMBINE_MODES:
        raise ValueError(f"mode must be one of {COMBINE_MODES}")
    if not counts:
        return 0
    return sum(counts) if mode == "sum" else max(counts)


def _hops_from(start: str, adjacency: dict) -> dict:
    """Breadth-first distance, in zone-to-zone steps, from start to every reachable zone."""
    hops = {start: 0}
    queue = deque([start])
    while queue:
        zone = queue.popleft()
        for neighbour in adjacency.get(zone, ()):
            if neighbour not in hops:
                hops[neighbour] = hops[zone] + 1
                queue.append(neighbour)
    return hops


def route(start: str, goal: str, adjacency: dict) -> list[str]:
    """Shortest zone-to-zone path from start to goal (both included), or [] if unreachable."""
    previous = {start: None}
    queue = deque([start])
    while queue:
        zone = queue.popleft()
        if zone == goal:
            path = []
            while zone is not None:
                path.append(zone)
                zone = previous[zone]
            return path[::-1]
        for neighbour in adjacency.get(zone, ()):
            if neighbour not in previous:
                previous[neighbour] = zone
                queue.append(neighbour)
    return []


def plan_relocations(zones: dict, adjacency: dict | None = None) -> list[dict]:
    """Pair each High Risk or Critical zone with a Safe zone that can absorb people.

    `zones` maps zone -> {"count", "risk", "thresholds"}. Each action zone sheds
    enough people to drop below its "high" threshold, capped by the room left in
    the target, which counts people already sent there so it is never overfilled.
    With an adjacency map (zone -> neighbouring zones) the nearest reachable zone
    wins and unreachable zones are never suggested; without one, the emptiest
    Safe zone wins. A plan with target None means nobody can take the people.
    """
    action = sorted(
        (z for z, info in zones.items() if info["risk"] in ("High Risk", "Critical")),
        key=lambda z: (zones[z]["risk"] != "Critical", -zones[z]["count"]),
    )
    incoming = {z: 0 for z, info in zones.items() if info["risk"] == "Safe"}

    def room(zone: str) -> int:
        info = zones[zone]
        return info["thresholds"]["high"] - 1 - info["count"] - incoming[zone]

    plans = []
    for zone in action:
        info = zones[zone]
        excess = max(1, info["count"] - (info["thresholds"]["high"] - 1))
        hops = _hops_from(zone, adjacency) if adjacency is not None else {}
        candidates = [z for z in incoming
                      if room(z) > 0 and (adjacency is None or z in hops)]
        candidates.sort(key=lambda z: (hops.get(z, 0), zones[z]["count"] + incoming[z]))

        target = candidates[0] if candidates else None
        move = min(excess, room(target)) if target else excess
        if target:
            incoming[target] += move
        plans.append({
            "high": zone,
            "high_count": info["count"],
            "risk": info["risk"],
            "target": target,
            "target_count": zones[target]["count"] if target else 0,
            "move": move,
            "thresholds": info["thresholds"],
        })
    return plans


# ---------------------------------------------------------------------------
# Optimised relocation: minimum-cost network flow
# ---------------------------------------------------------------------------
# Walking through a busy zone is slower, so entering a zone costs its distance times (1 + this).
CROWDING_COST = {"Safe": 0.0, "Moderate": 0.25, "High Risk": 0.75, "Critical": 1.0}
UNRESOLVED_COST = 1_000_000.0  # per person left with nowhere to go: always worse than any walk


def plan_optimal(zones: dict, adjacency: dict | None = None, distance_m: dict | None = None) -> list[dict]:
    """Move the most people out of High Risk and Critical zones, along the shortest walks.

    The zones form a network: people can walk between neighbouring zones (`adjacency`),
    and pass through zones on the way. Each High Risk or Critical zone must shed enough people
    to drop below its "high" threshold; every other zone can take in people up to just below its
    own "high" threshold, never more. This is a minimum-cost flow problem, solved exactly as a
    linear program (SciPy's HiGHS): first leave as few people as possible with nowhere to go, then
    minimise the people-metres walked. The simple rule in `plan_relocations` picks one nearest
    target per zone in turn, so zones that share a destination can strand people that a global plan
    would have placed elsewhere.

    `distance_m` maps a pair (a, b) to the walking distance in metres; without it every step
    counts as 1. Returns the same plan dicts as `plan_relocations`, one per (source, target) pair
    with a "path", plus a plan with target None for people who cannot be placed.
    """
    import numpy as np
    from scipy.optimize import linprog

    names = sorted(zones)
    if adjacency is None:  # nothing known about the layout: any zone may send to any other
        adjacency = {a: [b for b in names if b != a] for a in names}
    index = {z: i for i, z in enumerate(names)}
    supply, room = {}, {}
    for z in names:
        info = zones[z]
        limit = info["thresholds"]["high"] - 1
        if info["risk"] in ("High Risk", "Critical"):
            supply[z] = max(1, info["count"] - limit)
        else:
            room[z] = 0 if info["risk"] == UNKNOWN else max(0, limit - info["count"])
    if not supply:
        return []

    def dist(a, b):
        if distance_m is None:
            return 1.0
        return float(distance_m.get((a, b), distance_m.get((b, a), 1.0)))

    edges = [(a, b) for a in names for b in adjacency.get(a, ()) if b in index]
    cost = [dist(a, b) * (1.0 + CROWDING_COST.get(zones[b]["risk"], 0.0)) for a, b in edges]
    sinks = [z for z, r in room.items() if r > 0]
    sources = list(supply)
    # variables: edge flows, then people absorbed at each sink, then people left unplaced per source
    n_e, n_s = len(edges), len(sinks)
    n = n_e + n_s + len(sources)
    c = np.array(cost + [1e-3 * zones[z]["count"] / max(1, zones[z]["thresholds"]["high"]) for z in sinks]
                 + [UNRESOLVED_COST] * len(sources))
    A = np.zeros((len(names), n))
    for k, (a, b) in enumerate(edges):
        A[index[a], k] += 1.0     # leaves a
        A[index[b], k] -= 1.0     # arrives at b
    for k, z in enumerate(sinks):
        A[index[z], n_e + k] = 1.0
    for k, z in enumerate(sources):
        A[index[z], n_e + n_s + k] = 1.0
    b_eq = np.array([float(supply.get(z, 0)) for z in names])
    bounds = ([(0, None)] * n_e + [(0, room[z]) for z in sinks] + [(0, supply[z]) for z in sources])
    res = linprog(c, A_eq=A, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"relocation LP failed: {res.message}")

    x = np.rint(res.x).astype(int)
    flow = {e: int(x[k]) for k, e in enumerate(edges) if x[k] > 0}
    absorbed = {z: int(x[n_e + k]) for k, z in enumerate(sinks) if x[n_e + k] > 0}
    unplaced = {z: int(x[n_e + n_s + k]) for k, z in enumerate(sources) if x[n_e + n_s + k] > 0}

    plans = []
    for z in sorted(sources, key=lambda s: (zones[s]["risk"] != "Critical", -zones[s]["count"])):
        info, left = zones[z], supply[z] - unplaced.get(z, 0)
        while left > 0:
            path = _flow_path(z, flow, absorbed)
            if not path:
                break
            amount = min(left, absorbed[path[-1]], *(flow[(a, b)] for a, b in zip(path, path[1:])))
            for edge in zip(path, path[1:]):
                flow[edge] -= amount
            absorbed[path[-1]] -= amount
            left -= amount
            plans.append({"high": z, "high_count": info["count"], "risk": info["risk"],
                          "target": path[-1], "target_count": zones[path[-1]]["count"],
                          "move": amount, "thresholds": info["thresholds"], "path": path})
        stranded = unplaced.get(z, 0) + left
        if stranded:
            plans.append({"high": z, "high_count": info["count"], "risk": info["risk"], "target": None,
                          "target_count": 0, "move": stranded, "thresholds": info["thresholds"], "path": []})
    return plans


def _flow_path(start: str, flow: dict, absorbed: dict) -> list[str]:
    """A walk from `start` along edges that still carry flow, ending at a zone that still takes people."""
    previous, queue = {start: None}, deque([start])
    while queue:
        zone = queue.popleft()
        if zone != start and absorbed.get(zone, 0) > 0:
            path = []
            while zone is not None:
                path.append(zone)
                zone = previous[zone]
            return path[::-1]
        for (a, b), amount in flow.items():
            if a == zone and amount > 0 and b not in previous:
                previous[b] = zone
                queue.append(b)
    return []


def plan_summary(plans: list[dict], distance_m: dict | None = None, zones: dict | None = None) -> dict:
    """People moved, people left with nowhere to go, and total people-metres walked.

    With `zones`, the people left over are what the crowded zones had to shed minus what the plans
    placed, which also catches a plan that quietly moves only part of a zone's excess. Without it,
    only plans that say target None count.
    """
    def walk(path):
        if distance_m is None:
            return float(len(path) - 1)
        return sum(distance_m.get((a, b), distance_m.get((b, a), 1.0)) for a, b in zip(path, path[1:]))

    placed = [p for p in plans if p["target"]]
    moved = sum(p["move"] for p in placed)
    if zones is not None:
        required = sum(max(1, info["count"] - (info["thresholds"]["high"] - 1))
                       for info in zones.values() if info["risk"] in ("High Risk", "Critical"))
        unresolved = max(0, required - moved)
    else:
        unresolved = sum(p["move"] for p in plans if not p["target"])
    return {"moved": moved, "unresolved": unresolved,
            "people_metres": sum(p["move"] * walk(p.get("path") or []) for p in placed)}
