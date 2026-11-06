"""Save and load a working session as a small JSON file on the user's own computer.

The hosted app keeps nothing between visits (no database, by design: no photos or contacts are stored
on a server). This lets someone carry their zone counts, risk limits, contacts and drawn site across
visits themselves. Photos and overlay images are not saved, only the numbers.
"""

from __future__ import annotations

import json
import re

from .config import DEFAULT_THRESHOLDS, SEGMENTS

VERSION = 1
MAX_BYTES = 1_000_000
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class BadSession(ValueError):
    """The file is not a Crowd Shield session, or its contents don't make sense."""


def export_session(state) -> str:
    """The parts of the session worth keeping, as JSON text. `state` is st.session_state (or a dict)."""
    counts = {seg: {"count": int(d["count"]), "image_count": int(d.get("image_count", 0)),
                    "counter": d.get("counter")}
              for seg, d in (state.get("crowd_data") or {}).items()}
    payload = {
        "crowd_shield_session": VERSION,
        "thresholds": state.get("thresholds"),
        "seg_areas": state.get("seg_areas"),
        "agents": state.get("agents"),
        "zone_names": state.get("zone_names"),
        "zone_adjacency": state.get("zone_adjacency"),
        "custom_site": state.get("custom_site"),
        "limits_source": state.get("limits_source"),
        "counts": counts,
    }
    return json.dumps({k: v for k, v in payload.items() if v not in (None, {}, [])}, indent=1)


def _zone_map(value, check, what):
    if not isinstance(value, dict):
        raise BadSession(f"{what} should be a list of zones.")
    out = {}
    for seg, item in value.items():
        if seg not in SEGMENTS:
            raise BadSession(f"{what} has an unknown zone {seg!r}.")
        out[seg] = check(item, seg)
    return out


def _limits(item, seg):
    try:
        t = {k: int(item[k]) for k in ("moderate", "high", "critical")}
    except (KeyError, TypeError, ValueError):
        raise BadSession(f"Zone {seg}'s risk limits are incomplete.") from None
    if not 0 < t["moderate"] < t["high"] < t["critical"]:
        raise BadSession(f"Zone {seg}'s limits must rise from Moderate to High Risk to Critical.")
    return t


def _contact(item, seg):
    if item and not (isinstance(item, str) and _EMAIL.match(item)):
        raise BadSession(f"Zone {seg}'s contact is not an email address.")
    return item or ""


def _count(item, seg):
    try:
        count = int(item["count"])
    except (KeyError, TypeError, ValueError):
        raise BadSession(f"Zone {seg}'s count is missing.") from None
    if count < 0:
        raise BadSession(f"Zone {seg}'s count is negative.")
    return {"count": count, "body_count": count, "head_count": 0, "image_count": int(item.get("image_count") or 0),
            "per_image": [], "counter": item.get("counter") if isinstance(item.get("counter"), str) else None,
            "combine": "sum"}


def import_session(text: str) -> dict:
    """Check a saved file and return the session-state values it holds. Raises BadSession if unusable."""
    if len(text) > MAX_BYTES:
        raise BadSession("That file is too big to be a Crowd Shield session.")
    try:
        raw = json.loads(text)
    except ValueError:
        raise BadSession("That file is not a Crowd Shield session (it isn't JSON).") from None
    if not isinstance(raw, dict) or raw.get("crowd_shield_session") != VERSION:
        raise BadSession("That file is not a Crowd Shield session.")
    out = {}
    if "thresholds" in raw:
        out["thresholds"] = {s: dict(DEFAULT_THRESHOLDS) for s in SEGMENTS}
        out["thresholds"].update(_zone_map(raw["thresholds"], _limits, "The risk limits"))
    if "seg_areas" in raw:
        out["seg_areas"] = {s: 0 for s in SEGMENTS}
        out["seg_areas"].update(_zone_map(raw["seg_areas"], lambda v, s: max(0, int(v)), "The areas"))
    if "agents" in raw:
        out["agents"] = {s: "" for s in SEGMENTS}
        out["agents"].update(_zone_map(raw["agents"], _contact, "The contacts"))
    if "zone_names" in raw:
        out["zone_names"] = _zone_map(raw["zone_names"], lambda v, s: str(v)[:80], "The zone names")
    if "zone_adjacency" in raw:
        out["zone_adjacency"] = _zone_map(
            raw["zone_adjacency"], lambda v, s: [z for z in v if z in SEGMENTS and z != s], "The routes")
    if "custom_site" in raw and isinstance(raw["custom_site"], dict):
        out["custom_site"] = raw["custom_site"]
    if isinstance(raw.get("limits_source"), str):
        out["limits_source"] = raw["limits_source"][:200]
    if "counts" in raw:
        out["crowd_data"] = _zone_map(raw["counts"], _count, "The counts")
    return out
