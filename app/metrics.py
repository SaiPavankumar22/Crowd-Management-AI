"""Dependency-free Prometheus metrics, with labels.

Labels matter for the observability requirement in `improvement.md`: a count
without knowing *which camera, which ghat and which model version* produced it
cannot be audited after an event.
"""
from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_counters = defaultdict(float)
_gauges = defaultdict(float)
_types: dict[str, str] = {}


def _key(name: str, labels: dict | None) -> str:
    if not labels:
        return name
    rendered = ",".join(
        f'{k}="{str(v).replace(chr(34), chr(39))}"' for k, v in sorted(labels.items())
    )
    return f"{name}{{{rendered}}}"


def inc(name: str, value: float = 1.0, labels: dict | None = None):
    key = _key(name, labels)
    with _lock:
        _counters[key] += value
        _types.setdefault(key, "counter")


def set_gauge(name: str, value: float, labels: dict | None = None):
    key = _key(name, labels)
    with _lock:
        _gauges[key] = value
        _types.setdefault(key, "gauge")


def render() -> str:
    lines = []
    with _lock:
        counters = dict(_counters)
        gauges = dict(_gauges)
        types = dict(_types)
    emitted: set[str] = set()
    for name, value in sorted(counters.items()):
        base = name.split("{", 1)[0]
        if base not in emitted:
            lines.append(f"# TYPE {base} counter")
            emitted.add(base)
        lines.append(f"{name} {value:g}")
    for name, value in sorted(gauges.items()):
        base = name.split("{", 1)[0]
        if base not in emitted:
            lines.append(f"# TYPE {base} gauge")
            emitted.add(base)
        lines.append(f"{name} {value:g}")
    return "\n".join(lines) + "\n"
