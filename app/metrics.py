"""Dependency-free Prometheus metrics for the single-process service."""
from __future__ import annotations

import threading
from collections import defaultdict


_lock = threading.Lock()
_counters = defaultdict(float)
_gauges = defaultdict(float)


def inc(name: str, value: float = 1.0):
    with _lock:
        _counters[name] += value


def set_gauge(name: str, value: float):
    with _lock:
        _gauges[name] = value


def render() -> str:
    lines = []
    with _lock:
        counters = dict(_counters)
        gauges = dict(_gauges)
    for name, value in sorted(counters.items()):
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value:g}")
    for name, value in sorted(gauges.items()):
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value:g}")
    return "\n".join(lines) + "\n"
