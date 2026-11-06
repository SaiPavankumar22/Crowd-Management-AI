"""Live monitoring: grab a frame from a camera stream every few seconds, count it, watch the trend.

This is sampled monitoring, not video-rate tracking. Counting a frame with the head-point model takes
seconds, so the monitor looks at one frame every `interval` seconds and judges each zone by its count,
its trend and the time left until it fills. For speed, pressure and the surge alarm (which need
every frame from a fixed camera), use the Video page or run on a machine with a GPU.

Sources: RTSP, HTTP(S) streams (HLS .m3u8, MJPEG), a video file path, or the built-in demo clip.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .forecast import seconds_until
from .monitor import CELL, zone_mask

SCHEMES = ("rtsp", "rtsps", "http", "https")
MAX_WIDTH = 960
RISK_ORDER = {"Safe": 0, "Moderate": 1, "High Risk": 2, "Critical": 3}
# Share of a zone's capacity at which each risk level starts.
CAPACITY_SHARES = {"moderate": 0.5, "high": 0.75, "critical": 1.0}


def check_url(url: str, allow_private: bool | None = None) -> tuple[bool, str]:
    """Is this a stream address we are willing to open from the server?

    Only rtsp/rtsps/http/https, and by default only addresses on the public internet, so a hosted
    copy of the app can't be pointed at the server's own network. Set CROWDSHIELD_ALLOW_PRIVATE_URLS=1
    on your own machine to monitor a camera on your local network.
    """
    if allow_private is None:
        allow_private = os.environ.get("CROWDSHIELD_ALLOW_PRIVATE_URLS") == "1"
    parts = urlparse(url.strip())
    if parts.scheme.lower() not in SCHEMES or not parts.hostname:
        return False, "Use an address starting with rtsp://, http:// or https://."
    if allow_private:
        return True, ""
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parts.hostname, parts.port or 80)}
    except OSError:
        return False, "That address could not be found."
    for address in addresses:
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            return False, ("That address is on a private network, which this hosted app cannot reach. "
                           "Run Crowd Shield on your own computer for a camera on your network.")
    return True, ""


# Many public webcams (seafronts, town squares, ski slopes) are hosted on ipcamlive.com. Their stream
# address changes, so a camera's page link is turned into its current stream with the same open lookup
# the site's own player uses.
IPCAMLIVE_LOOKUP = "https://ipcamlive.com/player/getcamerastreamstate.php"
PUBLIC_CAMERAS = {
    "Blackpool seafront, UK (live)": {
        "link": "https://ipcamlive.com/6745eb6be37dd",
        "credit": "Live public webcam by Granada Apartments, Blackpool, hosted on ipcamlive.com. Busiest in the "
                  "daytime and on summer evenings; quiet at night.",
    },
}


def ipcamlive_alias(url: str) -> str | None:
    """The camera alias in an ipcamlive link (ipcamlive.com/<alias> or player.php?alias=<alias>), else None."""
    import re

    parsed = urlparse(url.strip())
    if not parsed.hostname or not parsed.hostname.endswith("ipcamlive.com") or parsed.path.endswith(".m3u8"):
        return None
    match = re.search(r"alias=([0-9a-z]+)", parsed.query) or re.fullmatch(r"/([0-9a-z]{8,20})/?", parsed.path)
    return match.group(1) if match else None


def resolve_stream(url: str, timeout_s: float = 15.0) -> tuple[str | None, str]:
    """(stream address, "") for a link the app can read: an ipcamlive camera page is looked up; any other
    address is returned as it is. (None, reason) when the camera is offline or the lookup fails."""
    alias = ipcamlive_alias(url)
    if not alias:
        return url.strip(), ""
    import json
    import time
    import urllib.request

    lookup = f"{IPCAMLIVE_LOOKUP}?_={int(time.time() * 1000)}&token=&alias={alias}&targetdomain=ipcamlive.com"
    request = urllib.request.Request(lookup, headers={
        "User-Agent": "Mozilla/5.0 (Crowd Shield)", "Referer": f"https://ipcamlive.com/player/player.php?alias={alias}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            details = json.loads(response.read().decode())["details"]
    except Exception as exc:
        return None, f"Could not look up that camera on ipcamlive ({type(exc).__name__})."
    if str(details.get("streamavailable")) != "1" or not details.get("streamid"):
        return None, "That camera is offline right now. Try again later or pick another one."
    return f"{details['address']}streams/{details['streamid']}/stream.m3u8", ""


def grab_frame(source, timeout_ms: int = 10000, skip: int = 2, max_width: int = MAX_WIDTH):
    """The newest frame from a stream or file, shrunk to `max_width`, or None if it cannot be read.

    Streams buffer, so a few frames are read and thrown away to get a current picture.
    """
    import cv2

    if isinstance(source, str) and source.lower().startswith("rtsp"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if isinstance(source, str) else cv2.VideoCapture(source)
    for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        if hasattr(cv2, prop):
            cap.set(getattr(cv2, prop), timeout_ms)
    frame = None
    try:
        if not cap.isOpened():
            return None
        for _ in range(skip + 1):
            ok, latest = cap.read()
            if not ok:
                break
            frame = latest
    finally:
        cap.release()
    if frame is None:
        return None
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


class DemoFeed:
    """A recorded clip played as if it were live: each call returns the frame one step later, looping."""

    def __init__(self, path: Path, step_s: float = 1.0):
        import cv2

        self.path, self.step_s = Path(path), step_s
        cap = cv2.VideoCapture(str(self.path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.position = 0

    def next_frame(self, max_width: int = MAX_WIDTH):
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.position)
        ok, frame = cap.read()
        cap.release()
        self.position = int(self.position + self.step_s * self.fps) % max(1, self.frames - 1)
        if not ok:
            return None
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return frame


def capacity_thresholds(capacity: int) -> dict:
    """Risk limits from the number of people a zone holds comfortably."""
    capacity = max(2, int(capacity))
    return {k: max(1, round(capacity * share)) for k, share in CAPACITY_SHARES.items()}


class LiveMonitor:
    """Counts people per zone on each new frame and raises an event when a zone gets worse.

    `zones` maps a name to {"polygon": N x 2 pixels, "thresholds": {...}, optional "area_m2"}.
    `counter(frame)` returns a count grid (the point model's `point_grid`). An event fires when a
    zone's risk level rises, or when a zone's count climbs fast; the same level is not announced
    again for `cooldown_s` seconds.
    """

    def __init__(self, zones: dict, counter, cooldown_s: float = 600.0, history: int = 240):
        self.zones, self.counter, self.cooldown_s = zones, counter, cooldown_s
        self.history = {z: deque(maxlen=history) for z in zones}
        self.level = {z: "Safe" for z in zones}
        self.last_alert = {}
        self._masks = None

    def _setup(self, grid_shape):
        self._masks = {z: zone_mask(info["polygon"], grid_shape, 1 / CELL) for z, info in self.zones.items()}

    def update(self, t: float, frame) -> dict:
        grid = self.counter(frame)
        if self._masks is None:
            self._setup(grid.shape)
        out, events = {}, []
        for z, info in self.zones.items():
            mask = self._masks[z][:grid.shape[0], :grid.shape[1]]
            count = float(grid[:mask.shape[0], :mask.shape[1]][mask].sum())
            th = info["thresholds"]
            risk = ("Critical" if count >= th["critical"] else "High Risk" if count >= th["high"]
                    else "Moderate" if count >= th["moderate"] else "Safe")
            hist = self.history[z]
            hist.append((t, count))
            times, values = zip(*hist)
            eta = seconds_until(times, values, level=th["critical"], window_s=max(300.0, 4 * self._step(times)),
                                min_points=3)
            per_m2 = count / info["area_m2"] if info.get("area_m2") else None
            out[z] = {"count": count, "risk": risk, "per_m2": per_m2, "eta_s": eta, "capacity": th["critical"]}
            if RISK_ORDER[risk] >= RISK_ORDER["High Risk"] and RISK_ORDER[risk] > RISK_ORDER[self.level[z]]:
                if t - self.last_alert.get((z, risk), -1e18) >= self.cooldown_s:
                    self.last_alert[(z, risk)] = t
                    events.append({"zone": z, "kind": "level", "risk": risk, "count": count,
                                   "per_m2": per_m2, "eta_s": eta})
            elif len(values) >= 4 and values[-4] >= th["moderate"] and count >= 1.4 * values[-4] \
                    and RISK_ORDER[risk] < RISK_ORDER["Critical"]:
                if t - self.last_alert.get((z, "build-up"), -1e18) >= self.cooldown_s:
                    self.last_alert[(z, "build-up")] = t
                    events.append({"zone": z, "kind": "build-up", "risk": risk, "count": count,
                                   "per_m2": per_m2, "eta_s": eta})
            self.level[z] = risk
        return {"t": t, "zones": out, "events": events}

    @staticmethod
    def _step(times) -> float:
        return float(np.median(np.diff(times))) if len(times) > 1 else 60.0
