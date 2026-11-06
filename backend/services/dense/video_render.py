"""Turn a video analysis into something a person can watch: the clip itself, annotated.

The monitor counts heads once per second and keeps per-zone numbers. `PointRecorder` wraps the
counter so each second's head points are kept too, and `render` then reads the clip a second time,
frame by frame, and draws on it: a see-through dot on every head, each zone outlined in the colour of
its risk, a running count, and a banner while a surge alarm is active. Frames stream straight into an
H.264 MP4 (via imageio-ffmpeg, which brings its own encoder) that every browser plays, so a clip is
never held in memory whole.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .detection import draw_head_dots

# BGR. Capacity levels match the Live feed; density levels match the Video monitor's Fruin levels.
LEVEL_BGR = {
    "Safe": (129, 185, 16), "Moderate": (11, 158, 245), "High Risk": (22, 115, 249), "Critical": (68, 68, 239),
    "Normal": (129, 185, 16), "Crowded": (11, 158, 245), "Dangerous": (22, 115, 249), "Crush risk": (68, 68, 239),
}
NEUTRAL_BGR = (235, 200, 40)
SURGE_SHOW_S = 1.5  # keep a surge banner up this long after the alarm, so a viewer can read it


class PointRecorder:
    """A counter for VideoMonitor that also remembers where the heads were at each count."""

    def __init__(self):
        self.points: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        from .monitor import points_to_grid
        from .points import count_points

        out = count_points(frame)
        self.points.append(out["points"])
        grid = points_to_grid(out["points"], frame.shape)
        if out["raw_count"]:
            grid *= out["count"] / out["raw_count"]
        return grid


def capacity_level(people: float, capacity: float | None) -> str | None:
    """Safe / Moderate / High Risk / Critical against what a zone holds comfortably (as on Live feed)."""
    if not capacity:
        return None
    share = people / capacity
    return "Critical" if share >= 1 else "High Risk" if share >= 0.75 else "Moderate" if share >= 0.5 else "Safe"


def zone_level(row: dict, capacity: float | None) -> str | None:
    """People per m² levels when the floor is calibrated, otherwise levels against capacity."""
    return row.get("level") or capacity_level(row["people"], capacity)


def _label(img, text, org, scale=0.55, colour=(255, 255, 255), bg=(15, 23, 42)):
    import cv2

    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + base + 2), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, t: float, zones: list[tuple[str, list]], readings: dict,
             capacities: dict, points=None, surge: str | None = None) -> np.ndarray:
    """One annotated frame. `readings` maps zone name -> its latest monitor row (or is empty)."""
    import cv2

    out = draw_head_dots(frame, points) if points is not None and len(points) else frame.copy()
    for name, poly in zones:
        row = readings.get(name)
        level = zone_level(row, capacities.get(name)) if row else None
        colour = LEVEL_BGR.get(level, NEUTRAL_BGR)
        pts = np.int32(poly)
        cv2.polylines(out, [pts], True, colour, 3, cv2.LINE_AA)
        x, y = pts.min(axis=0)
        text = name if row is None else f"{name}: {row['people']:.0f}" + (f" - {level}" if level else "")
        _label(out, text, (int(x) + 8, int(y) + 22), colour=(255, 255, 255), bg=colour)
    clock = f"Crowd Shield  {int(t // 60):02d}:{t % 60:04.1f}"
    (tw, _), _ = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    _label(out, clock, (out.shape[1] - tw - 12, 24), scale=0.5)  # top-right: zone labels sit top-left
    if surge:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, h - 36), (w, h), (40, 40, 220), -1)
        cv2.putText(out, f"SURGE ALARM: {surge}"[:90], (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _writer(out_path: str | Path, width: int, height: int, fps: float, crf: int = 28):
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(out_path), (width, height), fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        quality=None, output_params=["-crf", str(crf), "-preset", "veryfast", "-movflags", "+faststart"])
    writer.send(None)
    return writer


def encode_clean(frames: Iterable[np.ndarray], fps: float, out_path: str | Path) -> Path:
    """The clip as a small browser-playable MP4, untouched: the page's player draws the analysis on top."""
    import cv2

    writer = None
    try:
        for frame in frames:
            h, w = frame.shape[:2]
            frame = frame[: h - h % 2, : w - w % 2]
            if writer is None:
                writer = _writer(out_path, frame.shape[1], frame.shape[0], fps, crf=30)
            writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        if writer is not None:
            writer.close()
    return Path(out_path)


def render(frames: Iterable[np.ndarray], fps: float, out_path: str | Path, zones: list[tuple[str, list]],
           rows: list[dict], events: list[dict], capacities: dict | None = None,
           points: list[np.ndarray] | None = None, count_every: int | None = None,
           progress: Callable[[int], None] | None = None) -> Path:
    """Write the annotated clip as a browser-playable MP4 and return its path."""
    import cv2

    capacities = capacities or {}
    count_every = count_every or max(1, round(fps))
    by_time: dict[float, dict] = {}
    for r in rows:
        by_time.setdefault(r["t"], {})[r["zone"]] = r
    times = sorted(by_time)
    writer, readings, k = None, {}, -1
    try:
        for i, frame in enumerate(frames):
            t = i / fps
            while k + 1 < len(times) and times[k + 1] <= t + 1e-6:
                k += 1
                readings = by_time[times[k]]
            pts = points[min(i // count_every, len(points) - 1)] if points else None
            surge = next((e["event"].split(": ", 1)[-1] for e in events if 0 <= t - e["t"] <= SURGE_SHOW_S), None)
            img = annotate(frame, t, zones, readings, capacities, pts, surge)
            h, w = img.shape[:2]
            img = img[: h - h % 2, : w - w % 2]  # H.264 needs even sides
            if writer is None:
                writer = _writer(out_path, img.shape[1], img.shape[0], fps)
            writer.send(np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if progress:
                progress(i)
    finally:
        if writer is not None:
            writer.close()
    return Path(out_path)
