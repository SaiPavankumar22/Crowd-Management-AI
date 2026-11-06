"""Watch a fixed-camera video zone by zone: people per m², speed, crowd pressure and warnings.

Every frame feeds the optical flow (movement) and the panic alarm. Once per
`count_every_s` seconds a counting model reads the frame, and each zone gets a
reading:

    people      people inside the zone
    per_m2      people per square metre (needs a ground calibration)
    level       Normal / Crowded / Dangerous / Crush risk, from per_m2
    speed       average ground speed of moving pixels (m/s, or px/s uncalibrated)
    pressure    Helbing's crowd pressure, per_m2 x velocity variance (1/s²)
    eta_s       seconds until the zone reaches 5 people per m² at its current trend

The counter is any function that takes a BGR frame and returns a density grid
at 1/8 of the frame's size whose sum is the head count, so a density model and
a point model plug in the same way (see `points_to_grid`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .forecast import seconds_until
from .geometry import GroundCalibration
from .motion import MOVING_PX, MotionAlarm, flow_stats, optical_flow
from .safety import _inside, crowd_pressure, density_level

CELL = 8  # density grid cell size in pixels
# Crowd pressure uses velocity variance among neighbours: within 1 m ground cells
# when calibrated, else within 32-pixel cells.
PRESSURE_CELL_M = 1.0
PRESSURE_CELL_PX = 32
MIN_CELL_SAMPLES = 8


@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # N x 2 image points, in pixels

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, float)


def points_to_grid(points_yx, frame_shape, cell: int = CELL) -> np.ndarray:
    """Head points (N x 2, row then column, in pixels) as a count grid at 1/cell size."""
    h, w = frame_shape[:2]
    grid = np.zeros((max(1, h // cell), max(1, w // cell)))
    p = np.asarray(points_yx, float).reshape(-1, 2)
    rows = np.clip((p[:, 0] // cell).astype(int), 0, grid.shape[0] - 1)
    cols = np.clip((p[:, 1] // cell).astype(int), 0, grid.shape[1] - 1)
    np.add.at(grid, (rows, cols), 1.0)
    return grid


def zone_mask(polygon_px, shape, scale: float) -> np.ndarray:
    """Boolean mask of a zone polygon on an image of `shape`, whose pixels are `scale` x full size."""
    h, w = shape[:2]
    rows, cols = np.mgrid[0:h, 0:w]
    centres = np.c_[(cols.ravel() + 0.5) / scale, (rows.ravel() + 0.5) / scale]
    return _inside(centres, np.asarray(polygon_px, float)).reshape(h, w)


class VideoMonitor:
    def __init__(self, counter: Callable[[np.ndarray], np.ndarray], fps: float,
                 zones: list[Zone], calib: GroundCalibration | None = None,
                 count_every_s: float = 1.0, flow_scale: float = 0.5,
                 sample_step: int = 4, alarm: MotionAlarm | None = None):
        self.counter = counter
        self.fps = float(fps)
        self.zones = zones
        self.calib = calib
        self.count_every = max(1, round(count_every_s * self.fps))
        self.flow_scale = flow_scale
        self.sample_step = sample_step
        self.alarm = alarm or MotionAlarm(baseline_frames=round(3 * self.fps),
                                          hold_frames=max(1, round(0.5 * self.fps)))
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self._velocities = {z.name: [] for z in zones}
        self._history = {z.name: ([], []) for z in zones}
        self._masks = None
        self._areas = None

    def _setup(self, frame: np.ndarray, small_shape):
        h, w = frame.shape[:2]
        grid_shape = (max(1, h // CELL), max(1, w // CELL))
        self._masks = {z.name: zone_mask(z.polygon, small_shape, self.flow_scale)
                       for z in self.zones}
        self._grid_masks = {z.name: zone_mask(z.polygon, grid_shape, 1 / CELL)
                            for z in self.zones}
        self._areas = {z.name: self.calib.polygon_area(z.polygon) if self.calib else float("nan")
                       for z in self.zones}

    def _collect_velocities(self, flow: np.ndarray):
        """Keep a sparse sample of each zone's moving pixels' velocities (m/s or px/s)."""
        step = self.sample_step
        sub = flow[::step, ::step]
        mag = np.hypot(sub[..., 0], sub[..., 1])
        for z in self.zones:
            mask = self._masks[z.name][::step, ::step] & (mag > MOVING_PX)
            if not mask.any():
                continue
            rows, cols = np.nonzero(mask)
            flow_px = sub[rows, cols] / self.flow_scale  # back to full-size pixels per frame
            points_px = np.c_[(cols * step + 0.5) / self.flow_scale,
                              (rows * step + 0.5) / self.flow_scale]
            if self.calib is not None:
                v = self.calib.ground_velocity(points_px, flow_px, self.fps)
                cells = np.floor(self.calib.to_ground(points_px) / PRESSURE_CELL_M)
                ok = np.isfinite(v).all(axis=1) & np.isfinite(cells).all(axis=1)
                v, cells = v[ok], cells[ok]
            else:
                v = flow_px * self.fps
                cells = np.floor(points_px / PRESSURE_CELL_PX)
            self._velocities[z.name].append((cells.astype(np.int64), v))

    def _read_zones(self, t: float, frame: np.ndarray):
        grid = self.counter(frame)
        for z in self.zones:
            mask = self._grid_masks[z.name]
            gm = mask[:grid.shape[0], :grid.shape[1]]
            people = float(grid[:gm.shape[0], :gm.shape[1]][gm].sum())
            area = self._areas[z.name]
            per_m2 = people / area if area > 0 else float("nan")
            samples = self._velocities[z.name]
            v = np.concatenate([s[1] for s in samples]) if samples else np.zeros((0, 2))
            speed = float(np.hypot(v[:, 0], v[:, 1]).mean()) if len(v) else 0.0
            pressure = (self._local_pressure(per_m2, samples)
                        if len(v) and np.isfinite(per_m2) else float("nan"))
            times, values = self._history[z.name]
            times.append(t)
            values.append(per_m2)
            self.rows.append({
                "t": round(t, 2), "zone": z.name, "people": people, "area_m2": area,
                "per_m2": per_m2,
                "level": density_level(per_m2) if np.isfinite(per_m2) else None,
                "speed": speed, "pressure": pressure,
                "eta_s": seconds_until(times, values) if np.isfinite(per_m2) else None,
            })
            self._velocities[z.name] = []

    @staticmethod
    def _local_pressure(per_m2: float, samples) -> float:
        """Helbing's pressure with the velocity variance taken among neighbours.

        Variance across a whole zone would count people walking calmly in
        different directions as turbulence; within one small cell it only grows
        when people next to each other are pushed in different ways.
        """
        cells = np.concatenate([s[0] for s in samples])
        v = np.concatenate([s[1] for s in samples])
        _, cell_id, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        cell_id = cell_id.ravel()
        values = [crowd_pressure(per_m2, v[cell_id == c])
                  for c in np.nonzero(counts >= MIN_CELL_SAMPLES)[0]]
        return float(np.mean(values)) if values else 0.0

    def run(self, frames: Iterable[np.ndarray], progress: Callable[[int], None] | None = None):
        """Process every frame; returns (zone readings, alarm events)."""
        import cv2

        prev = None
        for i, frame in enumerate(frames):
            small = cv2.resize(frame, None, fx=self.flow_scale, fy=self.flow_scale,
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if self._masks is None:
                self._setup(frame, gray.shape)
            if prev is not None:
                flow = optical_flow(prev, gray)
                self._collect_velocities(flow)
                stats = flow_stats(flow)
                if self.alarm.update(stats):
                    self.events.append({"t": round(i / self.fps, 2),
                                        "event": f"Sudden movement: {self.alarm.reason}"})
            if i % self.count_every == 0:
                self._read_zones(i / self.fps, frame)
            prev = gray
            if progress:
                progress(i)
        return self.rows, self.events
