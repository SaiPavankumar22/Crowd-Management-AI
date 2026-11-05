"""Confidence and mode router.

`improvement.md` §"Mode router": detector boxes are trustworthy only while
people are separable. When they are not, the system must say so out loud
(`ANALYTICS_DEGRADED`) and keep the density estimate advisory rather than
silently blending two numbers into a confident-looking crowd count.

Rules implemented here:

* SPARSE / MODERATE  -> detector + tracker is the operational count.
* DENSE              -> tracking is marked unreliable, density is shown
                        separately, `selected_count` is null pending review.
* UNKNOWN            -> camera quality or model confidence is insufficient.

Mode changes use hysteresis (N consecutive frames) so the dashboard does not
flicker between modes. Density output can only become operational after that
estimator is marked validated *and* the per-camera policy allows it; even then
the source is reported with every count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .adapters import DensityResult, DetectionResult

SPARSE = "sparse"
MODERATE = "moderate"
DENSE = "dense"
UNKNOWN = "unknown"

# How the operating state is named in the UI / API.
MODE_LABEL = {
    SPARSE: "detector",
    MODERATE: "detector_crosscheck",
    DENSE: "degraded_review",
    UNKNOWN: "degraded_review",
    "detail": "detector_lower_bound",
}

BAD_CAMERA_STATES = {"stale", "frozen", "dark", "low_contrast", "blurry"}


@dataclass
class RouterConfig:
    dense_box_height_px: float = 32.0
    dense_min_count: int = 12
    moderate_min_count: int = 4
    # A crowd at conf 0.3-0.5 is normal for a small detector; only a majority
    # of weak boxes plus a real crowd is evidence that detection is failing.
    # An over-eager threshold here would degrade every camera and make the
    # warning worthless (alarm fatigue).
    low_confidence_ratio: float = 0.6
    min_track_continuity: float = 0.55
    hysteresis_frames: int = 3
    disagreement_tolerance: float = 0.35


@dataclass
class RouterOutput:
    mode: str
    estimation_mode: str
    detector_count: int
    density_count: float | None
    selected_count: int | None
    count_source: str
    confidence: str
    reason: str
    degraded: bool
    review_required: bool
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "estimation_mode": self.estimation_mode,
            "detector_count": self.detector_count,
            "density_count": (round(self.density_count, 1)
                              if self.density_count is not None else None),
            "selected_count": self.selected_count,
            "count_source": self.count_source,
            "confidence": self.confidence,
            "reason": self.reason,
            "degraded": self.degraded,
            "review_required": self.review_required,
            **self.detail,
        }


class ModeRouter:
    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self.mode = SPARSE
        self._candidate = SPARSE
        self._candidate_streak = 0

    # ------------------------------------------------------------------ #
    def reset(self):
        self.mode = SPARSE
        self._candidate = SPARSE
        self._candidate_streak = 0

    def _classify(self, detection: DetectionResult, track_continuity: float,
                  camera_health: str | None) -> tuple[str, str]:
        cfg = self.config
        if camera_health in BAD_CAMERA_STATES:
            return UNKNOWN, f"camera_{camera_health}"
        # The full-frame pass found almost nobody but a tiled probe found people:
        # resolution, not emptiness, is the limit. The recovered count is a lower
        # bound, so the mode is degraded rather than trusted. A small gap
        # (5 vs 9 in a narrow view) is ordinary tiling gain, not an undercount;
        # only a clear multiple justifies telling the operator the number is a
        # floor.
        if (detection.tier == "detail" and detection.count > 1
                and detection.count >= max(8, 2 * (detection.primary_count or 0))):
            return DENSE, "resolution_limited_undercount"
        if detection.count == 0:
            # An empty frame is only UNKNOWN if the camera looks wrong; a
            # genuinely empty ghat is SPARSE, not degraded.
            return SPARSE, "empty_scene"
        if (detection.median_box_height() < cfg.dense_box_height_px
                and detection.count >= cfg.dense_min_count):
            return DENSE, "small_boxes_many_people"
        # Both remaining signals only mean "the detector has lost the crowd"
        # when there is actually a crowd: a handful of people with weak boxes
        # is normal, and degrading on it would train operators to ignore the
        # warning.
        if (detection.low_confidence_ratio() >= cfg.low_confidence_ratio
                and detection.count >= cfg.dense_min_count):
            return DENSE, "many_low_confidence_boxes"
        if (track_continuity < cfg.min_track_continuity
                and detection.count >= cfg.dense_min_count):
            return DENSE, "track_continuity_failed"
        if detection.count >= cfg.moderate_min_count:
            return MODERATE, "moderate_density"
        return SPARSE, "sparse"

    def _apply_hysteresis(self, candidate: str) -> str:
        if candidate == self.mode:
            self._candidate = candidate
            self._candidate_streak = 0
            return self.mode
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_streak = 0
        self._candidate_streak += 1
        # Losing trust is immediate: never keep showing a confident count while
        # the evidence says the detector has failed. Recovering trust, or moving
        # between the two detector modes, is deliberate (hysteresis).
        urgent = candidate in (DENSE, UNKNOWN) and self.mode not in (DENSE, UNKNOWN)
        needed = 1 if urgent else max(1, self.config.hysteresis_frames)
        if self._candidate_streak >= needed:
            self.mode = candidate
            self._candidate_streak = 0
        return self.mode

    # ------------------------------------------------------------------ #
    def route(self, detection: DetectionResult, *, track_continuity: float = 1.0,
              density: DensityResult | None = None, density_error: str | None = None,
              camera_health: str | None = None,
              calibrated_density_enabled: bool = False) -> RouterOutput:
        candidate, reason = self._classify(detection, track_continuity, camera_health)
        mode = self._apply_hysteresis(candidate)
        detector_count = detection.count
        density_count = density.total_count if density is not None else None

        detail = {
            "tier": detection.tier,
            "primary_count": detection.primary_count,
            "median_box_height": round(detection.median_box_height(), 1),
            "low_confidence_ratio": round(detection.low_confidence_ratio(), 2),
            "track_continuity": round(float(track_continuity), 2),
            "model": detection.model_name,
            "model_version": detection.model_version,
            "tiled": detection.tiled,
        }

        if mode == UNKNOWN:
            return RouterOutput(
                UNKNOWN, MODE_LABEL[UNKNOWN], detector_count, density_count,
                None, "none", "low",
                f"camera or model state insufficient ({reason})",
                True, True, detail)

        if mode == DENSE:
            # A count recovered at higher resolution is a lower bound, not a
            # failure of the camera: say which one it is, because the operator
            # response is different (judge by eye vs fix the lens).
            if detection.tier == "detail":
                primary = detection.primary_count
                return RouterOutput(
                    DENSE, "detector_lower_bound", detector_count, density_count,
                    None, "detector_lower_bound", "low",
                    (f"resolution limit: full-frame pass saw {primary} person(s); "
                     f"{detector_count} is a lower bound recovered by tiled detail"
                     " (operator review required)"),
                    True, True, detail)
            if density_count is None:
                why = ("high_occlusion; density model unavailable"
                       if density_error else "high_occlusion")
            else:
                why = "high_occlusion; density estimate advisory pending validation"
            return RouterOutput(
                DENSE, MODE_LABEL[DENSE], detector_count, density_count,
                None, "none", "low",
                f"{why} (operator review required)",
                True, True, detail)

        # SPARSE / MODERATE: the detector is operational. A validated density
        # estimate can only *warn* about disagreement here, never replace it
        # unless the camera policy explicitly enables calibrated density.
        if calibrated_density_enabled and density is not None and density.confidence == "validated":
            return RouterOutput(
                mode, "calibrated_density", detector_count, density_count,
                int(round(density_count)), "density_validated", "medium",
                "per-camera policy selects validated calibrated density",
                False, False, detail)

        if density_count is not None and _disagrees(
                detector_count, density_count, self.config.disagreement_tolerance):
            return RouterOutput(
                mode, MODE_LABEL[mode], detector_count, density_count,
                detector_count, "detector", "low",
                "detector and density disagree beyond tolerance; review"
                " (detector count retained, marked low confidence)",
                True, True, detail)

        confidence = "high" if mode == SPARSE and detection.low_confidence_ratio() < 0.2 else "medium"
        return RouterOutput(
            mode, MODE_LABEL[mode], detector_count, density_count,
            detector_count, "detector", confidence,
            "box detector operating inside its validated range",
            False, False, detail)


def _severity(mode: str) -> int:
    return {SPARSE: 0, MODERATE: 1, DENSE: 2, UNKNOWN: 3}.get(mode, 0)


def _disagrees(detector_count: int, density_count: float, tolerance: float) -> bool:
    if detector_count <= 0:
        return False
    return abs(density_count - detector_count) / max(1.0, float(detector_count)) > tolerance
