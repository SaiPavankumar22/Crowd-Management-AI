"""Confidence and mode router."""
from __future__ import annotations

from dataclasses import dataclass, field

from .adapters import DensityResult, DetectionResult, PointResult

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
    # Two dense estimators agreeing this closely is corroboration, not proof:
    # it earns a consensus estimate, never promotion to a validated count.
    consensus_tolerance: float = 0.25

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
    # Independent head-point estimate.  It is never added to the detector or
    # density count; the dashboard needs to see disagreement, not a blend.
    head_point_count: float | None = None
    # Mean of two mutually-agreeing dense estimators.  Displayed as ≈N and
    # still advisory: selected_count stays null until a validation is signed.
    consensus_count: float | None = None
    estimator_spread_pct: float | None = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "estimation_mode": self.estimation_mode,
            "detector_count": self.detector_count,
            "density_count": (round(self.density_count, 1)
                              if self.density_count is not None else None),
            "head_point_count": (round(self.head_point_count, 1)
                                 if self.head_point_count is not None else None),
            "selected_count": self.selected_count,
            "count_source": self.count_source,
            "confidence": self.confidence,
            "reason": self.reason,
            "degraded": self.degraded,
            "review_required": self.review_required,
            "consensus_count": (round(self.consensus_count, 1)
                                if self.consensus_count is not None else None),
            "estimator_spread_pct": self.estimator_spread_pct,
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
              points: PointResult | None = None, point_error: str | None = None,
              camera_health: str | None = None,
              calibrated_density_enabled: bool = False,
              calibrated_count_source: str = "none") -> RouterOutput:
        candidate, reason = self._classify(detection, track_continuity, camera_health)
        mode = self._apply_hysteresis(candidate)
        detector_count = detection.count
        density_count = density.total_count if density is not None else None
        head_point_count = points.total_count if points is not None else None

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
                True, True, detail, head_point_count)

        if mode == DENSE:
            # Two independent dense estimators that agree are stronger evidence
            # than either alone, so the operator gets a consensus estimate (≈N)
            # instead of the detector floor.  It stays advisory: promotion to
            # selected_count still needs a camera validation signed off.
            consensus = _consensus(head_point_count, density_count,
                                   self.config.consensus_tolerance)
            if consensus is not None:
                spread = abs(head_point_count - density_count) / max(consensus, 1.0)
                return RouterOutput(
                    DENSE, "dense_consensus", detector_count, density_count,
                    None, "dense_consensus", "medium",
                    (f"PET {head_point_count:.0f} and DM-Count {density_count:.0f} "
                     f"agree within {spread * 100:.0f}%: about {consensus:.0f} people; "
                     f"body detector saw {detector_count} (operator review required)"),
                    True, True, detail, head_point_count,
                    consensus_count=round(consensus, 1),
                    estimator_spread_pct=round(spread * 100, 1))
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
                    True, True, detail, head_point_count)
            selected_source = str(calibrated_count_source or "none")
            if (selected_source == "head_point" and points is not None
                    and points.confidence == "validated"):
                return RouterOutput(
                    DENSE, "calibrated_head_points", detector_count, density_count,
                    int(round(head_point_count)), "head_points_validated", "medium",
                    "per-camera policy selects validated head-point estimate",
                    False, False, detail, head_point_count)
            if (selected_source == "density" and density is not None
                    and density.confidence == "validated"):
                return RouterOutput(
                    DENSE, "calibrated_density", detector_count, density_count,
                    int(round(density_count)), "density_validated", "medium",
                    "per-camera policy selects validated density estimate",
                    False, False, detail, head_point_count)
            if head_point_count is not None:
                why = "high_occlusion; head-point estimate advisory pending validation"
            elif density_count is not None:
                why = "high_occlusion; density estimate advisory pending validation"
            else:
                why = ("high_occlusion; dense estimators unavailable"
                       if density_error or point_error else "high_occlusion")
            return RouterOutput(
                DENSE, MODE_LABEL[DENSE], detector_count, density_count,
                None, "none", "low",
                f"{why} (operator review required)",
                True, True, detail, head_point_count)

        # SPARSE / MODERATE: body boxes remain the source of record.  Dense
        # estimators cross-check it, but never replace a working tracker.
        if density_count is not None and _disagrees(
                detector_count, density_count, self.config.disagreement_tolerance):
            return RouterOutput(
                mode, MODE_LABEL[mode], detector_count, density_count,
                detector_count, "detector", "low",
                "detector and density disagree beyond tolerance; review"
                " (detector count retained, marked low confidence)",
                True, True, detail, head_point_count)

        confidence = "high" if mode == SPARSE and detection.low_confidence_ratio() < 0.2 else "medium"
        return RouterOutput(
            mode, MODE_LABEL[mode], detector_count, density_count,
            detector_count, "detector", confidence,
            "box detector operating inside its validated range",
            False, False, detail, head_point_count)

def _severity(mode: str) -> int:
    return {SPARSE: 0, MODERATE: 1, DENSE: 2, UNKNOWN: 3}.get(mode, 0)

def _consensus(head_points: float | None, density: float | None,
               tolerance: float) -> float | None:
    """Mean of two independent dense estimates when they corroborate each other."""
    if head_points is None or density is None:
        return None
    if head_points <= 0 or density <= 0:
        return None
    spread = abs(head_points - density) / max(head_points, density)
    if spread > tolerance:
        return None
    return (float(head_points) + float(density)) / 2.0

def _disagrees(detector_count: int, density_count: float, tolerance: float) -> bool:
    if detector_count <= 0:
        return False
    return abs(density_count - detector_count) / max(1.0, float(detector_count)) > tolerance
