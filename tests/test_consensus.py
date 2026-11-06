"""Dense consensus: agreement earns an estimate, never a promotion."""
import numpy as np
import pytest

from app.adapters import DensityResult, DetectionResult, PointResult
from app.router import ModeRouter


def detection(count=148, primary=6):
    boxes = np.tile(np.array([[10.0, 10.0, 40.0, 90.0]]), (count, 1))
    return DetectionResult(boxes=boxes, confidences=np.full(count, 0.9),
                           class_ids=np.zeros(count, dtype=int), inference_ms=1.0,
                           model_name="yolo26n.pt", model_version="26",
                           image_size=(1280, 720), tiled=True, tile_count=9,
                           tier="detail", primary_count=primary)


def points(total, raw=10):
    return PointResult(points=np.zeros((raw, 2), np.float32), raw_count=raw,
                       corrected_count=total, zone_counts={}, confidence="advisory",
                       inference_ms=1.0, model_name="crowdshield-pet",
                       model_version="pet-shab-vgg16bn")


def density(total):
    return DensityResult(total_count=total, zone_counts={}, confidence="advisory",
                         inference_ms=1.0, model_name="crowdshield-dm-count",
                         model_version="dmcount-shab-vgg19")


def test_agreeing_estimators_give_consensus_without_promotion():
    router = ModeRouter()
    out = router.route(detection(), points=points(383.0), density=density(389.0))

    assert out.mode == "dense"
    assert out.estimation_mode == "dense_consensus"
    assert out.consensus_count == pytest.approx(386.0)
    assert out.selected_count is None          # no validation, no promotion
    assert out.degraded is True                # capacity alerts stay suppressed
    assert "agree" in out.reason


def test_disagreeing_estimators_fall_back_to_the_lower_bound():
    router = ModeRouter()
    out = router.route(detection(), points=points(30.0), density=density(420.0))

    assert out.estimation_mode == "detector_lower_bound"
    assert out.consensus_count is None
    assert out.selected_count is None


def test_one_estimator_is_not_a_consensus():
    router = ModeRouter()
    out = router.route(detection(), points=points(380.0), density=None)

    assert out.estimation_mode == "detector_lower_bound"
    assert out.consensus_count is None
