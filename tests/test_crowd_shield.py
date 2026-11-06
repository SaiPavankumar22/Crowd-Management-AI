"""Crowd Shield adapters: contract mapping, zone integration, checksum refusal."""
import numpy as np
import pytest

from app import adapters, crowd_shield

ZONE_ALL = {"id": "z1", "name": "Walkway",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}


def test_point_adapter_maps_points_and_zones(monkeypatch):
    points_row_col = np.array([[10, 20], [30, 40], [70, 80]], dtype=np.float32)
    monkeypatch.setattr(crowd_shield, "count_points",
                        lambda frame: {"count": 3.0, "raw_count": 3,
                                       "points": points_row_col})
    estimator = adapters.CrowdShieldPointEstimator()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = estimator.estimate(frame, [ZONE_ALL])

    assert result.raw_count == 3
    assert result.points.shape == (3, 2)
    # Crowd Shield emits (row, column); the contract is (x, y).
    assert tuple(result.points[0]) == (20.0, 10.0)
    assert result.zone_counts["z1"] == 3
    assert result.confidence == "advisory"


def test_point_adapter_keeps_correction_separate_from_raw(monkeypatch):
    monkeypatch.setattr(crowd_shield, "count_points",
                        lambda frame: {"count": 13.9, "raw_count": 10,
                                       "points": np.zeros((10, 2), np.float32)})
    estimator = adapters.CrowdShieldPointEstimator(validated=True)
    result = estimator.estimate(np.zeros((50, 50, 3), np.uint8))

    assert result.raw_count == 10
    assert result.corrected_count == pytest.approx(13.9)
    assert result.total_count == pytest.approx(13.9)
    assert result.confidence == "validated"


def test_point_adapter_reports_no_correction_when_count_equals_raw(monkeypatch):
    monkeypatch.setattr(crowd_shield, "count_points",
                        lambda frame: {"count": 7.0, "raw_count": 7,
                                       "points": np.zeros((7, 2), np.float32)})
    result = adapters.CrowdShieldPointEstimator().estimate(
        np.zeros((20, 20, 3), np.uint8))
    assert result.corrected_count is None
    assert result.total_count == 7.0


def test_density_adapter_reads_crowdshield_map_and_integrates_zones(monkeypatch):
    density = np.zeros((8, 8), dtype=np.float32)
    density[:, :4] = 2.0  # left half of the frame: 32 cells x 2.0 = 64 people
    monkeypatch.setattr(crowd_shield, "estimate_density",
                        lambda frame: {"count": float(density.sum()),
                                       "density": density, "scale": 1.0})
    estimator = adapters.CrowdShieldDensityEstimator()
    left = {"id": "left", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}
    right = {"id": "right", "polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}
    result = estimator.estimate(np.zeros((64, 64, 3), np.uint8), [left, right])

    assert result.density_map is not None and result.density_map.shape == (8, 8)
    assert result.total_count == pytest.approx(64.0)
    assert result.zone_counts["left"] == pytest.approx(64.0)
    assert result.zone_counts["right"] == pytest.approx(0.0)


def test_checksum_mismatch_refuses_to_load(tmp_path):
    checkpoint = tmp_path / "swapped.pth"
    checkpoint.write_bytes(b"not the checkpoint that was evaluated")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        crowd_shield.verify(checkpoint, "0" * 64)


def test_missing_checkpoint_names_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        crowd_shield.verify(tmp_path / "absent.pth", "0" * 64)


def test_pinned_checkpoints_are_present_and_verified():
    ok, reason = crowd_shield.available()
    assert ok, reason
