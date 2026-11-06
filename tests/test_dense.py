import numpy as np
import pytest

from app import adapters, dense


def test_point_adapter_maps_row_col_to_xy_and_zones(monkeypatch):
    # Service returns (row, column); adapter must expose Godavari (x, y).
    monkeypatch.setattr(dense, "count_points",
                        lambda frame: {"points": np.asarray([[10.0, 20.0],
                                                             [30.0, 40.0]]),
                                       "raw_count": 2, "count": 2.0})
    estimator = adapters.DensePointEstimator()
    zones = [{"id": "gate", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    result = estimator.estimate(np.zeros((100, 100, 3), np.uint8), zones)
    assert result.raw_count == 2
    assert result.points.tolist() == [[20.0, 10.0], [40.0, 30.0]]
    assert result.zone_counts["gate"] == 2
    assert result.confidence == "advisory"
    assert result.model_name == "dense-pet"


def test_point_adapter_validated_confidence(monkeypatch):
    monkeypatch.setattr(dense, "count_points",
                        lambda frame: {"points": np.zeros((0, 2)),
                                       "raw_count": 0, "count": 0.0})
    estimator = adapters.DensePointEstimator(validated=True)
    result = estimator.estimate(np.zeros((32, 32, 3), np.uint8))
    assert result.confidence == "validated"
    assert result.corrected_count is None


def test_point_adapter_keeps_real_correction(monkeypatch):
    monkeypatch.setattr(dense, "count_points",
                        lambda frame: {"points": np.asarray([[1.0, 2.0]]),
                                       "raw_count": 1, "count": 1.4})
    result = adapters.DensePointEstimator().estimate(
        np.zeros((16, 16, 3), np.uint8))
    assert result.corrected_count == pytest.approx(1.4)


def test_density_adapter_reads_dense_map_and_integrates_zones(monkeypatch):
    density = np.ones((8, 8), dtype=np.float32) * 0.5
    monkeypatch.setattr(dense, "estimate_density",
                        lambda frame: {"count": float(density.sum()),
                                       "density": density, "scale": 1.0})
    estimator = adapters.DenseDensityEstimator()
    zones = [{"id": "a", "polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]}]
    result = estimator.estimate(np.zeros((64, 64, 3), np.uint8), zones)
    assert result.total_count == pytest.approx(32.0)
    assert result.zone_counts["a"] == pytest.approx(12.5)
    assert result.model_name == "dense-dm-count"
    assert result.confidence == "advisory"


def test_verify_rejects_bad_checksum(tmp_path):
    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dense.verify(checkpoint, "0" * 64)


def test_verify_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dense.verify(tmp_path / "absent.pth", "0" * 64)


def test_available_reports_missing_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(dense, "PET_WEIGHTS", tmp_path / "pet_shab.pth")
    monkeypatch.setattr(dense, "DMCOUNT_WEIGHTS", tmp_path / "dmcount_shab.pth")
    ok, reason = dense.available()
    assert ok is False
    assert reason and "not found" in reason
