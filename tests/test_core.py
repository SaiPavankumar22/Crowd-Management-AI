import unittest
from app.calibration import validate_calibration

import numpy as np

from app.alerting import AlertEngine
from app.calibration import project_points, validate_calibration
from app.health import FrameHealth, measure_quality
from app.zones import Zone


class ZoneTests(unittest.TestCase):
    def test_occupancy_uses_smoothed_count_and_metric_density(self):
        zone = Zone("gate", [[0, 0], [1, 0], [1, 1]], capacity=10)
        zone.apply_count(5, 10000, area_m2=2)
        self.assertEqual(zone.count, 5)
        self.assertEqual(zone.density_m2, 2.5)
        zone.apply_count(10, 10000, area_m2=2)
        self.assertEqual(zone.count, 7)


class AlertTests(unittest.TestCase):
    def test_overcrowded_requires_sustain_window(self):
        zone = Zone("gate", [[0, 0], [1, 0], [1, 1]], capacity=1)
        zone.apply_count(1, 10000)
        engine = AlertEngine("cam", "Camera", sustain_s=2, cooldown_s=1)
        self.assertEqual(engine.update(zone, ts=10)[0]["level"], "WARNING")
        alerts = engine.update(zone, ts=12)
        self.assertEqual(alerts[0]["level"], "OVERCROWDED")


class HealthTests(unittest.TestCase):
    def test_quality_detects_dark_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(measure_quality(frame)["status"], "dark")

    def test_identical_frames_become_frozen(self):
        health = FrameHealth()
        frame = np.full((100, 100, 3), 120, dtype=np.uint8)
        health.frame_received(frame, now=10)
        health.frame_received(frame, now=21)
        self.assertEqual(health.snapshot(now=21)["status"], "frozen")


class CalibrationTests(unittest.TestCase):
    def test_projects_points(self):
        calibration = validate_calibration({
            "image_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "world_points": [[0, 0], [2, 0], [2, 2], [0, 2]],
        })
        result = project_points([[5, 5]], calibration)
        np.testing.assert_allclose(result[0], [1, 1], atol=0.01)

    def test_rejects_empty_and_non_finite_calibration(self):
        with self.assertRaises(ValueError):
            validate_calibration({})
        with self.assertRaises(ValueError):
            validate_calibration({
                "image_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "world_points": [[0, 0], [2, 0], [float("nan"), 2], [0, 2]],
            })


if __name__ == "__main__":
    unittest.main()
