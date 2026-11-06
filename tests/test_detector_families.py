"""Model families: registry, availability, adapter choice and manager catalogue."""
import unittest
from unittest import mock

import numpy as np

from app import adapters
from app.adapters import (KNOWN_MODELS, RFDetrDetector, RTDETRDetector,
                          UltralyticsDetector, build_detector, model_available,
                          spec_for)


class RegistryTests(unittest.TestCase):
    def test_non_yolo_families_are_registered_under_commercial_licences(self):
        for name in ("rtdetr-l.pt", "rf-detr-base.pth", "dfine_m_coco.pt"):
            self.assertIn(name, KNOWN_MODELS)
            self.assertEqual(spec_for(name).license, "Apache-2.0")

    def test_availability_follows_the_optional_package(self):
        with mock.patch("app.adapters.importlib.util.find_spec", return_value=None):
            self.assertFalse(model_available("rf-detr-base.pth"))
            self.assertFalse(model_available("dfine_m_coco.pt"))
            self.assertTrue(model_available("rtdetr-l.pt"))

    def test_factory_maps_families_onto_adapters(self):
        self.assertIsInstance(build_detector("rtdetr-l.pt"), RTDETRDetector)
        self.assertIsInstance(build_detector("rf-detr-base.pth"), RFDetrDetector)
        self.assertIsInstance(build_detector("yolo26n.pt"), UltralyticsDetector)
        self.assertEqual(build_detector("rtdetr-x.pt").backend, "rtdetr")

    def test_missing_package_names_the_install_step(self):
        with mock.patch.dict("sys.modules", {"rfdetr": None}):
            with self.assertRaises(RuntimeError) as ctx:
                RFDetrDetector("rf-detr-base.pth").load()
        self.assertIn("pip install rfdetr", str(ctx.exception))

        with mock.patch.dict("sys.modules", {"dfine": None}):
            with self.assertRaises(RuntimeError) as ctx:
                build_detector("dfine_m_coco.pt").load()
        self.assertIn("dfine", str(ctx.exception))


class DetectionConversionTests(unittest.TestCase):
    def test_detr_detections_keep_only_people(self):
        class Det:
            xyxy = np.array([[0, 0, 10, 20], [5, 5, 30, 40]], np.float32)
            confidence = np.array([0.9, 0.8], np.float32)
            class_id = np.array([0, 2], np.int32)

        out = adapters._convert_detections(Det(), np.zeros((100, 100, 3), np.uint8),
                                           5.0, "m", "v")
        self.assertEqual(out.count, 1)
        self.assertEqual(out.image_size, (100, 100))
        self.assertEqual(out.model_name, "m")


class ManagerCatalogueTests(unittest.TestCase):
    def test_default_model_is_the_dense_safe_one_on_every_device(self):
        """RT-DETR wins on medium density but misses packed frames; the default
        stays YOLO26n until its escalation trigger is tuned."""
        from app.detector import ModelManager

        for cuda in (True, False):
            with mock.patch("torch.cuda.is_available", return_value=cuda):
                self.assertEqual(ModelManager("", [], 0.30, 640).requested[0],
                                 "yolo26n.pt")

    def test_every_family_is_switchable_and_reports_availability(self):
        from app.detector import ModelManager

        manager = ModelManager("yolo26n.pt", [], 0.30, 640)
        models = {m["name"]: m for m in manager.models()}
        self.assertIn("rtdetr-l.pt", models)
        self.assertIn("rf-detr-base.pth", models)
        self.assertTrue(models["rtdetr-l.pt"]["available"])
        self.assertIn("requires", models["rf-detr-base.pth"])
