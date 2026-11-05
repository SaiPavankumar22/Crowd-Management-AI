import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config, store
from app.mongo_sync import mirror_camera
from app.pipeline import CameraRegistry


class _Model:
    def __init__(self):
        self.profiles = []

    def set_camera_profile(self, camera_id, profile):
        self.profiles.append((camera_id, profile))


class _Worker:
    def __init__(self):
        self.stopped = False
        self.joined = False

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True


class CameraManagementTests(unittest.TestCase):
    def test_remove_stops_worker_and_persists_camera_removal(self):
        worker = _Worker()
        model = _Model()
        registry = object.__new__(CameraRegistry)
        registry.model = model
        registry._lock = threading.Lock()
        registry.cameras = {
            "cam-1": {"id": "cam-1", "name": "Lobby", "profile": "dense"},
            "cam-2": {"id": "cam-2", "name": "Gate"},
        }
        registry.workers = {"cam-1": worker}

        with tempfile.TemporaryDirectory(dir=".tmp") as tmp, \
                patch.object(config, "DATA_DIR", tmp), \
                patch.object(store, "add_audit"), \
                patch("app.mongo_sync.mirror.delete") as mirror_delete:
            registry.remove("cam-1")
            saved = json.loads(Path(tmp, "cameras.json").read_text(encoding="utf-8"))

        self.assertTrue(worker.stopped)
        self.assertTrue(worker.joined)
        self.assertEqual(model.profiles, [("cam-1", "global")])
        self.assertEqual([camera["id"] for camera in saved], ["cam-2"])
        mirror_delete.assert_called_once_with("cameras", "cam-1")

    def test_remove_deletes_uploaded_video_only_when_no_camera_uses_it(self):
        with tempfile.TemporaryDirectory(dir=".tmp") as tmp:
            uploads = Path(tmp, "uploads")
            uploads.mkdir()
            data_dir = Path(tmp, "data")
            data_dir.mkdir()
            video = uploads / "clip.mp4"
            video.write_bytes(b"video")
            registry = object.__new__(CameraRegistry)
            registry.model = _Model()
            registry._lock = threading.Lock()
            registry.cameras = {"cam-1": {"id": "cam-1", "name": "Clip",
                                           "source_type": "file", "path": str(video)}}
            registry.workers = {}
            with patch.object(config, "DATA_DIR", str(Path(tmp, "data"))), \
                    patch.object(config, "UPLOADS_DIR", str(uploads)), \
                    patch.object(store, "add_audit") as add_audit, \
                    patch("app.mongo_sync.mirror.delete"):
                registry.remove("cam-1")
            self.assertFalse(video.exists())
            self.assertTrue(add_audit.call_args.args[3]["uploaded_file_deleted"])

    def test_remove_keeps_uploaded_video_if_another_camera_uses_it(self):
        with tempfile.TemporaryDirectory(dir=".tmp") as tmp:
            uploads = Path(tmp, "uploads")
            uploads.mkdir()
            data_dir = Path(tmp, "data")
            data_dir.mkdir()
            video = uploads / "clip.mp4"
            video.write_bytes(b"video")
            registry = object.__new__(CameraRegistry)
            registry.model = _Model()
            registry._lock = threading.Lock()
            registry.cameras = {
                "cam-1": {"id": "cam-1", "name": "Clip 1", "source_type": "file", "path": str(video)},
                "cam-2": {"id": "cam-2", "name": "Clip 2", "source_type": "file", "path": str(video)},
            }
            registry.workers = {}
            with patch.object(config, "DATA_DIR", str(Path(tmp, "data"))), \
                    patch.object(config, "UPLOADS_DIR", str(uploads)), \
                    patch.object(store, "add_audit"), \
                    patch("app.mongo_sync.mirror.delete"):
                registry.remove("cam-1")
            self.assertTrue(video.exists())

    def test_camera_archive_omits_local_source_path(self):
        camera = {"id": "cam-1", "name": "rtsp://operator:secret@example.invalid/stream", "source_type": "rtsp",
                  "path": "rtsp://operator:secret@example.invalid/stream",
                  "zones": [], "profile": "balanced"}
        with patch("app.mongo_sync.mirror.replace") as replace:
            mirror_camera(camera)
        document = replace.call_args.args[1]
        self.assertEqual(document["_id"], "cam-1")
        self.assertNotIn("path", document)
        self.assertNotIn("secret", json.dumps(document))


if __name__ == "__main__":
    unittest.main()
