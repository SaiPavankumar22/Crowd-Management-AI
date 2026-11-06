"""Optional API authentication: 401/403 behaviour and the public exceptions."""
import unittest

from fastapi.testclient import TestClient

from app import config
from app.main import app


class ApiAuthTests(unittest.TestCase):
    def setUp(self):
        self._op = config.AUTH_OPERATOR_TOKEN
        self._view = config.AUTH_VIEWER_TOKEN

    def tearDown(self):
        config.AUTH_OPERATOR_TOKEN = self._op
        config.AUTH_VIEWER_TOKEN = self._view

    def test_api_stays_open_when_no_token_is_configured(self):
        config.AUTH_OPERATOR_TOKEN = ""
        config.AUTH_VIEWER_TOKEN = ""
        self.assertEqual(TestClient(app).get("/api/models").status_code, 200)

    def test_reads_need_a_token_and_writes_need_the_operator_role(self):
        config.AUTH_OPERATOR_TOKEN = "op-secret"
        config.AUTH_VIEWER_TOKEN = "viewer-secret"
        client = TestClient(app)

        self.assertEqual(client.get("/api/models").status_code, 401)
        self.assertEqual(client.get("/api/models",
                                    headers={"x-api-key": "nope"}).status_code, 401)
        self.assertEqual(client.get("/api/models",
                                    headers={"x-api-key": "viewer-secret"}).status_code, 200)
        self.assertEqual(client.get("/api/models",
                                    headers={"authorization": "Bearer viewer-secret"}).status_code, 200)
        self.assertEqual(client.put("/api/models", json={"name": "yolo26n.pt"},
                                    headers={"x-api-key": "viewer-secret"}).status_code, 403)

    def test_health_and_static_stay_public(self):
        config.AUTH_OPERATOR_TOKEN = "op-secret"
        config.AUTH_VIEWER_TOKEN = ""
        client = TestClient(app)
        self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertEqual(client.get("/static/app.js").status_code, 200)
