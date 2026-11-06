"""District aggregation: signing, staleness and the merged totals."""
import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

from app import config
from app.main import app

client = TestClient(app)


def _post(payload: dict, secret: str = "district-secret", header: str | None = None):
    body = json.dumps(payload).encode()
    signature = header if header is not None else hmac.new(
        secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post("/api/district/ingest", content=body,
                       headers={"Content-Type": "application/json",
                                "X-Godavari-Signature": signature})


def _payload(node: str, count: int, level: str = "WARNING", cameras: int = 1) -> dict:
    return {"node": node, "ts": time.time(), "device": "gpu",
            "cameras": [{"id": f"{node}-{i}", "name": f"cam {i}", "count": count,
                         "level": level, "status": "running"}
                        for i in range(cameras)],
            "alerts": [{"id": f"{node}-a", "level": level, "ts": time.time()}]}


def test_ingest_requires_configured_secret(monkeypatch):
    monkeypatch.setattr(config, "DISTRICT_SECRET", "")
    assert _post(_payload("node-a", 10)).status_code == 503


def test_ingest_rejects_bad_signature(monkeypatch):
    monkeypatch.setattr(config, "DISTRICT_SECRET", "district-secret")
    assert _post(_payload("node-b", 5), header="nope").status_code == 401


def test_overview_merges_nodes_and_flags_stale(monkeypatch):
    monkeypatch.setattr(config, "DISTRICT_SECRET", "district-secret")
    monkeypatch.setattr(config, "DISTRICT_NODE_TTL_S", 60.0)
    assert _post(_payload("ghat-a", 40, cameras=2)).status_code == 200
    assert _post(_payload("ghat-b", 7, level="OVERCROWDED")).status_code == 200

    body = client.get("/api/district/overview").json()
    by_node = {row["node"]: row for row in body["nodes"]}
    assert by_node["ghat-a"]["people"] == 80
    assert by_node["ghat-a"]["cameras"] == 2
    assert by_node["ghat-b"]["overcrowded"] == 1
    assert body["totals"]["people"] >= 87
    assert body["stale_nodes"] == []

    # A node that stopped pushing is reported, not silently dropped.
    monkeypatch.setattr(config, "DISTRICT_NODE_TTL_S", -1.0)
    body = client.get("/api/district/overview").json()
    assert "ghat-a" in body["stale_nodes"]


def test_ingest_rejects_body_without_node(monkeypatch):
    monkeypatch.setattr(config, "DISTRICT_SECRET", "district-secret")
    assert _post({"cameras": []}).status_code == 400


def test_publisher_carries_a_valid_signature(monkeypatch):
    from app.district import DistrictPublisher, summary

    captured = {}

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = request.data
        captured["signature"] = request.headers["X-godavari-signature"]
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    publisher = DistrictPublisher(url="http://district.invalid/ingest",
                                  secret="district-secret", period_s=1.0)
    publisher.push_once()

    expected = hmac.new(b"district-secret", captured["body"], hashlib.sha256).hexdigest()
    assert captured["signature"] == expected
    assert json.loads(captured["body"])["node"] == summary()["node"]
