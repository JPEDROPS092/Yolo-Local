"""Contrato do POST /analyze e do websocket, sem carregar a rede YOLO."""

import cv2
import numpy as np
from fastapi.testclient import TestClient

from obstaclesense.api import create_app
from obstaclesense.models import Detection
from obstaclesense.pipeline import ObstaclePipeline
from tests.test_pipeline import FakeDetector, _config


def _client():
    frame_shape = (480, 640, 3)
    detector = FakeDetector([Detection("person", 0.87, (270, 20, 370, 479))])
    pipeline = ObstaclePipeline(_config(), detector=detector)
    pipeline._blank = frame_shape
    return TestClient(create_app(pipeline=pipeline)), pipeline


def _jpeg() -> bytes:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_analyze_returns_the_android_fields():
    client, pipeline = _client()
    response = client.post("/analyze", files={"file": ("frame.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert "pessoa" in body["guidance"]
    assert "à sua frente" in body["guidance"]
    obstacle = body["objects"][0]
    assert set(obstacle) == {"label", "confidence", "distance_meters", "direction", "danger_level"}
    assert obstacle["label"] == "person"
    assert obstacle["direction"] in {"left", "center", "right"}
    assert obstacle["danger_level"] in {"low", "medium", "high"}
    assert "latency_ms" in body
    assert "fps" in body
    pipeline.close()


def test_invalid_image_is_rejected():
    client, pipeline = _client()
    response = client.post("/analyze", files={"file": ("x.bin", b"nao-e-imagem", "application/octet-stream")})
    assert response.status_code == 400
    pipeline.close()


def test_health_and_websocket():
    client, pipeline = _client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    with client.websocket_connect("/ws") as socket:
        socket.send_bytes(_jpeg())
        payload = socket.receive_json()
    assert payload["objects"][0]["label"] == "person"
    assert "guidance" in payload
    pipeline.close()
