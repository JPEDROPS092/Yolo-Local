"""O fluxo inteiro, com um detector falso, sem baixar o YOLO."""

import numpy as np

from obstaclesense.config import load_config
from obstaclesense.models import Detection
from obstaclesense.pipeline import ObstaclePipeline, limit_frame


class FakeDetector:
    def __init__(self, detections):
        self._detections = detections
        self.model_path = "falso.pt"
        self.device = "cpu"

    def detect(self, _frame):
        return list(self._detections)


def _config():
    config = load_config()
    config["audio_enabled"] = False
    config["depth_enabled"] = False
    config["minimum_object_size"] = 0.05
    config["hfov_degrees"] = 90
    config["vfov_degrees"] = 60
    return config


def test_limit_frame_keeps_aspect_ratio():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    small = limit_frame(frame, 640, 480)
    assert small.shape[1] <= 640
    assert small.shape[0] <= 480
    assert abs((small.shape[1] / small.shape[0]) - (1920 / 1080)) < 0.02
    same = limit_frame(np.zeros((400, 300, 3), dtype=np.uint8), 640, 480)
    assert same.shape[:2] == (400, 300)


def test_centered_person_becomes_portuguese_guidance():
    # Pés fora do quadro: a largura de 100 px com campo de 90° dá cerca de 1,6 m.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FakeDetector([Detection("person", 0.91, (270, 20, 370, 479))])
    pipeline = ObstaclePipeline(_config(), detector=detector)
    result = pipeline.process(frame, speak=False, annotate=True)
    payload = result.to_dict()
    obstacle = payload["objects"][0]
    assert obstacle["label"] == "person"
    assert obstacle["direction"] == "center"
    assert obstacle["danger_level"] == "high"
    assert obstacle["distance_meters"] < 2.5
    assert set(obstacle) == {"label", "confidence", "distance_meters", "direction", "danger_level"}
    assert "pessoa" in payload["guidance"]
    assert "à sua frente" in payload["guidance"]
    assert "Desvie para a direita." in payload["guidance"]
    assert "person" not in payload["guidance"].lower()
    assert result.annotated is not None
    assert result.annotated.shape == frame.shape
    assert result.latency_ms >= 0
    pipeline.close()


def test_tiny_box_is_ignored_and_growing_box_is_approaching():
    config = _config()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tiny = FakeDetector([Detection("cup", 0.99, (300, 220, 310, 230))])
    empty = ObstaclePipeline(config, detector=tiny).process(frame, speak=False, annotate=False)
    assert empty.objects == []
    assert empty.guidance == "Nenhum obstáculo à frente."

    boxes = [
        (250, 160, 390, 320),
        (240, 120, 400, 380),
        (220, 60, 420, 450),
    ]
    pipeline = ObstaclePipeline(config, detector=FakeDetector([]))
    last = None
    for box in boxes:
        pipeline.detector = FakeDetector([Detection("car", 0.88, box)])
        last = pipeline.process(frame, speak=False, annotate=False)
    assert last.objects[0].label == "car"
    assert last.objects[0].approaching is True
    assert "aproximando" in last.guidance
    assert "carro" in last.guidance
    pipeline.close()
