"""O config.yaml deixa de ser decorativo: áudio e segurança entram no programa."""

from obstaclesense.config import load_config
from obstaclesense.danger import assess_danger
from obstaclesense.speech import SpeechGate


def test_audio_and_safety_keys_are_loaded():
    config = load_config()
    assert config["audio_interval"] == 2.0
    assert config["audio_rate"] == 175
    assert config["volume_level"] == 0.8
    assert config["proximity_warning_distance"] == 2.0
    assert config["collision_warning_threshold"] == 0.7
    assert config["minimum_object_size"] == 0.1
    assert config["yolo_model_path"] == "yolo11n.pt"
    assert config["enable_gpu"] is False
    assert "person" in config["obstacle_classes"]
    assert "car" in config["obstacle_classes"]
    assert config["depth_enabled"] is False


def test_loaded_values_change_behavior():
    config = load_config()
    gate = SpeechGate(config["audio_interval"])
    gate.mark(0.0, "low")
    assert gate.allow(config["audio_interval"] - 0.1, "low") is False
    assert gate.allow(config["audio_interval"], "low") is True

    level, _score = assess_danger(
        config["proximity_warning_distance"],
        "center",
        "person",
        proximity_m=config["proximity_warning_distance"],
        collision_threshold=config["collision_warning_threshold"],
    )
    assert level == "high"
