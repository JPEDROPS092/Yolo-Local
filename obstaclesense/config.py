"""Leitura do config.yaml. As chaves de áudio e de segurança são usadas de fato."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from obstaclesense.classes import DEFAULT_OBSTACLE_CLASSES

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps_target": 15,
    "yolo_model_path": "yolo11n.pt",
    "enable_gpu": False,
    "imgsz": 320,
    "confidence_threshold": 0.45,
    "iou_threshold": 0.45,
    "max_det": 30,
    "batch_size": 1,
    "obstacle_classes": list(DEFAULT_OBSTACLE_CLASSES),
    "audio_interval": 2.0,
    "audio_rate": 175,
    "volume_level": 0.8,
    "audio_enabled": True,
    "tts_engine": "auto",
    "piper_model_path": "",
    "collision_warning_threshold": 0.7,
    "proximity_warning_distance": 2.0,
    "minimum_object_size": 0.1,
    "hfov_degrees": 66.0,
    "vfov_degrees": 0.0,
    "camera_height_m": 1.2,
    "camera_pitch_deg": 0.0,
    "center_margin": 0.15,
    "depth_enabled": False,
    "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_every_n_frames": 4,
    "tracker_iou": 0.3,
    "approach_drop_m": 0.35,
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "window_width": 1280,
    "window_height": 920,
    "video_width": 800,
    "video_height": 600,
    "video_loop": False,
    "display_width": 800,
}


def config_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Mescla o arquivo sobre os padrões. Chaves ausentes continuam válidas."""
    cfg = dict(DEFAULTS)
    file_path = config_path(path)
    if file_path.exists():
        with open(file_path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuração inválida em {file_path}")
        for key, value in loaded.items():
            if value is not None:
                cfg[key] = value
    return normalize_config(cfg)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Garante faixas seguras sem apagar escolhas explícitas do arquivo."""
    out = dict(cfg)
    out["confidence_threshold"] = _clamp(float(out.get("confidence_threshold", 0.45)), 0.01, 1.0)
    out["iou_threshold"] = _clamp(float(out.get("iou_threshold", 0.45)), 0.01, 1.0)
    out["max_det"] = max(1, int(out.get("max_det", 30)))
    out["imgsz"] = max(32, int(out.get("imgsz", 320)))
    out["audio_interval"] = max(0.0, float(out.get("audio_interval", 2.0)))
    out["audio_rate"] = int(out.get("audio_rate", 175))
    out["volume_level"] = _clamp(float(out.get("volume_level", 0.8)), 0.0, 1.0)
    out["collision_warning_threshold"] = _clamp(
        float(out.get("collision_warning_threshold", 0.7)), 0.05, 1.0
    )
    out["proximity_warning_distance"] = max(0.2, float(out.get("proximity_warning_distance", 2.0)))
    out["minimum_object_size"] = _clamp(float(out.get("minimum_object_size", 0.1)), 0.0, 1.0)
    out["center_margin"] = _clamp(float(out.get("center_margin", 0.15)), 0.0, 0.45)
    out["hfov_degrees"] = _clamp(float(out.get("hfov_degrees", 66.0)), 20.0, 140.0)
    out["camera_height_m"] = max(0.2, float(out.get("camera_height_m", 1.2)))
    out["depth_every_n_frames"] = max(1, int(out.get("depth_every_n_frames", 4)))
    out["api_port"] = int(out.get("api_port", 8000))
    classes = out.get("obstacle_classes") or list(DEFAULT_OBSTACLE_CLASSES)
    out["obstacle_classes"] = [str(name) for name in classes]
    out["audio_enabled"] = bool(out.get("audio_enabled", True))
    out["enable_gpu"] = bool(out.get("enable_gpu", False))
    out["depth_enabled"] = bool(out.get("depth_enabled", False))
    out["yolo_model_path"] = str(out.get("yolo_model_path") or "yolo11n.pt")
    out["tts_engine"] = str(out.get("tts_engine") or "auto")
    out["piper_model_path"] = str(out.get("piper_model_path") or "")
    return out


def save_config(cfg: dict[str, Any], path: str | Path | None = None) -> Path:
    file_path = config_path(path)
    with open(file_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(normalize_config(cfg), handle, sort_keys=False, allow_unicode=True)
    return file_path


def resolve_device(enable_gpu: bool) -> str:
    if not enable_gpu:
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
