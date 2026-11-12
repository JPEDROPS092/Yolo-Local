"""Liga detecção, distância, risco, rastreamento, frase e voz."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from obstaclesense.config import load_config, normalize_config
from obstaclesense.danger import assess_danger, sort_obstacles
from obstaclesense.direction import box_center_x, classify_direction
from obstaclesense.distance import estimate_bbox_distance, fit_depth_scale, fuse_distance
from obstaclesense.models import FrameResult, Obstacle
from obstaclesense.overlay import draw_overlay
from obstaclesense.phrases import PhraseObstacle, build_guidance
from obstaclesense.tracking import SimpleTracker

logger = logging.getLogger(__name__)


class ObstaclePipeline:
    def __init__(self, config: dict | None = None, detector=None, speaker=None):
        self.config = normalize_config(config) if config is not None else load_config()
        if detector is None:
            from obstaclesense.detector import YoloDetector

            detector = YoloDetector(self.config)
        self.detector = detector
        self.speaker = speaker
        if self.speaker is None and self.config.get("audio_enabled", True):
            from obstaclesense.speech import Speaker

            self.speaker = Speaker(
                engine=self.config.get("tts_engine", "auto"),
                rate=int(self.config.get("audio_rate", 175)),
                volume=float(self.config.get("volume_level", 0.8)),
                interval_s=float(self.config.get("audio_interval", 2.0)),
                piper_model=self.config.get("piper_model_path", ""),
                enabled=True,
            )
        self.tracker = SimpleTracker(
            iou_threshold=float(self.config.get("tracker_iou", 0.3)),
            approach_drop_m=float(self.config.get("approach_drop_m", 0.35)),
        )
        self.depth = None
        self._depth_cache = None
        self._frame_index = 0
        self._fps = 0.0
        self.depth_enabled = bool(self.config.get("depth_enabled", False))
        if self.depth_enabled:
            self._enable_depth()

    @property
    def model_name(self) -> str:
        return str(getattr(self.detector, "model_path", self.config.get("yolo_model_path", "")))

    @property
    def device(self) -> str:
        return str(getattr(self.detector, "device", "cpu"))

    @property
    def class_names(self) -> list[str]:
        return list(self.config.get("obstacle_classes", []))

    def process(self, frame: np.ndarray, speak: bool = True, annotate: bool = True) -> FrameResult:
        started = time.perf_counter()
        frame = limit_frame(
            frame,
            int(self.config.get("frame_width", 640) or 640),
            int(self.config.get("frame_height", 480) or 480),
        )
        height, width = frame.shape[:2]
        detections = self.detector.detect(frame)
        depth_map = self._depth_for_frame(frame)

        prelim = []
        min_size = float(self.config.get("minimum_object_size", 0.0))
        for detection in detections:
            if not _passes_min_size(detection.box, width, height, min_size):
                continue
            bbox_m = estimate_bbox_distance(
                detection.box,
                width,
                height,
                detection.label,
                hfov_degrees=float(self.config.get("hfov_degrees", 66.0)),
                vfov_degrees=float(self.config.get("vfov_degrees", 0.0)),
                camera_height_m=float(self.config.get("camera_height_m", 1.2)),
                pitch_deg=float(self.config.get("camera_pitch_deg", 0.0)),
            )
            relative = None
            if depth_map is not None:
                from obstaclesense.depth import sample_relative_depth

                relative = sample_relative_depth(depth_map, detection.box)
            direction = classify_direction(
                box_center_x(detection.box),
                width,
                float(self.config.get("center_margin", 0.15)),
            )
            prelim.append((detection, bbox_m, relative, direction))

        calibration = None
        if self.depth_enabled:
            pairs = [(rel, bbox_m) for _det, bbox_m, rel, _direction in prelim if rel is not None]
            calibration = fit_depth_scale(pairs)

        obstacles: list[Obstacle] = []
        for detection, bbox_m, relative, direction in prelim:
            distance = fuse_distance(bbox_m, relative, calibration) if self.depth_enabled else bbox_m
            obstacles.append(
                Obstacle(
                    label=detection.label,
                    confidence=detection.confidence,
                    distance_meters=float(distance),
                    direction=direction,
                    danger_level="low",
                    box=detection.box,
                )
            )

        obstacles = self.tracker.update(obstacles, now=time.monotonic())
        proximity = float(self.config.get("proximity_warning_distance", 2.0))
        threshold = float(self.config.get("collision_warning_threshold", 0.7))
        for obstacle in obstacles:
            level, score = assess_danger(
                obstacle.distance_meters,
                obstacle.direction,
                obstacle.label,
                approaching=obstacle.approaching,
                proximity_m=proximity,
                collision_threshold=threshold,
            )
            obstacle.danger_level = level
            obstacle.danger_score = score
        obstacles = sort_obstacles(obstacles)

        phrase_items = [
            PhraseObstacle(
                label=item.label,
                distance_m=item.distance_meters,
                direction=item.direction,
                danger_level=item.danger_level,
                approaching=item.approaching,
            )
            for item in obstacles
        ]
        guidance = build_guidance(phrase_items, proximity_m=proximity)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._fps = _smooth_fps(self._fps, latency_ms)
        annotated = draw_overlay(frame, obstacles, guidance, self._fps, latency_ms) if annotate else None
        if speak and obstacles and self.speaker is not None:
            self.speaker.say(guidance, obstacles[0].danger_level)
        self._frame_index += 1
        return FrameResult(
            objects=obstacles,
            guidance=guidance,
            latency_ms=latency_ms,
            fps=self._fps,
            annotated=annotated,
        )

    def close(self) -> None:
        if self.speaker is not None and hasattr(self.speaker, "close"):
            self.speaker.close()

    def _enable_depth(self) -> None:
        try:
            from obstaclesense.depth import DepthAnythingEstimator

            self.depth = DepthAnythingEstimator(
                self.config.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf"),
                device=self.device,
            )
        except Exception:
            logger.exception("Profundidade indisponível. Seguindo só com a estimativa geométrica.")
            self.depth_enabled = False
            self.depth = None

    def _depth_for_frame(self, frame: np.ndarray):
        if not self.depth_enabled or self.depth is None:
            return None
        every = int(self.config.get("depth_every_n_frames", 4))
        if self._depth_cache is None or self._frame_index % every == 0:
            try:
                self._depth_cache = self.depth.estimate(frame)
            except Exception:
                logger.exception("Falha na profundidade neste quadro.")
                self._depth_cache = None
        return self._depth_cache


def limit_frame(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Reduz o quadro mantendo a proporção, para a CPU não receber foto de 12 MP."""
    if frame is None or max_width <= 0 or max_height <= 0:
        return frame
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))


def _passes_min_size(box, width: int, height: int, min_ratio: float) -> bool:
    if min_ratio <= 0 or width <= 0 or height <= 0:
        return True
    box_w = max(0, box[2] - box[0]) / width
    box_h = max(0, box[3] - box[1]) / height
    return max(box_w, box_h) >= min_ratio


def _smooth_fps(previous: float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return previous
    instant = 1000.0 / latency_ms
    if previous <= 0:
        return instant
    return 0.8 * previous + 0.2 * instant
