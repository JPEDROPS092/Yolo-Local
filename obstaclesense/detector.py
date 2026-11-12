"""Detector YOLO11, com YOLOv8 ainda disponível pelo caminho do modelo."""

from __future__ import annotations

import logging
import threading

import numpy as np

from obstaclesense.config import resolve_device
from obstaclesense.models import Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    def __init__(self, config: dict):
        self.model_path = str(config.get("yolo_model_path") or "yolo11n.pt")
        self.device = resolve_device(bool(config.get("enable_gpu", False)))
        if config.get("device"):
            self.device = str(config["device"])
        self.conf = float(config.get("confidence_threshold", 0.45))
        self.iou = float(config.get("iou_threshold", 0.45))
        self.max_det = int(config.get("max_det", 30))
        self.imgsz = int(config.get("imgsz", 320))
        self.class_names = [str(name) for name in config.get("obstacle_classes", [])]
        self._lock = threading.Lock()
        self.model = None
        self.class_ids: list[int] | None = None
        self._load()

    def _load(self) -> None:
        from ultralytics import YOLO

        logger.info("Carregando %s em %s (imgsz=%s).", self.model_path, self.device, self.imgsz)
        self.model = YOLO(self.model_path)
        names = self.model.names
        if isinstance(names, dict):
            name_to_id = {str(value).lower(): int(key) for key, value in names.items()}
        else:
            name_to_id = {str(value).lower(): index for index, value in enumerate(names)}
        if self.class_names:
            wanted = []
            missing = []
            for name in self.class_names:
                key = name.lower()
                if key in name_to_id:
                    wanted.append(name_to_id[key])
                else:
                    missing.append(name)
            self.class_ids = wanted or None
            if missing:
                logger.warning("Classes fora do modelo, ignoradas: %s", ", ".join(missing))
        else:
            self.class_ids = None
        warmup = np.zeros((max(64, self.imgsz), max(64, self.imgsz), 3), dtype=np.uint8)
        self.detect(warmup)
        logger.info("Modelo pronto. Classes ativas: %s", len(self.class_ids or []))

    def set_thresholds(
        self,
        confidence: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
    ) -> None:
        if confidence is not None:
            self.conf = float(confidence)
        if iou is not None:
            self.iou = float(iou)
        if max_det is not None:
            self.max_det = int(max_det)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or self.model is None:
            return []
        with self._lock:
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                device=self.device,
                classes=self.class_ids,
                verbose=False,
            )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        names = self.model.names
        found: list[Detection] = []
        for index in range(len(xyxy)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[index])
            class_id = int(classes[index])
            label = names[class_id] if isinstance(names, dict) else names[class_id]
            found.append(
                Detection(
                    label=str(label),
                    confidence=float(scores[index]),
                    box=(x1, y1, x2, y2),
                )
            )
        return found
