"""Exporta o modelo YOLO para ONNX ou TFLite, para uso no aparelho."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_model(model_path: str, fmt: str, imgsz: int = 320) -> Path:
    """`fmt` é `onnx` ou `tflite`. O TFLite pede TensorFlow instalado."""
    if fmt not in {"onnx", "tflite"}:
        raise ValueError("Formato aceito: onnx ou tflite.")
    from ultralytics import YOLO

    logger.info("Exportando %s para %s (imgsz=%s).", model_path, fmt, imgsz)
    model = YOLO(model_path)
    exported = model.export(format=fmt, imgsz=imgsz)
    path = Path(str(exported))
    logger.info("Modelo exportado em %s", path)
    return path
