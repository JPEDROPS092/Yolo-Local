"""Profundidade monocular opcional com Depth Anything V2 Small."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DepthAnythingEstimator:
    """Carrega o modelo só na primeira imagem, para o modo padrão continuar leve."""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._ensure()
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        output = self._pipe(Image.fromarray(rgb))
        predicted = output.get("predicted_depth")
        if predicted is not None:
            depth = predicted.squeeze().detach().float().cpu().numpy()
        else:
            depth = np.array(output["depth"], dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        height, width = frame_bgr.shape[:2]
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        return depth

    def _ensure(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline

        device = -1 if self.device == "cpu" else 0
        logger.info("Carregando profundidade %s em %s.", self.model_id, self.device)
        self._pipe = pipeline(task="depth-estimation", model=self.model_id, device=device)


def sample_relative_depth(depth_map: np.ndarray, box: tuple[int, int, int, int]) -> float | None:
    """Mediana da faixa inferior da caixa, onde o obstáculo costuma estar mais perto."""
    if depth_map is None:
        return None
    height, width = depth_map.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height - 1, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    y_lo = y1 + int(0.55 * (y2 - y1))
    x_lo = x1 + int(0.25 * (x2 - x1))
    x_hi = x1 + int(0.75 * (x2 - x1))
    roi = depth_map[y_lo:y2, x_lo:x_hi]
    if roi.size == 0:
        roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return float(np.median(roi))
