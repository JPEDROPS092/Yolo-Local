"""Desenho das caixas, distância, direção e cor de perigo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from obstaclesense.classes import spoken_name
from obstaclesense.models import Obstacle

_COLORS_BGR = {
    "high": (40, 40, 230),
    "medium": (0, 165, 255),
    "low": (40, 180, 60),
}
_DIR_PT = {"left": "esquerda", "center": "frente", "right": "direita"}
_LEVEL_PT = {"high": "alto", "medium": "médio", "low": "baixo"}
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def draw_overlay(
    frame: np.ndarray,
    obstacles: list[Obstacle],
    guidance: str,
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    """Copia o quadro e escreve a leitura de cada obstáculo."""
    canvas = frame.copy()
    for obstacle in obstacles:
        color = _COLORS_BGR.get(obstacle.danger_level, (255, 255, 255))
        x1, y1, x2, y2 = obstacle.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    label_font = _font(16)
    hud_font = _font(18)
    used_tags: list[tuple[int, int, int]] = []
    for obstacle in obstacles:
        color = _COLORS_BGR.get(obstacle.danger_level, (255, 255, 255))
        rgb_color = (color[2], color[1], color[0])
        _article, name = spoken_name(obstacle.label)
        text = (
            f"{name} {obstacle.distance_meters:.1f} m  "
            f"{_DIR_PT.get(obstacle.direction, obstacle.direction)}  "
            f"{_LEVEL_PT.get(obstacle.danger_level, obstacle.danger_level)}"
        )
        if obstacle.approaching:
            text += "  aproximando"
        x1, y1, _x2, _y2 = obstacle.box
        text_w = int(draw.textlength(text, font=label_font))
        text_x = min(max(0, x1), max(0, image.width - text_w - 6))
        text_y = max(0, y1 - 22)
        for _ in range(8):
            if not any(_tags_overlap(text_x, text_y, text_w, prev) for prev in used_tags):
                break
            text_y += 22
        used_tags.append((text_x, text_y, text_w))
        _draw_tag(draw, text, text_x, text_y, rgb_color, label_font)
    hud = f"FPS {fps:.1f}   {latency_ms:.0f} ms"
    _draw_tag(draw, hud, 8, 8, (255, 255, 255), hud_font, fill=(20, 20, 20))
    if guidance:
        _draw_banner(draw, guidance, image.width, image.height, hud_font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _tags_overlap(x: int, y: int, width: int, previous: tuple[int, int, int]) -> bool:
    px, py, pw = previous
    horizontal = x < px + pw and px < x + width
    vertical = abs(y - py) < 20
    return horizontal and vertical


def _draw_tag(draw, text, x, y, color, font, fill=(0, 0, 0)) -> None:
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=fill)
    draw.text((x, y), text, font=font, fill=color)


def _draw_banner(draw, text, width, height, font) -> None:
    margin = 8
    max_width = width - 2 * margin
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = 22
    block_h = line_h * len(lines) + 12
    top = max(0, height - block_h - 8)
    draw.rectangle((4, top, width - 4, top + block_h), fill=(10, 10, 10))
    for index, line in enumerate(lines):
        draw.text((margin, top + 6 + index * line_h), line, font=font, fill=(255, 255, 255))
