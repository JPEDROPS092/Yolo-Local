"""Direção do obstáculo em relação ao caminho à frente."""

from __future__ import annotations


def classify_direction(center_x: float, image_width: float, center_margin: float = 0.15) -> str:
    """Classifica o centro da caixa como left, center ou right.

    `center_margin` é a fração da largura, para cada lado do meio, que ainda
    conta como caminho à frente. 0.15 deixa uma faixa central de 30%.
    """
    if image_width <= 0:
        return "center"
    margin = max(0.0, min(0.49, float(center_margin)))
    left_bound = image_width * (0.5 - margin)
    right_bound = image_width * (0.5 + margin)
    if center_x < left_bound:
        return "left"
    if center_x > right_bound:
        return "right"
    return "center"


def box_center_x(box: tuple[int, int, int, int]) -> float:
    return (float(box[0]) + float(box[2])) / 2.0
